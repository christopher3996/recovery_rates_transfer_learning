# Enable full heterogeneous features (shared feature set and then can add or reduce other features)

import copy
import os
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from models.base_model import BaseModel


# --- Mixture Density Network Head ---
class MDNHead(nn.Module):
    """
    Mixture Density Network (MDN) head.

    Converts a d_token-dimensional representation into parameters of a
    Gaussian mixture:
      - alpha: mixture weights
      - mu: component means
      - sigma: component std deviations

    Parameters
    ----------
    d_token : int
        Dimension of input features.
    n_components : int, default=3
        Number of mixture components.
    hidden_size : int, default=32
        Size of hidden layer.
    temperature : float, default=1.0
        Initial temperature for softmax.
    """
    def __init__(self, d_token, n_components=3, hidden_size=32, temperature=1.0):
        super().__init__()
        self.n_components = n_components
        self.temperature = nn.Parameter(torch.tensor(temperature))
        # Hidden MLP
        self.fc1 = nn.Linear(d_token, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.ReLU()
        # Output linear -> 3 values per component: [logit, pre-sigmoid mean, pre-softplus sigma]
        self.fc2 = nn.Linear(hidden_size, n_components * 3)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, d_token)
            Input features.

        Returns
        -------
        alpha : torch.Tensor, shape (batch, n_components)
        mu : torch.Tensor, shape (batch, n_components)
        sigma : torch.Tensor, shape (batch, n_components)
        """
        # Hidden projection
        h = self.act(self.norm(self.fc1(x)))
        # Reshape to [batch, components, 3]
        out = self.fc2(h).view(x.size(0), self.n_components, 3)
        # Mixture weights with temperature-scaled softmax
        alpha = F.softmax(out[:, :, 0] * self.temperature, dim=1)
        # Means bounded to [0, 1.2]
        mu = torch.sigmoid(out[:, :, 1]) * 1.2
        # Positive std deviations (softplus ensures >0)
        sigma = F.softplus(out[:, :, 2]) + 1e-3
        return alpha, mu, sigma


def mdn_loss(alpha, mu, sigma, y):
    """
    Computes negative log-likelihood of target y under the predicted Gaussian mixture.

    Args:
      alpha: [batch, comps] mixture weights
      mu:    [batch, comps] component means
      sigma: [batch, comps] component stddevs
      y:     [batch, 1]    true targets
    Returns:
      Scalar mean NLL loss
    """
    # Expand y to [batch, comps]
    y_exp = y.expand_as(mu)
    # Gaussian PDF formula
    coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = torch.exp(-0.5 * ((y_exp - mu) / sigma) ** 2)
    pdf = coeff * exponent
    # Weighted sum across mixture
    weighted = alpha * pdf
    # Negative log of likelihood
    nll = -torch.log(weighted.sum(dim=1) + 1e-8)
    return nll.mean()


def bounded_mdn_loss(alpha, mu, sigma, y,
                     lambda_center=0.03,
                     lambda_band=0.02,
                     lambda_entropy=0.01):
    """
    MDN loss with regularization:
      1) Negative log-likelihood (NLL)
      2) Center penalty: discourage all means collapsing near 0.5
      3) Band penalty: encourage one mean in [0.0,0.2] and another in [0.6,0.8]
      4) Entropy bonus: promote weight diversity

    Args:
      lambda_center:  weight for center collapse penalty
      lambda_band:    weight for band penalty
      lambda_entropy: weight for entropy bonus (subtracted)
    """
    nll = mdn_loss(alpha, mu, sigma, y)
    # 2) center collapse penalty
    center_pen = torch.exp(-((mu - 0.5) / 0.08) ** 2).mean()
    # 3) band prior penalty
    mu_sorted, _ = torch.sort(mu, dim=1)
    low_pen = F.relu(0.0 - mu_sorted[:, 0]) + F.relu(mu_sorted[:, 0] - 0.2)
    high_pen = F.relu(0.6 - mu_sorted[:, 1]) + F.relu(mu_sorted[:, 1] - 0.8)
    band_pen = (low_pen + high_pen).mean()
    # 4) entropy bonus
    entropy = - (alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()
    return nll + lambda_center * center_pen + lambda_band * band_pen - lambda_entropy * entropy

# --- Transformer Encoder Block ---
class FTTransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block:
      - Multi-head self-attention + dropout + residual + layernorm
      - Feed-forward network + dropout + residual + layernorm

    Args:
      d_token: dimension of token embeddings
      n_heads: number of attention heads
      ff_factor: expansion factor for feed-forward layer
      dropout: dropout probability
    """
    def __init__(self, d_token, n_heads, ff_factor=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            nn.Linear(d_token, int(ff_factor * d_token)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(ff_factor * d_token), d_token)
        )
        self.ff_drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_token)

    def forward(self, x, *, key_padding_mask=None):
        """
        Args:
          x: [batch, n_tokens, d_token]
        Returns:
          x processed by attention and feed-forward
        """
        attn_out, _ = self.attn(x, x, x,
                               key_padding_mask=key_padding_mask
                               )
        x = self.attn_norm(x + self.attn_drop(attn_out))
        ff_out = self.ff(x)
        return self.ff_norm(x + self.ff_drop(ff_out))



# --- FT-Transformer Backbone ---
class FTTransformerBackbone(nn.Module):
    """
    FT-Transformer backbone:
      1) Embed numeric features individually
      2) Embed categorical features
      3) Prepend CLS token for pooling
      4) Optional cross-token interaction
      5) Stack transformer encoder blocks
      6) MDN head for output

    Args:
      n_numeric: number of numeric input features
      cat_cardinalities: list of cardinalities for categorical features
      d_token, n_heads, n_blocks, ff_factor, dropout: transformer params
      mdn_n_components, mdn_hidden_size, mdn_temperature: MDN head params
      use_interaction: whether to apply a token interaction layer before transformers
    """
    def __init__(self,
                n_numeric_shared, n_numeric_task,
                cat_cards_shared, cat_cards_task,
                d_token=32,
                n_heads=4,
                n_blocks=2,
                ff_factor=4.0,
                dropout=0.1,
                mdn_n_components=2,
                mdn_hidden_size=32,
                mdn_temperature=1.5,
                use_interaction=False
        ):
        super().__init__()
        self.d_token = d_token
        self.mask_task = True

        def _embed_num() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(1, d_token),
                nn.LayerNorm(d_token),
                nn.ReLU(),
            )

        # shared embeddings
        self.shared_num_emb = nn.ModuleList(
            [_embed_num() for _ in range(n_numeric_shared)]
        )
        self.shared_cat_emb = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in cat_cards_shared]
        )

        # task embeddings
        self.task_num_emb = nn.ModuleList(
            [_embed_num() for _ in range(n_numeric_task)]
        )
        self.task_cat_emb = nn.ModuleList(
            [nn.Embedding(c, d_token) for c in cat_cards_task]
        )

        self.pad_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.use_interaction = use_interaction

        if use_interaction:
            self.interaction = nn.Sequential(
                nn.Linear(d_token, d_token),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token, d_token),
                nn.LayerNorm(d_token),
            )

        self.blocks = nn.ModuleList([
            FTTransformerEncoderBlock(d_token, n_heads, ff_factor,
                                      dropout)
            for _ in range(n_blocks)
        ])
        self.head = MDNHead(
            d_token,
            n_components=mdn_n_components,
            hidden_size=mdn_hidden_size,
            temperature=mdn_temperature,
        )

    def rebuild_task_banks(
        self,
        n_numeric_task: int,
        cat_cards_task: List[int],
    ) -> None:
        """
        Reset task-specific embeddings.
        """
        def _embed_num() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(1, self.d_token),
                nn.LayerNorm(self.d_token),
                nn.ReLU(),
            )

        self.task_num_emb = nn.ModuleList(
            [_embed_num() for _ in range(n_numeric_task)]
        )
        self.task_cat_emb = nn.ModuleList(
            [nn.Embedding(c, self.d_token) for c in cat_cards_task]
        )

    


    def forward(
        self,
        x_num_shared: Optional[torch.Tensor] = None, # [B, k_num]
        x_num_task:   Optional[torch.Tensor] = None, # [B, n_num]   (may be empty)
        x_cat_shared: Optional[torch.Tensor] = None, # [B, k_cat]
        x_cat_task:   Optional[torch.Tensor] = None, # [B, n_cat]   (may be empty)   
        ):
        """
        Forward pass with *fixed* shared tokens and *variable* task tokens.
    
        Parameters
        ----------
        x_num_shared : numeric features that exist in **every** dataset
        x_num_task   : numeric features that can change in count (0-N)
        x_cat_shared : categorical features that exist in **every** dataset
        x_cat_task   : categorical features that can change in count (0-N)
    
        All tensors are optional.  Shape conventions:
            numeric  → [batch, num_cols]
            categorical → [batch, num_cols]  (already int-encoded)
        """
        first_tensor = next(t for t in 
                            (x_num_shared, x_num_task, x_cat_shared, x_cat_task) 
                            if t is not None)
        B = first_tensor.size(0)
    
        # ------------------------------------------------------------------ tokens & masks
        tokens, skip_masks = [], []                        # masks: True = **pad**, False = keep
    
        # 1) ── shared numeric ─────────────────────────────────────────────
        if x_num_shared is not None:
            for i, emb in enumerate(self.shared_num_emb):
                tok = emb(x_num_shared[:, i:i+1]).unsqueeze(1)
                tokens.append(tok)
                # shared features are never masked → all-False
                skip_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=tok.device))
            
        # 2) ── shared categorical ─────────────────────────────────────────
        if x_cat_shared is not None:
            for i, emb in enumerate(self.shared_cat_emb):
                tok = emb(x_cat_shared[:, i]).unsqueeze(1)
                tokens.append(tok)
                # shared features are never masked → all-False
                skip_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=tok.device))


        # 3) ── task numeric (may be 0) ────────────────────────────────────
        n_num_present = 0 if x_num_task is None else x_num_task.size(1)
        for i, emb in enumerate(self.task_num_emb):
            if i < n_num_present:                          # real column
                tok  = emb(x_num_task[:, i:i+1]).unsqueeze(1)
                # always keep real task features
                mask = torch.zeros(B, 1, dtype=torch.bool, device=tok.device)  # keep
            else:                                          # padded placeholder
                tok  = self.pad_token.expand(B, 1, -1)
                # mask out only if mask_task=True, else keep
                mask = (torch.ones if self.mask_task else torch.zeros)(
                           B, 1, dtype=torch.bool, device=tok.device)
            tokens.append(tok)
            skip_masks.append(mask)
        
        # 4) ── task categorical (may be 0) ────────────────────────────────
        n_cat_present = 0 if x_cat_task is None else x_cat_task.size(1)
        for i, emb in enumerate(self.task_cat_emb):
            if i < n_cat_present:
                tok  = emb(x_cat_task[:, i]).unsqueeze(1)
                # always keep real task features
                mask = torch.zeros(B, 1, dtype=torch.bool, device=tok.device)
            else:
                tok  = self.pad_token.expand(B, 1, -1)
                # mask out only if mask_task=True, else keep
                mask = (torch.ones if self.mask_task else torch.zeros)(
                           B, 1, dtype=torch.bool, device=tok.device)
            tokens.append(tok)
            skip_masks.append(mask)
        
        # ------------------------------------------------------------------ stack sequence
        x      = torch.cat(tokens, 1)                               # [B, L, d]
        if skip_masks:                                              # could be empty
            skip = torch.cat(skip_masks, 1)                         # [B, L]
        else:
            skip = torch.zeros(B, x.size(1), dtype=torch.bool, device=x.device)

        # prepend CLS  ------------------------------------------------------
        cls = self.cls_token.expand(B, -1, -1)               # [B,1,d]
        x   = torch.cat([cls, x], dim=1)                     # [B, L+1, d]
        skip = torch.cat([torch.zeros(B, 1, device=skip.device, dtype=torch.bool), skip], 1)
    
        # optional cross-token interaction  -------------------------------
        if self.use_interaction:
            x = x + self.interaction(x)
    
        # transformer blocks  ---------------------------------------------
        for blk in self.blocks:
            # MultiHeadAttention in the block uses key_padding_mask
            x = blk(x, key_padding_mask=skip)
    
        # ------------------------------------------------------------------ pooling & head
        pooled            = x[:, 0]                          # CLS
        alpha, mu, sigma  = self.head(pooled)
        y_pred            = (alpha * mu).sum(1, keepdim=True)
        return y_pred, (alpha, mu, sigma)
    
    

class FTTransformerModel(BaseModel):
    """
    End-to-end wrapper for FT-Transformer regression with MDN output.

    Responsibilities:
      - Auto-detect numeric and categorical columns from a sample DataFrame
      - Instantiate backbone, optimizer, and optional scheduler
      - Run training/validation loops
      - Track and store per-epoch metrics:
        * train_loss
        * val_loss
        * val_r2
        * val_means (mixture component means)
        * temperature_history
        * best_epoch and best_val_r2
    """
    def __init__(
        self,
        sample_df:        pd.DataFrame,
        *,
        # ── schema info -------------------------------------------------
        shared_num_cols: Optional[List[str]] = None,   # numeric cols always present
        shared_cat_cols: Optional[List[str]] = None,   # categorical cols always present
        # ── hyper-parameters -------------------------------------------
        d_token:            int   = 32,
        n_heads:            int   = 4,
        n_blocks:           int   = 2,
        ff_factor:          float = 4.0,
        dropout:            float = 0.1,
        learning_rate:      float = 1e-3,
        weight_decay:       float = 1e-3,
        epochs:             int   = 10,
        batch_size:         int   = 32,
        seed:               int   = 42,
        deterministic:      bool  = True,
        benchmark:          bool  = False,
        verbose:            bool  = True,
        mdn_n_components:   int   = 2,
        mdn_hidden_size:    int   = 32,
        mdn_temperature:    float = 1.5,
        use_interaction:    bool  = False,
        use_scheduler:      bool  = True,
        early_stopping:     bool  = True,
        **kwargs,
    ):
        super().__init__(seed=seed,
                         deterministic=deterministic,
                         benchmark=benchmark)

        # ------------------------------------------------------------------
        # 0)  basic bookkeeping / device
        # ------------------------------------------------------------------
        self.seed        = seed
        self.device      = torch.device('cuda' if torch.cuda.is_available()
                                                     else 'cpu')
        self.verbose     = verbose
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.criterion   = bounded_mdn_loss

        # ------------------------------------------------------------------
        # 1)  split sample_df columns into   shared   vs   task
        # ------------------------------------------------------------------
        shared_num_cols = shared_num_cols or []
        shared_cat_cols = shared_cat_cols or []

        all_num = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        all_cat = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()

        # verify user input
        missing  = set(shared_num_cols + shared_cat_cols) - set(all_num + all_cat)
        if missing:
            raise ValueError(f"Shared columns not found in sample_df: {missing}")

        self.shared_num_cols = shared_num_cols
        self.shared_cat_cols = shared_cat_cols
        self.task_num_cols   = [c for c in all_num if c not in shared_num_cols]
        self.task_cat_cols   = [c for c in all_cat if c not in shared_cat_cols]

        self.num_cols = self.shared_num_cols + self.task_num_cols
        self.cat_cols = self.shared_cat_cols + self.task_cat_cols

        # ------------------------------------------------------------------
        # 2)  build categorical lookup tables & cardinalities
        # ------------------------------------------------------------------
        self.cat_maps: dict[str, dict[str, int]] = {}
        cat_cards_shared, cat_cards_task = [], []

        # shared cats -------------------------------------------------------
        for col in self.shared_cat_cols:
            uniq = pd.Series(sample_df[col].astype(str).unique(), dtype="string")
            mapping = {v: i for i, v in enumerate(uniq)}
            mapping["<UNK>"] = len(uniq)
            self.cat_maps[col] = mapping
            cat_cards_shared.append(len(mapping))

        # task cats ---------------------------------------------------------
        for col in self.task_cat_cols:
            uniq = pd.Series(sample_df[col].astype(str).unique(), dtype="string")
            mapping = {v: i for i, v in enumerate(uniq)}
            mapping["<UNK>"] = len(uniq)
            self.cat_maps[col] = mapping
            cat_cards_task.append(len(mapping))

        # ------------------------------------------------------------------
        # 3)  instantiate backbone
        # ------------------------------------------------------------------
        self.model = FTTransformerBackbone(
            n_numeric_shared = len(self.shared_num_cols),
            n_numeric_task   = len(self.task_num_cols),
            cat_cards_shared = cat_cards_shared,
            cat_cards_task   = cat_cards_task,
            d_token          = d_token,
            n_heads          = n_heads,
            n_blocks         = n_blocks,
            ff_factor        = ff_factor,
            dropout          = dropout,
            mdn_n_components = mdn_n_components,
            mdn_hidden_size  = mdn_hidden_size,
            mdn_temperature  = mdn_temperature,
            use_interaction  = use_interaction
        ).to(self.device)

        # ------------------------------------------------------------------
        # 4)  optimiser + optional scheduler
        # ------------------------------------------------------------------
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        if use_scheduler:
            steps_per_epoch = max(1, int(len(sample_df) / batch_size))
            total_steps     = epochs * steps_per_epoch
            self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                                  self.optimizer, T_max=total_steps, eta_min=1e-5)
        else:
            self.scheduler = None

        # ------------------------------------------------------------------
        # 5)  metric histories / early-stopping
        # ------------------------------------------------------------------
        self.train_loss, self.val_loss  = [], []
        self.train_r2,  self.val_r2     = [], []
        self.val_means, self.temperature_history = [], []
        self.train_jsd, self.val_jsd   = [], []
        self.best_epoch, self.best_val_r2 = None, -float('inf')

        self.early_stopping = early_stopping
        self._early_stop_counter = 0
        self.patience       = 15
        self.min_delta      = 1e-4


    def _to_tensor(self, data, dtype):
        """
        Convert input array or pandas structure to torch.Tensor.
        Logs shape and dtype if verbose.
        """
        arr = data.values if hasattr(data, 'values') else data
        t = torch.tensor(arr, dtype=dtype)
        if self.verbose:
            print(f"Converted to tensor: shape={t.shape}, dtype={dtype}")
        return t

    def _encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace string categories with integer codes.
        Unknown or NaN → code of the '<UNK>' token.
        """
        df = df.copy()
        for col in self.shared_cat_cols + self.task_cat_cols:
            mapping = self.cat_maps[col]
            unk = mapping["<UNK>"]
            # vectorised map → pandas Series of ints
            df[col] = (
                df[col]
                .astype("string")
                .map(mapping)              # unseen → NaN
                .fillna(unk)               # NaN or new category
                .astype("int64")
            )
        return df


    def _compute_jsd_divergence(self, p_data, q_data, n_bins=30, epsilon=1e-9):
        """
        Compute the Jensen-Shannon Divergence (JSD) between two distributions 
        using histogram binning.
    
        Args:
            p_data: array-like, first distribution (e.g. y_true)
            q_data: array-like, second distribution (e.g. y_pred)
            n_bins: number of bins for histogram
            epsilon: small value to avoid log(0)
    
        Returns:
            jsd: scalar float, Jensen-Shannon divergence
        """
        p_counts, bin_edges = np.histogram(p_data, bins=n_bins, density=True)
        q_counts, _ = np.histogram(q_data, bins=bin_edges, density=True)
    
        p = p_counts + epsilon
        q = q_counts + epsilon
        p /= p.sum()
        q /= q.sum()
        
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        jsd = 0.5 * (kl_pm + kl_qm)
        return float(jsd)



    def _plot_histogram(self, y_train, y_train_pred, y_val, y_val_pred, epoch=None):
        """
        Save side-by-side histograms of train/val predictions vs. truth as PDF.
        Each saved with a unique name like 'epoch_005_hist.pdf' in a subfolder.
        """
        
        # Create folder (once)
        save_dir = "histograms"
        os.makedirs(save_dir, exist_ok=True)
    
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        #fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharey=False)
    
        # Left: Train
        axs[0].hist(y_train_pred, bins=40, alpha=0.5, edgecolor='C0', label='Prediction', density=True)
        axs[0].hist(y_train, bins=40, histtype='step', edgecolor='black', linewidth=1.5, label='Truth', density=True)
        axs[0].set_title('Train Set')
        axs[0].legend()
        axs[0].grid(True)
        #axs[0].set_yscale('log')
        axs[0].set_ylim(0, 10)
    
        # Right: Validation
        axs[1].hist(y_val_pred, bins=40, alpha=0.5, edgecolor='C0', label='Prediction', density=True)
        axs[1].hist(y_val, bins=40, histtype='step', edgecolor='black', linewidth=1.5, label='Truth', density=True)
        axs[1].set_title('Validation Set')
        axs[1].legend()
        axs[1].grid(True)
        #axs[1].set_yscale('log')
        axs[1].set_ylim(0, 10)
    
        plt.tight_layout()
    
        # Filename with epoch and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if epoch is None:
            filename = f"hist_final_{timestamp}.pdf"
        else:
            filename = f"epoch_{epoch:03d}_hist_{timestamp}.pdf"
    
        fig.savefig(os.path.join(save_dir, filename))
        plt.show()
        #plt.close(fig)
   


    # ----------------------------------------------------------------------
    def _train_loop(
            self,
            X_train: pd.DataFrame, y_train: pd.Series,
            X_val:   pd.DataFrame, y_val:   pd.Series,
            *,
            start_epoch: int = 0,
    ):
        """
        One complete training/validation cycle of `self.epochs` epochs.
    
        Splits every dataframe into
            * shared_num,  task_num
            * shared_cat,  task_cat
        and feeds them to the backbone.  Works even if a task split has 0 cols.
        """
        # ────────────────────────────────────────────────────────── encode cats
        X_train = self._encode_categories(X_train)
        X_val   = self._encode_categories(X_val)
    
        # ------------------------------------------------------------------ helpers
        def _tensor(df, cols, dtype):
            if cols:                                          # ≥ 1 column
                arr = df[cols].values
                t   = torch.tensor(arr, dtype=dtype)
            else:                                             # 0 columns → empty (B,0)
                t = torch.empty(len(df), 0, dtype=dtype)
            return t.to(self.device)
    
        # ------------------------------------------------------------------ make tensors (train)
        Xtr_num_sh = _tensor(X_train, self.shared_num_cols, torch.float32)
        Xtr_num_ta = _tensor(X_train, self.task_num_cols,   torch.float32)
        Xtr_cat_sh = _tensor(X_train, self.shared_cat_cols, torch.int64)
        Xtr_cat_ta = _tensor(X_train, self.task_cat_cols,   torch.int64)


        # y_train / y_val can be either pd.Series or np.ndarray
        def _np(a):                          # -> 1-D NumPy array
            return a.values if hasattr(a, "values") else a
            
        Ytr = torch.tensor(_np(y_train), dtype=torch.float32, device=self.device).unsqueeze(1)

    
        train_ds = torch.utils.data.TensorDataset(
            Xtr_num_sh, Xtr_num_ta, Xtr_cat_sh, Xtr_cat_ta, Ytr)
    
        g = torch.Generator().manual_seed(self.seed)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, generator=g,
            worker_init_fn=lambda wid: np.random.seed(self.seed + wid))
    
        # ------------------------------------------------------------------ tensors (val)
        Xv_num_sh = _tensor(X_val, self.shared_num_cols, torch.float32)
        Xv_num_ta = _tensor(X_val, self.task_num_cols,   torch.float32)
        Xv_cat_sh = _tensor(X_val, self.shared_cat_cols, torch.int64)
        Xv_cat_ta = _tensor(X_val, self.task_cat_cols,   torch.int64)
        Yv  = torch.tensor(_np(y_val), dtype=torch.float32, device=self.device).unsqueeze(1)
    
        # ------------------------------------------------------------------ epoch loop
        epoch_iter = tqdm(range(start_epoch, start_epoch + self.epochs),
                          desc="Training epochs (FT-MDN-Transformer)")
    
        for global_ep in epoch_iter:
            ep = global_ep - start_epoch
    
            # ─── training ───────────────────────────────────────────────────
            self.model.train()
            running = 0.0
            for xb_s, xb_t, xc_s, xc_t, yb in train_loader:
                self.optimizer.zero_grad()
                y_hat, (a, m, s) = self.model(xb_s, xb_t, xc_s, xc_t)
                loss = self.criterion(a, m, s, yb)
                loss.backward()
                self.optimizer.step()
                if self.scheduler: 
                    self.scheduler.step()
                running += loss.item() * yb.size(0)
            train_loss = running / len(train_ds)
    
            # ─── evaluate on train set (R²) ────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                y_pred_tr = self.model(Xtr_num_sh, Xtr_num_ta,
                                       Xtr_cat_sh, Xtr_cat_ta)[0].cpu().numpy().ravel()
            train_r2 = r2_score(y_train, y_pred_tr)
    
            # ─── validation ────────────────────────────────────────────────
            with torch.no_grad():
                _, (a_v, m_v, s_v) = self.model(Xv_num_sh, Xv_num_ta,
                                                Xv_cat_sh, Xv_cat_ta)
                val_loss = self.criterion(a_v, m_v, s_v, Yv).item()
                y_pred_v = (a_v * m_v).sum(1).cpu().numpy()
            val_r2 = r2_score(y_val, y_pred_v)
    
            # optional distribution metric
            train_jsd = self._compute_jsd_divergence(y_train, y_pred_tr)
            val_jsd = self._compute_jsd_divergence(y_val,   y_pred_v)
    
            # ─── bookkeeping ───────────────────────────────────────────────
            self.train_loss.append(train_loss);  self.val_loss.append(val_loss)
            self.train_r2.append(train_r2);      self.val_r2.append(val_r2)
            self.train_jsd.append(train_jsd);    self.val_jsd.append(val_jsd)
            self.val_means.append(m_v.mean(0).cpu().tolist())
            self.temperature_history.append(self.model.head.temperature.item())
    
            # ─── early-stopping ────────────────────────────────────────────
            if self.early_stopping:
                improved = val_r2 > self.best_val_r2 + self.min_delta
                if improved:
                    self.best_val_r2 = val_r2
                    self.best_epoch  = ep + 1
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                    self._early_stop_counter = 0
                else:
                    self._early_stop_counter += 1
                    if self._early_stop_counter >= self.patience:
                        epoch_iter.write(f"✅ Early stop at epoch {ep+1}.")
                        epoch_iter.close()
                        return global_ep    # last epoch actually executed

            # ---- Verbose Logging ----
            if self.verbose:
                comp_means = m_v.mean(dim=0).cpu().numpy().tolist()
                temp       = self.model.head.temperature.item()
                print(
                    f"Epoch {ep+1}/{self.epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"train_R2={train_r2:.4f}, "
                    f"train_jsd={train_jsd:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_R2={val_r2:.4f}, "
                    f"val_jsd={val_jsd:.4f}, "
                    f"means={comp_means}, "
                    f"temp={temp:.4f}"
                )

                self._plot_histogram(y_train, y_pred_tr,
                                     y_val,   y_pred_v, epoch=ep + 1
                                    )
    
            # ─── console output / tqdm postfix ─────────────────────────────
            epoch_iter.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_R2":     f"{val_r2:.4f}",
                "best_R2":    f"{self.best_val_r2:.4f}"
            })
    
        return None   # loop finished all epochs




    def train(self, X_train, y_train, X_val, y_val):
        """
        Train model and record all metrics.

        Clears previous histories before training.
        """
        self.train_loss.clear()
        self.val_loss.clear()
        self.train_r2.clear()
        self.val_r2.clear()
        self.val_means.clear()
        self.temperature_history.clear()
        self.best_epoch = None
        self.best_val_r2 = -float('inf')
        self._early_stop_counter = 0
        
        self.max_train_epoch = self._train_loop(X_train, y_train, X_val, y_val)
        
        self.max_train_epoch = self.epochs if self.max_train_epoch is None else self.max_train_epoch

    def _rebuild_optimizer(self, lr, wd):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)


    # ----------------------------------------------------------------------
    def fine_tune(
            self,
            X_new_train: pd.DataFrame, y_new_train: pd.Series,
            X_new_val:   pd.DataFrame, y_new_val:   pd.Series,
            *,
            additional_epochs: int = 40,
            freeze_backbone:   bool = True,
            freeze_shared:     bool = False,
            lr:                float = 5e-4,
            weight_decay:      float = 1e-4,
    ):
        """
        • Replaces the **task** embedding bank so it matches the columns in
          `X_new_*`.
        • Lets you decide what to freeze / unfreeze.
        • Trains for `additional_epochs`.
        """
        if additional_epochs <= 0:
            raise ValueError("`additional_epochs` must be ≥ 1")

        self.best_epoch = None
        self.best_val_r2 = -float('inf')
    
        # ── 1.  determine current task columns in the new dataset ─────────────
        self.task_num_cols = [c for c in
                              X_new_train.select_dtypes(include=[np.number]).columns
                              if c not in self.shared_num_cols]
        self.task_cat_cols = [c for c in
                              X_new_train.select_dtypes(exclude=[np.number]).columns
                              if c not in self.shared_cat_cols]

        self.num_cols = self.shared_num_cols + self.task_num_cols
        self.cat_cols = self.shared_cat_cols + self.task_cat_cols
    
        # ── 2.  rebuild task embedding banks  ─────────────────────────────────
        cat_cards_task = [(X_new_train[c].nunique() + 1) for c in self.task_cat_cols]
        self.model.rebuild_task_banks(len(self.task_num_cols), cat_cards_task)
    
        # also rebuild cat-lookup tables for *task* categoricals
        for col in self.task_cat_cols:
            uniq = pd.Series(X_new_train[col].astype(str).unique(), dtype="string")
            mapping = {v: i for i, v in enumerate(uniq)}
            mapping["<UNK>"] = len(uniq)
            self.cat_maps[col] = mapping
    
        # ── 3.  set requires_grad flags  ──────────────────────────────────────
        for p in self.model.parameters():
            p.requires_grad_(False)
    
        # 3a) shared embeddings
        if not freeze_shared:
            for emb in list(self.model.shared_num_emb) + list(self.model.shared_cat_emb):
                for p in emb.parameters(): p.requires_grad_(True)
    
        # 3b) new task embeddings
        for emb in list(self.model.task_num_emb) + list(self.model.task_cat_emb):
            for p in emb.parameters(): p.requires_grad_(True)
    
        # 3c) MDN head always fine-tunes
        for p in self.model.head.parameters(): p.requires_grad_(True)
    
        # 3d) backbone blocks
        if not freeze_backbone:
            for blk in self.model.blocks:
                for p in blk.parameters(): p.requires_grad_(True)



    
        # ── 4.  fresh optimiser / scheduler  ──────────────────────────────────
        self._rebuild_optimizer(lr, weight_decay)
    
        steps_per_epoch = max(1, int(len(X_new_train) / self.batch_size))
        total_steps     = additional_epochs * steps_per_epoch
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer, T_max=total_steps, eta_min=1e-5)
    
        # ── 5.  run training loop  ────────────────────────────────────────────
        original_epochs = self.epochs
        self.epochs     = additional_epochs

        # Turn off masking of task‐specific pads during fine-tune:
        self.model.mask_task = False
    
        start_ep = len(self.train_loss)
        self.max_finetune_epoch = self._train_loop(
            X_new_train, y_new_train, X_new_val, y_new_val,
            start_epoch=start_ep)
        if self.max_finetune_epoch is None:
            self.max_finetune_epoch = additional_epochs
    
        self.epochs = original_epochs



    def predict(self, X_test):
        """
        Generate mean-based predictions on new data.

        Args:
          X_test: DataFrame with same numeric/categorical columns
        Returns:
          numpy array of shape [n_samples]
        """
        X_test = self._encode_categories(X_test)
        self.model.eval()
        with torch.no_grad():
            y_pred, _ = self.model(
                torch.tensor(X_test[self.shared_num_cols].values,
                             dtype=torch.float32, device=self.device),
                torch.tensor(X_test[self.task_num_cols].values,
                             dtype=torch.float32, device=self.device) if self.task_num_cols else None,
                torch.tensor(X_test[self.shared_cat_cols].values,
                             dtype=torch.int64,  device=self.device) if self.shared_cat_cols else None,
                torch.tensor(X_test[self.task_cat_cols].values,
                             dtype=torch.int64,  device=self.device) if self.task_cat_cols else None
            )
  
        return y_pred.cpu().numpy().flatten()

    def get_training_history(self):
        """
        Retrieve the recorded training and validation metrics.

        Returns:
          Dictionary with keys:
            train_loss, val_loss, val_r2,
            val_means, temperature_history,
            best_epoch, best_val_r2
        """
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'val_means': self.val_means,
            'temperature_history': self.temperature_history,
            'best_epoch': self.best_epoch,
            'best_val_r2': self.best_val_r2,
            'train_jsd': self.train_jsd,
            'val_jsd': self.val_jsd,
            'max_train_epoch': self.max_train_epoch,
            'max_finetune_epoch': getattr(self, "max_finetune_epoch", None)
        }

    #For Simulation Class
    def get_epoch_metrics(self):
        """
        Get epoch-wise metrics recorded during training and/or fine-tuning.
    
        Returns:
            dict: A dictionary with keys:
                'loss_pretrain', 'r2_pretrain' for the initial training phase.
                'loss_finetune', 'r2_finetune' for the fine-tuning phase.
        """
        metrics = {}
    
        # Pretraining (from 0 to max_train_epoch)
        if self.max_train_epoch is not None:
            metrics['loss_pretrain'] = self.train_loss[:self.max_train_epoch]
            metrics['r2_pretrain']   = self.train_r2[:self.max_train_epoch]
    
        # Finetuning (after max_train_epoch)
        if hasattr(self, 'max_finetune_epoch') and self.max_finetune_epoch:
            start = self.max_train_epoch
            end   = start + self.max_finetune_epoch
            metrics['loss_finetune'] = self.train_loss[start:end]
            metrics['r2_finetune']   = self.train_r2[start:end]
    
        return metrics


    def select_best_model(self):
        """
        Select the model with weights from the best validation epoch.
        """
        if hasattr(self, 'best_model_state_dict') and self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            if self.verbose:
                print(f"Switched to best model from epoch {self.best_epoch} with val_R2={self.best_val_r2:.4f}")
        else:
            print("No best model state saved yet. Run training first.")
