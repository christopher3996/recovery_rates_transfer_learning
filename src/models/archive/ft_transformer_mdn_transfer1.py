# First TL model with Heterogeneous Features

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from models.base_model import BaseModel

from datetime import datetime

# --- Mixture Density Network Head ---
class MDNHead(nn.Module):
    """
    Mixture Density Network (MDN) head.

    Converts a d_token-dimensional representation into parameters of a
    Gaussian mixture:
      - alpha (mixture weights)
      - mu    (component means)
      - sigma (component standard deviations)

    Args:
      d_token:      dimension of input features
      n_components: number of mixture components
      hidden_size:  size of hidden layer
      temperature:  initial temperature for softmax
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

        Args:
          x: Tensor of shape [batch_size, d_token]
        Returns:
          alpha: [batch_size, n_components]
          mu:    [batch_size, n_components]
          sigma: [batch_size, n_components]
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

    def forward(self, x):
        """
        Args:
          x: [batch, n_tokens, d_token]
        Returns:
          x processed by attention and feed-forward
        """
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + self.attn_drop(attn_out))
        ff_out = self.ff(x)
        return self.ff_norm(x + self.ff_drop(ff_out))


# --- Column registry ---
class ColumnRegistry:
    """
    Maps column-names → token indices and lets us append new columns later.
    """
    def __init__(self, names):
        self.names   = list(names)
        self.indices = {n: i for i, n in enumerate(names)}

    def add(self, new_names):
        """Register unseen columns, return their indices (in same order)."""
        out = []
        for n in new_names:
            if n not in self.indices:
                self.indices[n] = len(self.names)
                self.names.append(n)
            out.append(self.indices[n])
        return out


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
    def __init__(
        self,
        n_numeric,
        cat_cardinalities,
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
        # Numeric embedding MLPs
        self.num_emb = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_token),
                nn.LayerNorm(d_token),
                nn.ReLU()
            ) for _ in range(n_numeric)
        ])
        # Categorical embedding layers
        self.cat_emb = None
        if cat_cardinalities:
            self.cat_emb = nn.ModuleList([nn.Embedding(card, d_token)
                                          for card in cat_cardinalities])
        # CLS token for sequence pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        # Optional interaction layer
        self.use_interaction = use_interaction
        if use_interaction:
            self.interaction = nn.Sequential(
                nn.Linear(d_token, d_token),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_token, d_token),
                nn.LayerNorm(d_token)
            )
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            FTTransformerEncoderBlock(d_token, n_heads, ff_factor, dropout)
            for _ in range(n_blocks)
        ])
        # MDN head for output
        self.head = MDNHead(d_token,
                            n_components=mdn_n_components,
                            hidden_size=mdn_hidden_size,
                            temperature=mdn_temperature)

        self.d_token  = d_token                       # we’ll need the size
        self.registry = ColumnRegistry(
                [f"num_{i}" for i in range(n_numeric)] +
                [f"cat_{i}" for i in range(len(cat_cardinalities))]
        )


    def forward(self, x_num=None, x_cat=None):
        """
        Forward pass.

        Args:
          x_num: [batch, n_numeric]
          x_cat: [batch, n_categorical]
        Returns:
          y_pred: [batch, 1] mixture mean
          (alpha, mu, sigma): full mixture params
        """
        tokens = []
        # Numeric tokens
        if x_num is not None:
            toks = [net(x_num[:, i:i+1]).unsqueeze(1)
                    for i, net in enumerate(self.num_emb)]
            tokens.append(torch.cat(toks, dim=1))
        # Categorical tokens
        if x_cat is not None and self.cat_emb:
            toks = [emb(x_cat[:, i]).unsqueeze(1)
                    for i, emb in enumerate(self.cat_emb)]
            tokens.append(torch.cat(toks, dim=1))
        if not tokens:
            raise ValueError("No input features provided.")
        # Stack tokens into sequence
        x = torch.cat(tokens, dim=1)
        B = x.size(0)
        # Prepend CLS
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        # Interaction
        if self.use_interaction:
            x = x + self.interaction(x)
        # Transformer
        for block in self.blocks:
            x = block(x)
        # Pool CLS rep
        pooled = x[:, 0]
        # MDN output
        alpha, mu, sigma = self.head(pooled)
        # Mean prediction
        y_pred = (alpha * mu).sum(dim=1, keepdim=True)
        return y_pred, (alpha, mu, sigma)


    def add_new_features(self, n_new_num, new_cat_cards):
        """
        Grow num_emb / cat_emb and return *embedding positions*
        (not registry indices!) so that fine_tune can index safely.
        """
        new_num_pos, new_cat_pos = [], []
    
        # numeric ---------------------------------------------------------------
        for _ in range(n_new_num):
            pos = len(self.num_emb)                 # ← actual position
            self.num_emb.append(
                nn.Sequential(nn.Linear(1, self.d_token),
                              nn.LayerNorm(self.d_token),
                              nn.ReLU())
            )
            self.registry.add([f"num_{pos}"])       # still register for completeness
            new_num_pos.append(pos)
    
        # categorical -----------------------------------------------------------
        if new_cat_cards and self.cat_emb is None:  # create the list on-demand
            self.cat_emb = nn.ModuleList()
    
        for card in new_cat_cards:
            pos = len(self.cat_emb)
            self.cat_emb.append(nn.Embedding(card, self.d_token))
            self.registry.add([f"cat_{pos}"])
            new_cat_pos.append(pos)
    
        return new_num_pos, new_cat_pos            # <<< changed



# --- Model Wrapper ---
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
        sample_df: pd.DataFrame,
        d_token=32,
        n_heads=4,
        n_blocks=2,
        ff_factor=4.0,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-3,
        epochs=10,
        batch_size=32,
        seed=42,
        deterministic=True,
        benchmark=False,
        verbose=True,
        mdn_n_components=2,
        mdn_hidden_size=32,
        mdn_temperature=1.5,
        use_interaction=False,
        use_scheduler=True,
        early_stopping=True,
        **kwargs
    ):
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)
        self.seed = seed
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        # Column detection
        self.num_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()

        # ----------------------------------------------------------------------
        # Build categorical lookup tables + decide embedding sizes
        # ----------------------------------------------------------------------
        self.cat_maps = {}          # {col: {category_string: int_code}}
        cat_cardinalities = []      # replaces the old calculation
        
        for col in self.cat_cols:
            # unique categories seen in the *sample* dataframe
            uniq = pd.Series(sample_df[col].astype(str).unique(), dtype="string")
            self.cat_maps[col] = {v: i for i, v in enumerate(uniq)}
            unk_index = len(uniq)              # reserve last index for "unknown"
            self.cat_maps[col]["<UNK>"] = unk_index
            cat_cardinalities.append(unk_index + 1)   # +1 so embedding has room for UNK


        # Build backbone
        #cat_cards = [int(sample_df[c].nunique()) for c in self.cat_cols]
        self.model = FTTransformerBackbone(
            n_numeric=len(self.num_cols),
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            n_heads=n_heads,
            n_blocks=n_blocks,
            ff_factor=ff_factor,
            dropout=dropout,
            mdn_n_components=mdn_n_components,
            mdn_hidden_size=mdn_hidden_size,
            mdn_temperature=mdn_temperature,
            use_interaction=use_interaction
        ).to(self.device)
        # Optimizer & scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = None
        if use_scheduler:
            total_steps = epochs * max(1, int(len(sample_df) / batch_size))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-5)
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = bounded_mdn_loss
        # Metric histories
        self.train_loss = []
        self.val_loss = []
        self.train_r2 = []
        self.val_r2 = []
        self.val_means = []
        self.temperature_history = []
        self.train_jsd = []
        self.val_jsd = []
        self.best_epoch = None
        self.best_val_r2 = -float('inf')

        self.early_stopping = early_stopping 
        self.patience = 15           # Number of epochs to wait for improvement
        self.min_delta = 1e-4        # Minimum change to qualify as improvement


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
        for col in self.cat_cols:
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
        import matplotlib.pyplot as plt
        import os
    
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
   


    def _train_loop(self, X_train, y_train, X_val, y_val, *, start_epoch: int = 0):
        """
        Core training loop:
          - Prepares data
          - Training with NLL loss
          - Evaluates train R2 in eval mode (no dropout)
          - Evaluates validation metrics
          - Records histories and best epoch
          - Verbose logging
        """
        # encode once for this loop
        X_train = self._encode_categories(X_train)
        X_val   = self._encode_categories(X_val)

        # Prepare train set tensors
        Xn = self._to_tensor(X_train[self.num_cols], torch.float32).to(self.device)
        Xc = self._to_tensor(X_train[self.cat_cols], torch.int64).to(self.device)
        Y  = self._to_tensor(y_train, torch.float32).unsqueeze(1).to(self.device)
        train_ds     = torch.utils.data.TensorDataset(Xn, Xc, Y)
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id)
        )

        # Prepare validation tensors
        Xv_n = self._to_tensor(X_val[self.num_cols], torch.float32).to(self.device)
        Xv_c = self._to_tensor(X_val[self.cat_cols], torch.int64).to(self.device)
        Yv   = self._to_tensor(y_val, torch.float32).unsqueeze(1).to(self.device)

        epoch_iter = tqdm(range(start_epoch, start_epoch + self.epochs), desc="Training epochs")
        for global_epoch in epoch_iter:
            epoch = global_epoch - start_epoch      # local counter (0…self.epochs-1)

            # ---- Training Phase ----
            self.model.train()
            total_loss = 0.0
            for xb_num, xb_cat, yb in train_loader:
                xb_num, xb_cat, yb = xb_num.to(self.device), xb_cat.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                y_pred, (alpha, mu, sigma) = self.model(xb_num, xb_cat)
                loss = self.criterion(alpha, mu, sigma, yb)
                loss.backward(); self.optimizer.step()
                if self.scheduler: self.scheduler.step()
                total_loss += loss.item() * xb_num.size(0)
            train_loss = total_loss / len(train_ds)

            # ---- Train R² (eval mode) ----
            self.model.eval()
            with torch.no_grad():
                y_train_pred = self.model(Xn, Xc)[0].cpu().numpy().flatten()
            train_r2 = r2_score(y_train, y_train_pred)

            # ---- Validation Phase ----
            with torch.no_grad():
                _, (alpha_v, mu_v, sigma_v) = self.model(Xv_n, Xv_c)
                val_loss = self.criterion(alpha_v, mu_v, sigma_v, Yv).item()
                y_val_pred = (alpha_v * mu_v).sum(dim=1).cpu().numpy()
            val_r2 = r2_score(y_val, y_val_pred)

            train_jsd = self._compute_jsd_divergence(y_train, y_train_pred)
            val_jsd   = self._compute_jsd_divergence(y_val, y_val_pred)
            self.train_jsd.append(train_jsd)
            self.val_jsd.append(val_jsd)


            # ---- Record Metrics ----
            self.train_loss.append(train_loss)
            self.train_r2.append(train_r2)
            self.val_loss.append(val_loss)
            self.val_r2.append(val_r2)
            self.val_means.append(mu_v.mean(dim=0).cpu().numpy().tolist())
            self.temperature_history.append(self.model.head.temperature.item())

            # ---- Best Epoch Tracking and Early Stopping ----
            if self.early_stopping:
                if val_r2 > self.best_val_r2 + self.min_delta:
                    self.best_val_r2 = val_r2
                    self.best_epoch = epoch + 1
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                    self._early_stop_counter = 0
                else:
                    self._early_stop_counter += 1
                    if self._early_stop_counter >= self.patience:
                        epoch_iter.write(f"✅ Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs).")
                        epoch_iter.close()  # prevents red bar
                        #print(f"Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs).")
                        return global_epoch
                        #break


            # ---- Verbose Logging ----
            if self.verbose:
                comp_means = mu_v.mean(dim=0).cpu().numpy().tolist()
                temp       = self.model.head.temperature.item()
                print(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"train_R2={train_r2:.4f}, "
                    f"train_jsd={train_jsd:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_R2={val_r2:.4f}, "
                    f"val_jsd={val_jsd:.4f}, "
                    f"means={comp_means}, "
                    f"temp={temp:.4f}"
                )
                self._plot_histogram(y_train, y_train_pred, y_val, y_val_pred, epoch=epoch + 1)

            epoch_iter.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_R2": f"{val_r2:.4f}",
                "best_R2": f"{self.best_val_r2:.4f} at Epoch {self.best_epoch}"
            })


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
        self.max_train_epoch = self._train_loop(X_train, y_train, X_val, y_val)
        self.max_train_epoch = self.epochs if self.max_train_epoch is None else self.max_train_epoch


    def fine_tune(
            self,
            X_new_train: pd.DataFrame, y_new_train: pd.Series,
            X_new_val:   pd.DataFrame, y_new_val:   pd.Series,
            *,
            additional_epochs: int = 10,
            freeze_backbone: bool = True,
            freeze_shared: bool = False,
            lr: float = 5e-4,
            weight_decay: float = 1e-4,
        ):
        """
        Warm-start fine-tuning that
          • detects NEW columns,
          • appends fresh embedding layers,
          • (optionally) freezes backbone and/or shared embeddings,
          • re-runs training for `additional_epochs`.
        
        Parameters
        ----------
        additional_epochs : int
            Extra epochs to run.
        freeze_backbone : bool, default=True
            If True, keeps transformer blocks frozen.
        freeze_shared : bool, default=False
            If True, keeps *shared* column embeddings frozen.
        lr, weight_decay : float
            Hyper-params for the new optimiser.
        """
        if additional_epochs <= 0:
            raise ValueError("`additional_epochs` must be ≥ 1")

        self.best_val_r2 = 0
        
        # ── 1. identify NEW columns ────────────────────────────────────────────
        present = set(self.num_cols + self.cat_cols)
        new_num_cols = [c for c in X_new_train.select_dtypes(include=[np.number])
                        if c not in present]
        new_cat_cols = [c for c in X_new_train.select_dtypes(exclude=[np.number])
                        if c not in present]
        
        # ── 2. capture OLD lengths, then expand backbone ───────────────────────
        old_num_len = len(self.model.num_emb)
        old_cat_len = len(self.model.cat_emb) if self.model.cat_emb else 0
        
        new_num_pos, new_cat_pos = self.model.add_new_features(
            n_new_num=len(new_num_cols),
            #new_cat_cards=[int(X_new_train[c].nunique()) for c in new_cat_cols]
            new_cat_cards=[int(X_new_train[c].nunique()) + 1  # +1 for <UNK>
                           for c in new_cat_cols]
        )
        
        # ── 3. extend bookkeeping lists & cat-maps ─────────────────────────────
        self.num_cols.extend(new_num_cols)
        self.cat_cols.extend(new_cat_cols)
        
        for col in new_cat_cols:
            uniq = pd.Series(X_new_train[col].astype(str).unique(), dtype="string")
            mapping = {v: i for i, v in enumerate(uniq)}
            mapping["<UNK>"] = len(uniq)
            self.cat_maps[col] = mapping
        
        # ── 4. set trainable flags ─────────────────────────────────────────────
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        # 4a: shared embeddings
        if not freeze_shared:
            for emb in self.model.num_emb[:old_num_len]:
                for p in emb.parameters(): p.requires_grad_(True)
            if self.model.cat_emb:
                for emb in self.model.cat_emb[:old_cat_len]:
                    for p in emb.parameters(): p.requires_grad_(True)
        
        # 4b: new embeddings
        for pos in new_num_pos:
            for p in self.model.num_emb[pos].parameters():
                p.requires_grad_(True)
        for pos in new_cat_pos:
            for p in self.model.cat_emb[pos].parameters():
                p.requires_grad_(True)
        
        # 4c: MDN head always adapts
        for p in self.model.head.parameters():
            p.requires_grad_(True)
        
        # 4d: optionally unfreeze backbone
        if not freeze_backbone:
            for blk in self.model.blocks:
                for p in blk.parameters():
                    p.requires_grad_(True)
        
        # ── 5. fresh optimiser / scheduler ─────────────────────────────────────
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        
        total_steps = additional_epochs * max(1, int(len(X_new_train) / self.batch_size))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-5)
        
        # ── 6. launch normal train loop ────────────────────────────────────────
        original_epochs = self.epochs
        self.epochs = additional_epochs
        
        start_epoch = len(self.train_loss)          # continue counter
        self.max_finetune_epoch = self._train_loop(X_new_train, y_new_train,
                         X_new_val,   y_new_val,
                         start_epoch=start_epoch)
        self.max_finetune_epoch = additional_epochs if self.max_finetune_epoch is None else self.max_finetune_epoch
        
        self.epochs = original_epochs              # restore for future calls


    def predict(self, X_test):
        """
        Generate mean-based predictions on new data.

        Args:
          X_test: DataFrame with same numeric/categorical columns
        Returns:
          numpy array of shape [n_samples]
        """
        X_test = self._encode_categories(X_test)

        Xn = torch.tensor(X_test[self.num_cols].values, dtype=torch.float32).to(self.device)
        Xc = torch.tensor(X_test[self.cat_cols].values, dtype=torch.int64).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred, _ = self.model(Xn, Xc)
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
