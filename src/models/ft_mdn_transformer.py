# FTTransformerModel - Fully Reworked Version 08/2025

from __future__ import annotations
import copy
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm.auto import tqdm

# ------------------------------ Helpers ------------------------------

def _geglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)

class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    def forward(self, x):
        return _geglu(self.proj(x))

class StochasticDepth(nn.Module):
    """Drop paths per sample (when applied in main path of residual blocks)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask

# -------------------------- Output Heads -----------------------------

class RegressionHead(nn.Module):
    def __init__(self, d_token: int):
        super().__init__()
        self.out = nn.Linear(d_token, 1)
    def forward(self, x):
        return self.out(x), None

class MDNHead(nn.Module):
    """Gaussian Mixture Density head with numerically stable log-sum-exp.

    Args:
        d_token: input dimension
        n_components: number of mixture components
        hidden_size: hidden width
        temperature: initial temperature (softmax over logits uses /temperature)
        learnable_temperature: if True, temperature is a learnable positive scalar
        mu_max: upper bound for means (lower bound is 0 via sigmoid)
        min_sigma: additive floor on stddev for stability
    """
    def __init__(self,
                 d_token: int,
                 n_components: int = 2,
                 hidden_size: int = 32,
                 temperature: float = 1.0,
                 learnable_temperature: bool = True,
                 mu_max: float = 1.0,
                 min_sigma: float = 1e-3):
        super().__init__()
        self.n_components = n_components
        self.mu_max = mu_max
        self.min_sigma = min_sigma
        self.fc1 = nn.Sequential(
            nn.Linear(d_token, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        # 3 params per component: logit, mean_pre, sigma_pre
        self.fc2 = nn.Linear(hidden_size, n_components * 3)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.fc1(x)
        out = self.fc2(h).view(x.size(0), self.n_components, 3)
        logits = out[:, :, 0]
        mu_pre = out[:, :, 1]
        sig_pre = out[:, :, 2]
        tau = torch.clamp(self.log_temperature.exp(), min=1e-3, max=100.0)
        # standard: softmax(logits / tau) → higher tau = softer
        alpha = F.softmax(logits / tau, dim=1)
        mu = torch.sigmoid(mu_pre) * self.mu_max
        sigma = F.softplus(sig_pre) + self.min_sigma
        return alpha, mu, sigma

# Optional: Beta mixture (bounded support) — off by default but available
class BetaMDNHead(nn.Module):
    def __init__(self, d_token: int, n_components: int = 2, hidden_size: int = 32,
                 temperature: float = 1.0, learnable_temperature: bool = True,
                 min_conc: float = 0.5):
        super().__init__()
        self.n_components = n_components
        self.min_conc = min_conc
        self.fc1 = nn.Sequential(
            nn.Linear(d_token, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
        )
        # 3 params per component: logit, alpha_pre, beta_pre (concentrations)
        self.fc2 = nn.Linear(hidden_size, n_components * 3)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))

    def forward(self, x: torch.Tensor):
        h = self.fc1(x)
        out = self.fc2(h).view(x.size(0), self.n_components, 3)
        logits = out[:, :, 0]
        a_pre = out[:, :, 1]
        b_pre = out[:, :, 2]
        tau = torch.clamp(self.log_temperature.exp(), min=1e-3, max=100.0)
        alpha = F.softmax(logits / tau, dim=1)
        # Concentration parameters ≥ min_conc
        conc1 = F.softplus(a_pre) + self.min_conc
        conc2 = F.softplus(b_pre) + self.min_conc
        return alpha, conc1, conc2

# -------------------------- MDN Losses -------------------------------

def mdn_gaussian_nll(alpha: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Numerically stable NLL for Gaussian mixture.
    alpha: [B,K], mu: [B,K], sigma: [B,K], y: [B,1]
    """
    y = y.expand_as(mu)
    # log N(y|mu,sigma)
    log_norm = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    log_mix = torch.log(alpha + 1e-12) + log_norm
    log_sum = torch.logsumexp(log_mix, dim=1)  # [B]
    return -(log_sum.mean())

def mdn_beta_nll(alpha: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # y in [0,1]
    eps = 1e-8
    y = torch.clamp(y.squeeze(1), eps, 1 - eps).unsqueeze(1)
    # log Beta(y | a,b) up to stability using lgamma
    log_pdf = (a - 1) * torch.log(y) + (b - 1) * torch.log(1 - y) \
              - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    log_mix = torch.log(alpha + 1e-12) + log_pdf
    log_sum = torch.logsumexp(log_mix, dim=1)
    return -(log_sum.mean())

# ---------------------- Transformer Backbone ------------------------

class FTTransformerEncoderBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, ff_factor: float = 4.0, dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop_path1 = StochasticDepth(drop_path)

        inner = int(ff_factor * d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            GEGLU(d_token, inner),
            nn.Dropout(dropout),
            nn.Linear(inner, d_token),
        )
        self.drop2 = nn.Dropout(dropout)
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # PreNorm attention
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x_res + self.drop_path1(self.drop1(attn_out))
        # PreNorm FFN
        x_res = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x_res + self.drop_path2(self.drop2(x))
        return x

class FTTransformerBackbone(nn.Module):
    def __init__(self,
                 n_numeric_shared: int, n_numeric_task: int,
                 cat_cards_shared: List[int], cat_cards_task: List[int],
                 d_token: int = 32, n_heads: int = 4, n_blocks: int = 2,
                 ff_factor: float = 4.0, dropout: float = 0.1,
                 drop_path_rate: float = 0.0,
                 use_mdn: bool = True,
                 mdn_n_components: int = 2,
                 mdn_hidden_size: int = 32,
                 mdn_temperature: float = 1.0,
                 learnable_temperature: bool = True,
                 mdn_mu_max: float = 1.2,
                 mdn_min_sigma: float = 1e-3,
                 mdn_distribution: str = "gaussian",
                 use_interaction: bool = False):
        super().__init__()
        self.d_token = d_token
        self.mask_task = True
        self.use_interaction = use_interaction

        def _embed_num() -> nn.Sequential:
            return nn.Sequential(nn.Linear(1, d_token), nn.LayerNorm(d_token), nn.GELU())

        self.shared_num_emb = nn.ModuleList([_embed_num() for _ in range(n_numeric_shared)])
        self.shared_cat_emb = nn.ModuleList([nn.Embedding(c, d_token) for c in cat_cards_shared])
        self.task_num_emb = nn.ModuleList([_embed_num() for _ in range(n_numeric_task)])
        self.task_cat_emb = nn.ModuleList([nn.Embedding(c, d_token) for c in cat_cards_task])

        self.pad_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        if use_interaction:
            self.interaction = nn.Sequential(
                nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_token, d_token)
            )

        # stochastic depth schedule
        dpr = torch.linspace(0, drop_path_rate, steps=n_blocks).tolist()
        self.blocks = nn.ModuleList([
            FTTransformerEncoderBlock(d_token, n_heads, ff_factor, dropout, dpr[i])
            for i in range(n_blocks)
        ])

        if use_mdn:
            if mdn_distribution == "gaussian":
                self.head = MDNHead(d_token, mdn_n_components, mdn_hidden_size,
                                    mdn_temperature, learnable_temperature,
                                    mu_max=mdn_mu_max, min_sigma=mdn_min_sigma)
            elif mdn_distribution == "beta":
                self.head = BetaMDNHead(d_token, mdn_n_components, mdn_hidden_size,
                                         mdn_temperature, learnable_temperature)
            else:
                raise ValueError("mdn_distribution must be 'gaussian' or 'beta'")
        else:
            self.head = RegressionHead(d_token)

    def rebuild_task_banks(self, n_numeric_task: int, cat_cards_task: List[int]) -> None:
        def _embed_num() -> nn.Sequential:
            return nn.Sequential(nn.Linear(1, self.d_token), nn.LayerNorm(self.d_token), nn.GELU())
        self.task_num_emb = nn.ModuleList([_embed_num() for _ in range(n_numeric_task)])
        self.task_cat_emb = nn.ModuleList([nn.Embedding(c, self.d_token) for c in cat_cards_task])

    def forward(self,
                x_num_shared: Optional[torch.Tensor] = None,
                x_num_task: Optional[torch.Tensor] = None,
                x_cat_shared: Optional[torch.Tensor] = None,
                x_cat_task: Optional[torch.Tensor] = None):
        first_tensor = next(t for t in (x_num_shared, x_num_task, x_cat_shared, x_cat_task) if t is not None)
        B = first_tensor.size(0)
        tokens, skip_masks = [], []

        if x_num_shared is not None:
            for i, emb in enumerate(self.shared_num_emb):
                tok = emb(x_num_shared[:, i:i+1]).unsqueeze(1)
                tokens.append(tok)
                skip_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=tok.device))
        if x_cat_shared is not None:
            for i, emb in enumerate(self.shared_cat_emb):
                tok = emb(x_cat_shared[:, i]).unsqueeze(1)
                tokens.append(tok)
                skip_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=tok.device))

        n_num_present = 0 if x_num_task is None else x_num_task.size(1)
        for i, emb in enumerate(self.task_num_emb):
            if i < n_num_present:
                tok = emb(x_num_task[:, i:i+1]).unsqueeze(1)
                mask = torch.zeros(B, 1, dtype=torch.bool, device=tok.device)
            else:
                tok = self.pad_token.expand(B, 1, -1)
                mask = (torch.ones if self.mask_task else torch.zeros)(B, 1, dtype=torch.bool, device=tok.device)
            tokens.append(tok); skip_masks.append(mask)

        n_cat_present = 0 if x_cat_task is None else x_cat_task.size(1)
        for i, emb in enumerate(self.task_cat_emb):
            if i < n_cat_present:
                tok = emb(x_cat_task[:, i]).unsqueeze(1)
                mask = torch.zeros(B, 1, dtype=torch.bool, device=tok.device)
            else:
                tok = self.pad_token.expand(B, 1, -1)
                mask = (torch.ones if self.mask_task else torch.zeros)(B, 1, dtype=torch.bool, device=tok.device)
            tokens.append(tok); skip_masks.append(mask)

        x = torch.cat(tokens, 1)
        skip = torch.cat(skip_masks, 1) if skip_masks else torch.zeros(B, x.size(1), dtype=torch.bool, device=x.device)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], 1)
        skip = torch.cat([torch.zeros(B, 1, device=skip.device, dtype=torch.bool), skip], 1)

        if self.use_interaction:
            x = x + self.interaction(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=skip)

        pooled = x[:, 0]
        if isinstance(self.head, RegressionHead):
            y_pred, _ = self.head(pooled)
            return y_pred, None
        else:
            params = self.head(pooled)
            if isinstance(self.head, MDNHead):
                alpha, mu, sigma = params
                y_pred = (alpha * mu).sum(1, keepdim=True)
                return y_pred, (alpha, mu, sigma)
            else:  # BetaMDNHead
                alpha, a, b = params
                mean = a / (a + b)
                y_pred = (alpha * mean).sum(1, keepdim=True)
                return y_pred, (alpha, a, b)

# -------------------------- Top-level Model --------------------------

class FTTransformerModel:
    def __init__(self, sample_df: pd.DataFrame, *,
                 # schema
                 shared_num_cols: Optional[List[str]] = None,
                 shared_cat_cols: Optional[List[str]] = None,
                 # transformer
                 d_token: int = 32, n_heads: int = 4, n_blocks: int = 2,
                 ff_factor: float = 4.0, dropout: float = 0.1,
                 drop_path_rate: float = 0.0,
                 # training
                 learning_rate: float = 1e-3, weight_decay: float = 1e-3,
                 epochs: int = 10, batch_size: int = 32,
                 seed: int = 42, deterministic: bool = True, benchmark: bool = False,
                 verbose: bool = True, use_scheduler: bool = True,
                 early_stopping: bool = True,
                 # head
                 use_mdn: bool = True,
                 mdn_n_components: int = 2,
                 mdn_hidden_size: int = 32,
                 mdn_temperature: float = 1.0,
                 learnable_temperature: bool = True,
                 mdn_mu_max: float = 1.0,
                 mdn_min_sigma: float = 1e-3,
                 mdn_distribution: str = "gaussian",
                 use_interaction: bool = False,
                 **kwargs):
        # Backward-compat arg aliases
        if 'shared_num' in kwargs and shared_num_cols is None:
            shared_num_cols = kwargs.pop('shared_num')
        if 'shared_cat' in kwargs and shared_cat_cols is None:
            shared_cat_cols = kwargs.pop('shared_cat')
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(benchmark)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_mdn = use_mdn
        self.mdn_distribution = mdn_distribution

        shared_num_cols = shared_num_cols or []
        shared_cat_cols = shared_cat_cols or []
        all_num = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        all_cat = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()
        missing = set((shared_num_cols or []) + (shared_cat_cols or [])) - set(all_num + all_cat)
        if missing:
            raise ValueError(f"Shared columns not found: {missing}")
        self.shared_num_cols = shared_num_cols
        self.shared_cat_cols = shared_cat_cols
        self.task_num_cols = [c for c in all_num if c not in shared_num_cols]
        self.task_cat_cols = [c for c in all_cat if c not in shared_cat_cols]
        self.num_cols = self.shared_num_cols + self.task_num_cols
        self.cat_cols = self.shared_cat_cols + self.task_cat_cols

        # categorical maps
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        cat_cards_shared, cat_cards_task = [], []
        for col in self.shared_cat_cols:
            uniq = pd.Series(sample_df[col].astype(str).unique(), dtype="string")
            mapping = {v: i for i, v in enumerate(uniq)}; mapping["<UNK>"] = len(uniq)
            self.cat_maps[col] = mapping; cat_cards_shared.append(len(mapping))
        for col in self.task_cat_cols:
            uniq = pd.Series(sample_df[col].astype(str).unique(), dtype="string")
            mapping = {v: i for i, v in enumerate(uniq)}; mapping["<UNK>"] = len(uniq)
            self.cat_maps[col] = mapping; cat_cards_task.append(len(mapping))

        self.model = FTTransformerBackbone(
            n_numeric_shared=len(self.shared_num_cols), n_numeric_task=len(self.task_num_cols),
            cat_cards_shared=cat_cards_shared, cat_cards_task=cat_cards_task,
            d_token=d_token, n_heads=n_heads, n_blocks=n_blocks,
            ff_factor=ff_factor, dropout=dropout, drop_path_rate=drop_path_rate,
            use_mdn=use_mdn, mdn_n_components=mdn_n_components, mdn_hidden_size=mdn_hidden_size,
            mdn_temperature=mdn_temperature, learnable_temperature=learnable_temperature,
            mdn_mu_max=mdn_mu_max, mdn_min_sigma=mdn_min_sigma,
            mdn_distribution=mdn_distribution,
            use_interaction=use_interaction
        ).to(self.device)

        # optimizer
        self.weight_decay = weight_decay
        self.base_lr = learning_rate
        self._build_optimizer(head_lr_mult=1.0 if not use_mdn else 3.0)
        self.scheduler = None
        if use_scheduler:
            steps_per_epoch = max(1, int(len(sample_df) / batch_size))
            total_steps = epochs * steps_per_epoch
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-5)

        self._init_state_dict = copy.deepcopy(self.model.state_dict())
        self.early_stopping = early_stopping
        self.patience = 15
        self.min_delta = 1e-4
        self._early_stop_counter = 0
        self.best_epoch, self.best_val_r2 = None, -float('inf')
        self.best_model_state_dict = None

        # histories
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_r2: List[float] = []
        self.val_r2: List[float] = []
        self.train_rmse: List[float] = []
        self.val_rmse: List[float] = []
        self.val_means: List[Any] = []
        self.temperature_history: List[Optional[float]] = []
        self.train_jsd: List[float] = []
        self.val_jsd: List[float] = []
        self.max_train_epoch: Optional[int] = None
        self.max_finetune_epoch: Optional[int] = None

    # ---------------------- utils ----------------------
    def _build_optimizer(self, head_lr_mult: float = 3.0):
        # separate param groups for head vs. rest
        head_params, base_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            (head_params if n.startswith('head') or '.head.' in n else base_params).append(p)
        self.optimizer = optim.AdamW([
            {"params": base_params, "lr": self.base_lr, "weight_decay": self.weight_decay},
            {"params": head_params, "lr": self.base_lr * head_lr_mult, "weight_decay": self.weight_decay},
        ])


    def _encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.shared_cat_cols + self.task_cat_cols:
            if col not in df.columns:
                # Column missing in this frame → skip encoding
                continue
            mapping = self.cat_maps.get(col)
            if mapping is None:
                # Build mapping on the fly if we didn't see this col before
                uniq = pd.Series(df[col].astype(str).unique(), dtype="string")
                mapping = {v: i for i, v in enumerate(uniq)}
                mapping["<UNK>"] = len(uniq)
                self.cat_maps[col] = mapping
            unk = mapping["<UNK>"]
            df[col] = df[col].astype("string").map(mapping).fillna(unk).astype("int64")
        return df


    def _to_tensor_cols(self, df: pd.DataFrame, cols: List[str], dtype: torch.dtype) -> torch.Tensor:
        if not cols:
            return torch.empty(len(df), 0, dtype=dtype, device=self.device)
    
        N, M = len(df), len(cols)
        if dtype == torch.float32:
            arr = np.zeros((N, M), dtype=np.float32)            # numeric: pad with zeros
            for j, c in enumerate(cols):
                if c in df.columns:
                    arr[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            return torch.from_numpy(arr).to(self.device)
        else:
            arr = np.zeros((N, M), dtype=np.int64)
            for j, c in enumerate(cols):
                if c in df.columns:
                    arr[:, j] = df[c].astype("int64").to_numpy()
                else:
                    # categorical: pad with <UNK> if we have a mapping; else 0
                    arr[:, j] = self.cat_maps.get(c, {}).get("<UNK>", 0)
            return torch.from_numpy(arr).to(self.device)


    def _compute_jsd_divergence(self, p_data, q_data, n_bins=30, epsilon=1e-9):
        p_counts, bin_edges = np.histogram(p_data, bins=n_bins, density=True)
        q_counts, _ = np.histogram(q_data, bins=bin_edges, density=True)
        p = p_counts + epsilon; q = q_counts + epsilon
        p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m)); kl_qm = np.sum(q * np.log(q / m))
        jsd = 0.5 * (kl_pm + kl_qm)
        return float(jsd)

    def _hist_pdf_plot(self, y_train, y_train_pred, y_val, y_val_pred, epoch=None):
        import matplotlib.pyplot as plt
        save_dir = "histograms"; os.makedirs(save_dir, exist_ok=True)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        axs[0].hist(y_train_pred, bins=40, alpha=0.5, edgecolor='k', label='Prediction', density=True)
        axs[0].hist(y_train, bins=40, histtype='step', edgecolor='k', linewidth=1.5, label='Truth', density=True)
        axs[0].set_title('Train'); axs[0].legend(); axs[0].grid(True); axs[0].set_ylim(0, 10)
        axs[1].hist(y_val_pred, bins=40, alpha=0.5, edgecolor='k', label='Prediction', density=True)
        axs[1].hist(y_val, bins=40, histtype='step', edgecolor='k', linewidth=1.5, label='Truth', density=True)
        axs[1].set_title('Validation'); axs[1].legend(); axs[1].grid(True); axs[1].set_ylim(0, 10)
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epoch_{epoch:03d}_hist_{timestamp}.pdf" if epoch is not None else f"hist_final_{timestamp}.pdf"
        fig.savefig(os.path.join(save_dir, filename))
        plt.close(fig)

    # ---------------------- training loops ----------------------
    def _train_loop(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                    *, start_epoch: int = 0) -> int:
        X_train = self._encode_categories(X_train); X_val = self._encode_categories(X_val)
        def _tensor(df, cols, dtype):
            return self._to_tensor_cols(df, cols, dtype)
        Xtr_num_sh = _tensor(X_train, self.shared_num_cols, torch.float32)
        Xtr_num_ta = _tensor(X_train, self.task_num_cols, torch.float32)
        Xtr_cat_sh = _tensor(X_train, self.shared_cat_cols, torch.int64)
        Xtr_cat_ta = _tensor(X_train, self.task_cat_cols, torch.int64)
        Ytr = torch.tensor((y_train.values if hasattr(y_train, 'values') else y_train), dtype=torch.float32, device=self.device).unsqueeze(1)

        train_ds = torch.utils.data.TensorDataset(Xtr_num_sh, Xtr_num_ta, Xtr_cat_sh, Xtr_cat_ta, Ytr)
        g = torch.Generator().manual_seed(self.seed)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                                   generator=g, worker_init_fn=lambda wid: np.random.seed(self.seed + wid))
        Xv_num_sh = _tensor(X_val, self.shared_num_cols, torch.float32)
        Xv_num_ta = _tensor(X_val, self.task_num_cols, torch.float32)
        Xv_cat_sh = _tensor(X_val, self.shared_cat_cols, torch.int64)
        Xv_cat_ta = _tensor(X_val, self.task_cat_cols, torch.int64)
        Yv = torch.tensor((y_val.values if hasattr(y_val, 'values') else y_val), dtype=torch.float32, device=self.device).unsqueeze(1)

        epoch_iter = tqdm(range(start_epoch, start_epoch + self.epochs), desc="Training epochs (FT-MDN-Transformer)")
        for global_ep in epoch_iter:
            ep = global_ep - start_epoch
            self.model.train()
            running = 0.0
            for xb_s, xb_t, xc_s, xc_t, yb in train_loader:
                self.optimizer.zero_grad()
                y_hat, extra = self.model(xb_s, xb_t, xc_s, xc_t)
                if self.use_mdn:
                    if self.mdn_distribution == 'gaussian':
                        a, m, s = extra; loss = mdn_gaussian_nll(a, m, s, yb)
                    else:
                        a, A, B = extra; loss = mdn_beta_nll(a, A, B, yb)
                else:
                    loss = F.mse_loss(y_hat, yb)
                loss.backward(); self.optimizer.step();
                if self.scheduler: self.scheduler.step()
                running += loss.item() * yb.size(0)
            train_loss = running / len(train_ds)

            self.model.eval()
            with torch.no_grad():
                y_pred_tr = self.model(Xtr_num_sh, Xtr_num_ta, Xtr_cat_sh, Xtr_cat_ta)[0].cpu().numpy().ravel()
            train_r2 = r2_score(y_train, y_pred_tr)
            train_rmse = math.sqrt(mean_squared_error(y_train, y_pred_tr))

            with torch.no_grad():
                y_pred_v_torch, extra_v = self.model(Xv_num_sh, Xv_num_ta, Xv_cat_sh, Xv_cat_ta)
                if self.use_mdn:
                    if self.mdn_distribution == 'gaussian':
                        a_v, m_v, s_v = extra_v
                        val_loss = mdn_gaussian_nll(a_v, m_v, s_v, Yv).item()
                        y_pred_v = (a_v * m_v).sum(1).cpu().numpy()
                    else:
                        a_v, A_v, B_v = extra_v
                        val_loss = mdn_beta_nll(a_v, A_v, B_v, Yv).item()
                        mean_v = A_v / (A_v + B_v)
                        y_pred_v = (a_v * mean_v).sum(1).cpu().numpy()
                else:
                    val_loss = F.mse_loss(y_pred_v_torch, Yv).item()
                    y_pred_v = y_pred_v_torch.cpu().numpy()

            val_r2 = r2_score(y_val, y_pred_v)
            val_rmse = math.sqrt(mean_squared_error(y_val, y_pred_v))
            train_jsd = self._compute_jsd_divergence(y_train, y_pred_tr)
            val_jsd = self._compute_jsd_divergence(y_val, y_pred_v)

            self.train_loss.append(train_loss); self.val_loss.append(val_loss)
            self.train_r2.append(train_r2); self.val_r2.append(val_r2)
            self.train_rmse.append(train_rmse); self.val_rmse.append(val_rmse)
            self.train_jsd.append(train_jsd); self.val_jsd.append(val_jsd)
            if self.use_mdn:
                if self.mdn_distribution == 'gaussian':
                    self.val_means.append(m_v.mean(0).cpu().tolist())
                    temp = float(self.model.head.log_temperature.exp().detach().cpu())
                else:
                    self.val_means.append((A_v / (A_v + B_v)).mean(0).cpu().tolist())
                    temp = float(self.model.head.log_temperature.exp().detach().cpu())
                self.temperature_history.append(temp)
            else:
                self.val_means.append(None); self.temperature_history.append(None)

            # early stopping on R2 improvement
            if self.early_stopping:
                improved = val_r2 > self.best_val_r2 + self.min_delta
                if improved:
                    self.best_val_r2 = val_r2; self.best_epoch = ep + 1
                    self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                    self._early_stop_counter = 0
                else:
                    self._early_stop_counter += 1
                    if self._early_stop_counter >= self.patience:
                        epoch_iter.write(f"✅ Early stop at epoch {ep+1}.")
                        epoch_iter.close(); return ep + 1

            if self.verbose:
                if self.use_mdn:
                    if self.mdn_distribution == 'gaussian':
                        comp_means = m_v.mean(dim=0).cpu().numpy().tolist()
                    else:
                        comp_means = (A_v / (A_v + B_v)).mean(dim=0).cpu().numpy().tolist()
                    print(f"Epoch {ep+1}/{self.epochs}: train_loss={train_loss:.4f}, train_R2={train_r2:.4f}, val_loss={val_loss:.4f}, val_R2={val_r2:.4f}, means={comp_means}")
                else:
                    print(f"Epoch {ep+1}/{self.epochs}: train_loss={train_loss:.4f}, train_R2={train_r2:.4f}, val_loss={val_loss:.4f}, val_R2={val_r2:.4f}")

        return self.epochs

    # ---------------------- public API ----------------------
    def train(self, X_train, y_train, X_val, y_val):
        self.train_loss.clear(); self.val_loss.clear(); self.train_r2.clear(); self.val_r2.clear()
        self.train_rmse.clear(); self.val_rmse.clear(); self.val_means.clear(); self.temperature_history.clear()
        self.best_epoch = None; self.best_val_r2 = -float('inf'); self._early_stop_counter = 0
        self.max_train_epoch = self._train_loop(X_train, y_train, X_val, y_val)

    def _rebuild_optimizer(self, lr, wd, head_lr_mult: float = 3.0):
        self.base_lr = lr; self.weight_decay = wd
        self._build_optimizer(head_lr_mult=head_lr_mult)

    def reset_weights(self, parts=("tokenizer", "backbone", "head"), mode: str = "restore"):
        name_filters = {
            "tokenizer": ("shared_num_emb", "shared_cat_emb", "task_num_emb", "task_cat_emb"),
            "backbone": ("blocks.",),
            "head": ("head.",),
        }
        wanted = set(parts)
        def _belongs(name):
            return any(name.startswith(prefix) for p in wanted for prefix in name_filters[p])
        if not hasattr(self, "_init_state_dict"):
            self._init_state_dict = copy.deepcopy(self.model.state_dict())
        def _init_module_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        if mode == "restore":
            current = self.model.state_dict()
            for k, v in self._init_state_dict.items():
                if _belongs(k): current[k].copy_(v)
            self.model.load_state_dict(current)
        elif mode == "reinit":
            for n, p in self.model.named_parameters():
                if _belongs(n):
                    module_name = n.split('.')[0]
                    module = getattr(self.model, module_name, None)
                    if module is not None:
                        module.apply(_init_module_weights)
        else:
            raise ValueError("mode must be 'restore' or 'reinit'")
        self._build_optimizer(head_lr_mult=3.0 if self.use_mdn else 1.0)

    def fine_tune(self, X_new_train: pd.DataFrame, y_new_train: pd.Series,
                  X_new_val: pd.DataFrame, y_new_val: pd.Series,
                  *, additional_epochs: int = 20, freeze_backbone: bool = False,
                  freeze_shared: bool = False, lr: float = 5e-4, weight_decay: float = 1e-4, **kwargs):
        if additional_epochs <= 0: raise ValueError("additional_epochs must be ≥1")
        self.best_epoch = None; self.best_val_r2 = -float('inf')
        # 1) figure task columns
        self.task_num_cols = [c for c in X_new_train.select_dtypes(include=[np.number]).columns if c not in self.shared_num_cols]
        self.task_cat_cols = [c for c in X_new_train.select_dtypes(exclude=[np.number]).columns if c not in self.shared_cat_cols]
        self.num_cols = self.shared_num_cols + self.task_num_cols
        self.cat_cols = self.shared_cat_cols + self.task_cat_cols
        # 2) rebuild task banks
        cat_cards_task = [(X_new_train[c].nunique() + 1) for c in self.task_cat_cols]
        self.model.rebuild_task_banks(len(self.task_num_cols), cat_cards_task)
        # update cat maps for task cats
        for col in self.task_cat_cols:
            uniq = pd.Series(X_new_train[col].astype(str).unique(), dtype="string")
            mapping = {v: i for i, v in enumerate(uniq)}; mapping["<UNK>"] = len(uniq)
            self.cat_maps[col] = mapping
        # 3) set requires_grad
        for p in self.model.parameters():
            p.requires_grad_(False)
        if not freeze_shared:
            for emb in list(self.model.shared_num_emb) + list(self.model.shared_cat_emb):
                for p in emb.parameters(): p.requires_grad_(True)
        for emb in list(self.model.task_num_emb) + list(self.model.task_cat_emb):
            for p in emb.parameters(): p.requires_grad_(True)
        for p in self.model.head.parameters(): p.requires_grad_(True)
        if not freeze_backbone:
            for blk in self.model.blocks:
                for p in blk.parameters(): p.requires_grad_(True)
        # 4) new optimizer with higher LR for head
        self._rebuild_optimizer(lr, weight_decay, head_lr_mult=5.0 if self.use_mdn else 1.0)
        steps_per_epoch = max(1, int(len(X_new_train) / self.batch_size))
        total_steps = additional_epochs * steps_per_epoch
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-5)
        # 5) run
        original_epochs = self.epochs; self.epochs = additional_epochs
        self.model.mask_task = False
        start_ep = len(self.train_loss)
        self.max_finetune_epoch = self._train_loop(X_new_train, y_new_train, X_new_val, y_new_val, start_epoch=start_ep) or additional_epochs
        self.epochs = original_epochs
        
        missing_shared = [c for c in (self.shared_num_cols + self.shared_cat_cols) if c not in X_new_train.columns]
        if missing_shared and self.verbose:
            print(f"[FT] Using padded tokens for {len(missing_shared)} missing shared cols (e.g., {missing_shared[:3]})")


    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X_test = self._encode_categories(X_test)
        self.model.eval()
        with torch.no_grad():
            y_pred, _ = self.model(
                self._to_tensor_cols(X_test, self.shared_num_cols, torch.float32),
                self._to_tensor_cols(X_test, self.task_num_cols, torch.float32) if self.task_num_cols else None,
                self._to_tensor_cols(X_test, self.shared_cat_cols, torch.int64) if self.shared_cat_cols else None,
                self._to_tensor_cols(X_test, self.task_cat_cols, torch.int64) if self.task_cat_cols else None,
            )
        return y_pred.cpu().numpy().ravel()

    def predict_distribution(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return distribution params for each sample. For regression head, returns mean only.
        Returns dict with keys depending on head:
          - MDN Gaussian: {'alpha': [N,K], 'mu': [N,K], 'sigma': [N,K], 'mean': [N]}
          - MDN Beta: {'alpha','a','b','mean'}
          - Regression: {'mean'}
        """
        X = self._encode_categories(X)
        self.model.eval()
        with torch.no_grad():
            y_pred, extra = self.model(
                self._to_tensor_cols(X, self.shared_num_cols, torch.float32),
                self._to_tensor_cols(X, self.task_num_cols, torch.float32) if self.task_num_cols else None,
                self._to_tensor_cols(X, self.shared_cat_cols, torch.int64) if self.shared_cat_cols else None,
                self._to_tensor_cols(X, self.task_cat_cols, torch.int64) if self.task_cat_cols else None,
            )
        out = {'mean': y_pred.cpu().numpy().ravel()}
        if self.use_mdn:
            if self.mdn_distribution == 'gaussian':
                a, m, s = extra
                out.update({
                    'alpha': a.cpu().numpy(),
                    'mu': m.cpu().numpy(),
                    'sigma': s.cpu().numpy(),
                })
            else:
                a, A, B = extra
                out.update({
                    'alpha': a.cpu().numpy(),
                    'a': A.cpu().numpy(),
                    'b': B.cpu().numpy(),
                })
        return out

    def select_best_model(self):
        if hasattr(self, 'best_model_state_dict') and self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            if self.verbose:
                print(f"Switched to best model from epoch {self.best_epoch} with val_R2={self.best_val_r2:.4f}")
        else:
            print("No best model state saved yet. Run training first.")

    def get_training_history(self) -> Dict[str, Any]:
        return {
            'train_loss': self.train_loss, 'val_loss': self.val_loss,
            'train_r2': self.train_r2, 'val_r2': self.val_r2,
            'train_rmse': self.train_rmse, 'val_rmse': self.val_rmse,
            'val_means': self.val_means, 'temperature_history': self.temperature_history,
            'best_epoch': self.best_epoch, 'best_val_r2': self.best_val_r2,
            'train_jsd': self.train_jsd, 'val_jsd': self.val_jsd,
            'max_train_epoch': self.max_train_epoch, 'max_finetune_epoch': self.max_finetune_epoch,
        }

    def get_epoch_metrics(self) -> Dict[str, Any]:
        metrics = {}
        n_pre = getattr(self, 'max_train_epoch', 0) or 0
        if n_pre:
            metrics['loss_pretrain'] = self.train_loss[:n_pre]
            metrics['r2_pretrain'] = self.train_r2[:n_pre]
        n_ft = getattr(self, 'max_finetune_epoch', 0) or 0
        if n_ft:
            start = n_pre; end = start + n_ft
            metrics['loss_finetune'] = self.train_loss[start:end]
            metrics['r2_finetune'] = self.train_r2[start:end]
        return metrics

    # ---------------------- Portfolio utilities ----------------------
    @staticmethod
    def gaussian_mixture_pdf(y: np.ndarray, alpha: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        # y: [M], alpha/mu/sigma: [N,K]
        y = y[None, :, None]  # [1,M,1]
        mu = mu[:, None, :]   # [N,1,K]
        sigma = sigma[:, None, :]  # [N,1,K]
        alpha = alpha[:, None, :]  # [N,1,K]
        # PDF for each loan & comp at y-grid
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
        pdf = norm * np.exp(-0.5 * ((y - mu) / sigma) ** 2)
        mix = (alpha * pdf).sum(-1)  # [N,M]
        return mix

    @staticmethod
    def beta_mixture_pdf(y: np.ndarray, alpha: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        from scipy.special import beta as Bf
        y = np.clip(y, 1e-8, 1 - 1e-8)
        y = y[None, :, None]
        a = a[:, None, :]; b = b[:, None, :]; alpha = alpha[:, None, :]
        # Beta pdf
        B_ab = Bf(a, b)
        pdf = (y ** (a - 1) * (1 - y) ** (b - 1)) / B_ab
        mix = (alpha * pdf).sum(-1)
        return mix

    def portfolio_density(self, X: pd.DataFrame, y_grid: Optional[np.ndarray] = None,
                      grid_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mixture-of-mixtures predictive density across X.
        Returns (y_grid, density) where density is averaged across loans.
        """
        if y_grid is None:
            y_grid = np.linspace(0.0, 1.0, grid_points)
    
        dist = self.predict_distribution(X)
    
        # Regression head → KDE on point means as a fallback
        if not self.use_mdn:
            from scipy.stats import gaussian_kde  # pip install scipy
            kde = gaussian_kde(dist['mean'])
            return y_grid, kde(y_grid)
    
        # MDN head → analytic mixture-of-mixtures
        if self.mdn_distribution == 'gaussian':
            mix = self.gaussian_mixture_pdf(y_grid, dist['alpha'], dist['mu'], dist['sigma'])
        else:  # 'beta'
            mix = self.beta_mixture_pdf(y_grid, dist['alpha'], dist['a'], dist['b'])
    
        density = mix.mean(0)  # average across loans
        return y_grid, density
