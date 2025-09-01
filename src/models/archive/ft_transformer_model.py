import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score
from models.base_model import BaseModel

class FTTransformerEncoderBlock(nn.Module):
    """
    A single Transformer block consisting of:
      - MultiHead Attention
      - FeedForward Network (2-layer MLP)
      - Residual connections + LayerNorm
    """
    def __init__(self, d_token, n_heads, ff_factor=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_token)

        self.ff = nn.Sequential(
            nn.Linear(d_token, int(ff_factor * d_token)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(ff_factor * d_token), d_token)
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_token)

    def forward(self, x):
        # x shape: (batch_size, n_features, d_token)
        # 1) Multi-Head Attention
        attn_out, _ = self.attn(x, x, x)  # Self-attention
        x = x + self.attn_dropout(attn_out)
        x = self.attn_norm(x)

        # 2) Feed-Forward
        ff_out = self.ff(x)
        x = x + self.ff_dropout(ff_out)
        x = self.ff_norm(x)
        return x


class FTTransformerBackbone(nn.Module):
    """
    An FT-Transformer backbone for tabular data:
      1. A feature tokenizer (each feature => embedding)
      2. A stack of Transformer blocks
      3. Pooling across feature dimension
      4. Final linear to 1-dim output (regression)
    """
    def __init__(
        self,
        n_features,
        d_token=32,
        n_heads=4,
        n_blocks=2,
        ff_factor=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # --- 1) Feature Tokenizer (simple linear projection per feature) ---
        # We'll learn an embedding vector of size d_token for each feature.
        # shape => (n_features, d_token)
        self.feature_embeddings = nn.Parameter(torch.randn(n_features, d_token))
        nn.init.xavier_uniform_(self.feature_embeddings)

        # Optional: if you want a small scale param or other feature-specific parameters,
        # you can include it here. We keep it minimal for clarity.

        # --- 2) Stack of Transformer Encoder Blocks ---
        self.blocks = nn.ModuleList([
            FTTransformerEncoderBlock(d_token=d_token, n_heads=n_heads, ff_factor=ff_factor, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # --- 3) Final layer for regression (after pooling) ---
        self.head = nn.Linear(d_token, 1)
        # Initialize the output layer
        nn.init.xavier_uniform_(self.head.weight)

        # We'll use a simple average pooling across the 'feature' dimension in forward.
        # Alternatively, you can add a CLS token or other pooling strategy if desired.

    def forward(self, x):
        """
        x shape = (batch_size, n_features).
        We turn each scalar feature into a d_token embedding, then pass into the transformer.
        """
        bsz, n_feat = x.shape
        assert n_feat == self.n_features, f"Expected {self.n_features} features, got {n_feat}"

        # Expand + multiply => shape becomes (batch_size, n_features, d_token)
        # We treat each numeric feature as scaling of a learned embedding vector.
        # Alternatively, you could project the numeric input [batch_size, n_features]
        # through a linear layer if you prefer. Here is a simple approach:
        embeddings = x.unsqueeze(-1) * self.feature_embeddings.unsqueeze(0)

        # Pass through each Transformer block
        for block in self.blocks:
            embeddings = block(embeddings)

        # Pool across the feature dimension: shape => (batch_size, d_token)
        # Simple average:
        pooled = embeddings.mean(dim=1)

        # Final linear => shape => (batch_size, 1)
        out = self.head(pooled)
        return out


class FTTransformerModel(BaseModel):
    """
    A PyTorch-based FTTransformer for regression, 
    with the same method signatures as your baseline PyTorchModel.
    """

    def __init__(self,
                 n_features,
                 d_token=32,
                 n_heads=4,
                 n_blocks=2,
                 ff_factor=4.0,
                 dropout=0.1,
                 learning_rate=1e-3,
                 epochs=50,
                 batch_size=32,
                 seed=42,
                 deterministic=True,
                 benchmark=False,
                 **kwargs):
        """
        Initialize the FTTransformerModel.

        Args:
            n_features (int): Number of input features (columns in your tabular data).
            d_token (int): Dimensionality of token embeddings.
            n_heads (int): Number of attention heads.
            n_blocks (int): Number of Transformer blocks.
            ff_factor (float): Expansion factor for MLP in each transformer block.
            dropout (float): Dropout probability in the transformer.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of epochs to train or fine-tune.
            batch_size (int): Batch size for training or fine-tuning.
            seed (int): Random seed.
            deterministic (bool): Use deterministic ops for reproducibility (slower).
            benchmark (bool): If True, enable cudnn.benchmark (faster but less reproducible).
        """
        # Call BaseModel constructor to set the seed, device, etc.
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build the actual FT-Transformer backbone
        self.model = FTTransformerBackbone(
            n_features=n_features,
            d_token=d_token,
            n_heads=n_heads,
            n_blocks=n_blocks,
            ff_factor=ff_factor,
            dropout=dropout
        ).to(self.device)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Track training and fine-tuning metrics
        self.train_loss_history = []
        self.train_r2_history = []
        self.finetune_loss_history = []
        self.finetune_r2_history = []
        self.is_fine_tuning = False

    def train(self, X_train, y_train, X_eval, y_eval):
        """
        Train (pre-train) the model on source data (or any data you like).
        Evaluate MSE/R² on the specified eval set each epoch.
        """
        self.is_fine_tuning = False
        self.train_loss_history.clear()
        self.train_r2_history.clear()
        self._run_training_loop(X_train, y_train, X_eval, y_eval,
                                self.train_loss_history, self.train_r2_history)

    def fine_tune(self, X_train, y_train, X_eval, y_eval, **kwargs):
        """
        Fine-tune the model on target data (or any second-stage data).
        Evaluate MSE/R² on the specified eval set each epoch.
        """
        self.is_fine_tuning = True
        self.finetune_loss_history.clear()
        self.finetune_r2_history.clear()
        self._run_training_loop(X_train, y_train, X_eval, y_eval,
                                self.finetune_loss_history, self.finetune_r2_history)

    def _run_training_loop(self, X_train, y_train, X_eval, y_eval, loss_history, r2_history):
        """
        Internal method: runs a standard training loop for the specified number of epochs.
        Stores epoch-wise losses in `loss_history` and R² in `r2_history`.
        """
        # Convert input to Torch tensors & create DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(X_batch)

            # Average training loss
            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)

            # Compute R² on the evaluation set
            r2_val = self.compute_r2(X_eval, y_eval)
            r2_history.append(r2_val)

    def compute_r2(self, X, y):
        """
        Compute R² score using the current model on dataset (X, y).
        """
        self.model.eval()
        X_t = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().flatten()
        return r2_score(y, preds)

    def predict(self, X_test):
        """
        Predict on new data. Output shape = (num_samples,).
        """
        self.model.eval()
        X_t = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t)
        return preds.cpu().numpy().flatten()

    def save_model(self, filepath):
        """
        Save the model's state_dict to a given file path.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """
        Load the model's state_dict from a file path (if it exists).
        """
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.to(self.device)
        else:
            print(f"Model file {filepath} does not exist.")

    def get_epoch_metrics(self):
        """
        Return a dictionary with epoch-wise metrics for training + fine-tuning phases.
        Same keys as your baseline model: 'loss_pretrain', 'r2_pretrain', 'loss_finetune', 'r2_finetune'.
        """
        metrics = {}
        if self.train_loss_history:
            metrics['loss_pretrain'] = self.train_loss_history
        if self.train_r2_history:
            metrics['r2_pretrain'] = self.train_r2_history

        if self.finetune_loss_history:
            metrics['loss_finetune'] = self.finetune_loss_history
        if self.finetune_r2_history:
            metrics['r2_finetune'] = self.finetune_r2_history

        return metrics
