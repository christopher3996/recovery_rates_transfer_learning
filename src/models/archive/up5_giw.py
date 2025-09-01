import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

# Adjust project path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'giw_helpers'))

from utils.seed import set_seed
from kmm import kmm
from up5_utils import get_feature, get_activation, val_split
from up5_model import EnhancedMLP, SimpleMLP


class MLPTransfer:
    """
    A two-phase training class similar to LGBMTrans:
    1) Pre-training on the training set for 'pretrain_epochs' epochs.
    2) If validation data is provided, register a forward hook, extract features,
       perform KMM fine-tuning for ' finetune_epochs' epochs, and estimate 'alpha'.
       Otherwise, run a normal training (fine-tuning) on the training set only.
    """

    def __init__(
            self,
            learning_rate=1e-3,
            pretrain_epochs=100,
            finetune_epochs=200,
            batch_size=128,
            seed=42
    ):
        """
        Args:
            learning_rate (float): Learning rate for the optimizer.
            pretrain_epochs (int): Number of epochs for the pre-training phase.
            finetune_epochs (int): Number of epochs for the fine-tuning (second) phase.
            batch_size (int): Mini-batch size.
            seed (int): Random seed for reproducibility.
        """
        set_seed(seed)
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_loss_history = []
        self.train_r2_history = []
        self.train_test_loss_history = []
        self.finetune_loss_history = []
        self.finetune_r2_history = []
        self.finetune_test_loss_history = []

        self.alpha = None
        self.net = None
        self.optimizer = None

    def compute_test_r2(self, X_test, y_test, device):
        """
        计算模型在 X_test, y_test 上的 R² 分数。
        X_test, y_test 均为 pandas DataFrame/Series。
        """
        self.net.eval()
        X_test_t = torch.from_numpy(X_test.values.astype(np.float32)).to(device)
        with torch.no_grad():
            preds = self.net(X_test_t)  # (batch_size, 1)
            preds_np = preds.cpu().numpy().flatten()
        return r2_score(y_test, preds_np)

    def compute_test_loss(self, X_test, y_test, device):
            """
            计算模型在 X_test, y_test 上的平均 MSE Loss。
            """
            self.net.eval()
            X_test_t = torch.from_numpy(X_test.values.astype(np.float32)).to(device)
            y_test_t = torch.from_numpy(y_test.values.astype(np.float32)).to(device).unsqueeze(-1)
            with torch.no_grad():
                preds = self.net(X_test_t)
                test_loss = F.mse_loss(preds, y_test_t, reduction='mean').item()
            return test_loss

    def train(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        """
        Train the model in two phases:
          1) Pre-train on (X_train, y_train) for 'pretrain_epochs' epochs.
          2) If (X_val, y_val) is provided, do KMM fine-tuning, etc.
             Otherwise, run normal training (fine-tuning) on (X_train, y_train).
        """
        # --- (1) 数据转换 ---
        X_train_t = torch.from_numpy(X_train.values.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.values.astype(np.float32))
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        if X_val is not None and y_val is not None:
            X_val_t = torch.from_numpy(X_val.values.astype(np.float32))
            y_val_t = torch.from_numpy(y_val.values.astype(np.float32))
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        input_size = X_train_t.shape[1]
        # 使用 SimpleMLP（也可改成 EnhancedMLP）
        self.net = SimpleMLP(input_size=input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # --- (2) 如果没有验证数据，就直接进行普通训练 ---
        if X_val is None or y_val is None:
            print("    *[MLPTransfer] No validation data -> Skipping KMM. Doing normal training.")
            for epoch in range(self.pretrain_epochs):
                accum_loss = 0.0
                num_samples = 0
                self.net.train()
                
                for features_batch, labels_batch in train_loader:
                    features_batch = features_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device).unsqueeze(-1)

                    preds = self.net(features_batch)
                    batch_loss = F.mse_loss(preds, labels_batch, reduction='mean')
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                    batch_size = features_batch.size(0)
                    accum_loss += batch_loss.item() * batch_size
                    num_samples += batch_size

                mean_loss = accum_loss / num_samples
                self.train_loss_history.append(mean_loss)

                test_loss = self.compute_test_loss(X_test, y_test, self.device)
                self.train_test_loss_history.append(test_loss)

                epoch_r2 = self.compute_test_r2(X_test, y_test, self.device)
                self.train_r2_history.append(epoch_r2)

                #print(f"[Training Epoch {epoch+1}/{self.pretrain_epochs}] "
                 #     f"Train MSE: {mean_loss:.4f}, Test MSE: {test_loss:.4f}, Test R2: {epoch_r2:.4f}")

            self.alpha = 1.0
            return

        # --- (3) 有验证数据，执行两阶段训练 ---
        #print("     === [1] Pre-training start ===")
        for epoch in range(self.pretrain_epochs):
            accum_loss = 0.0
            num_samples = 0
            self.net.train()

            for features_batch, labels_batch in train_loader:
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).unsqueeze(-1)

                preds = self.net(features_batch)
                batch_loss = F.mse_loss(preds, labels_batch, reduction='mean')

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                batch_size = features_batch.size(0)
                accum_loss += batch_loss.item() * batch_size
                num_samples += batch_size

            mean_loss = accum_loss / num_samples
            self.train_loss_history.append(mean_loss)

            test_loss = self.compute_test_loss(X_test, y_test, self.device)
            self.train_test_loss_history.append(test_loss)

            epoch_r2 = self.compute_test_r2(X_test, y_test, self.device)
            self.train_r2_history.append(epoch_r2)

            #print(f"[Pretrain Epoch {epoch+1}/{self.pretrain_epochs}] "
             #     f"Train MSE: {mean_loss:.4f}, Test MSE: {test_loss:.4f}, Test R2: {epoch_r2:.4f}")
        
        # --- (4) KMM fine-tuning 阶段 ---
        print("    *[MLPTransfer] === [2] KMM Fine-tuning start ===")
        self.net.input_layer1.register_forward_hook(get_activation('res_block2_output'))

        fe_tr, fe_val, index_val = get_feature(self.net, train_loader, val_loader)
        val_dic, alpha_est = val_split(fe_tr, fe_val, index_val)
        self.alpha = alpha_est
        print(f"     Estimated alpha = {self.alpha:.4f}")

        val_iter = iter(val_loader)
        count = 0

        for epoch in range( self.finetune_epochs):
            accum_loss = 0.0
            num_samples = 0
            self.net.train()

            for features_batch, labels_batch in train_loader:
                self.net.eval()
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device).unsqueeze(-1)

                out_train = self.net(features_batch)
                l_tr = F.mse_loss(out_train, labels_batch, reduction='none').reshape(-1, 1)

                try:
                    val_features, val_labels = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_features, val_labels = next(val_iter)
                    count = 0

                val_features = val_features.to(self.device)
                val_labels = val_labels.to(self.device).unsqueeze(-1)
                batch_size = len(val_labels)
                val_indices = list(range(count, count + batch_size))
                count += batch_size

                split_labels = [val_dic.get(idx, False) for idx in val_indices]
                split_labels = torch.tensor(split_labels, dtype=torch.bool).to(self.device)

                val1_features = val_features[split_labels]
                val1_labels = val_labels[split_labels]
                val2_features = val_features[~split_labels]
                val2_labels = val_labels[~split_labels]

                out_val1 = self.net(val1_features)
                l_val1 = F.mse_loss(out_val1, val1_labels, reduction='none').reshape(-1, 1)

                # 动态估计 kernel_width
                with torch.no_grad():
                    dist_mat = torch.cdist(l_tr, l_tr)
                    tril_idx = torch.tril_indices(l_tr.size(0), l_tr.size(0), offset=-1)
                    dist_flat = dist_mat[tril_idx[0], tril_idx[1]]
                    kernel_width = float(torch.quantile(dist_flat, 0.5))

                coef = kmm(l_tr.detach().cpu().numpy(), l_val1.detach().cpu().numpy(), kernel_width)
                w = torch.from_numpy(np.asarray(coef)).float().to(self.device)

                self.net.train()
                out_train_wr = self.net(features_batch)
                l_tr_wr = F.mse_loss(out_train_wr, labels_batch, reduction='none').squeeze(1)
                weighted_loss = torch.mean(l_tr_wr * w)

                if (1 - self.alpha) > 0 and len(val2_features) > 0:
                    out_val2 = self.net(val2_features)
                    l_val2 = F.mse_loss(out_val2, val2_labels, reduction='mean')
                    total_loss = self.alpha * weighted_loss + (1 - self.alpha) * l_val2
                else:
                    total_loss = self.alpha * weighted_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # 累加当前 batch 的 (batch_loss * batch_size)
                accum_loss += total_loss.item() * batch_size
                num_samples += batch_size

            mean_loss = accum_loss / num_samples
            self.finetune_loss_history.append(mean_loss)

            test_loss = self.compute_test_loss(X_test, y_test, self.device)
            self.finetune_test_loss_history.append(test_loss)

            epoch_r2 = self.compute_test_r2(X_test, y_test, self.device)
            self.finetune_r2_history.append(epoch_r2)

           # print(f"[Fine-tune Epoch {epoch+1}/{ self.finetune_epochs}] "
            #      f"FineTune MSE: {mean_loss:.4f}, Test MSE: {test_loss:.4f}, Test R2: {epoch_r2:.4f}")

    def predict(self, X):
        """
        Predict outputs for input X.
        X must be a Pandas DataFrame, Returns a NumPy array of predictions.
        """

        # Convert DataFrame to Torch Tensor
        X = torch.from_numpy(X.values.astype(np.float32)).to(self.device)

        self.net.eval()
        with torch.no_grad():
            out = self.net(X)
        return out.cpu().numpy()

    def save_model(self, filepath):
        """
        Save the model's parameters to a file.
        """
        if self.net is None:
            print("No model to save.")
            return
        torch.save(self.net.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath, input_size):
        """
        Load the model's parameters from a file, using the same input size.
        """
        self.net = EnhancedMLP(input_size=input_size).to(self.device)
        self.net.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")

    def get_epoch_metrics(self):
        """
        Return metrics from the training process, including:
          - alpha
          - loss_pretrain / r2_pretrain
          - test_loss_pretrain
          - loss_finetune / r2_finetune
          - test_loss_finetune
        """
        metrics = {
            'alpha': self.alpha,

            # 这里为了名称对齐，依然叫 train_* 表示第一阶段
            'loss_pretrain': self.train_loss_history,
            'r2_pretrain': self.train_r2_history,
            'test_loss_pretrain': self.train_test_loss_history,
            # 名称对齐，叫finetune_* 表示transfer learning
            'loss_finetune': self.finetune_loss_history,
            'r2_finetune': self.finetune_r2_history,
            'test_loss_finetune': self.finetune_test_loss_history
        }
        return metrics