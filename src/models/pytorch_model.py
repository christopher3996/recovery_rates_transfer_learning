import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from models.base_model import BaseModel


class PyTorchModel(BaseModel):
    """
    A PyTorch-based regression model that supports a training phase and an optional fine-tuning phase.
    Each phase records epoch-wise loss and R² on the specified evaluation dataset.
    """

    def __init__(self,
                 #input_size,
                 hidden_layers=[64, 32],
                 output_size=1,
                 learning_rate=0.001,
                 epochs=50,
                 batch_size=32,
                 seed=42,
                 deterministic=True,
                 benchmark=False,
                 **kwargs):
        """
        Initialize the PyTorchModel.

        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int): Sizes of the hidden layers.
            output_size (int): Size of the output layer (usually 1 for regression).
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of epochs to train/fine-tune.
            batch_size (int): Batch size for training/fine-tuning.
            seed (int): Random seed (passed to BaseModel).
            deterministic (bool): If True, PyTorch uses deterministic algorithms (slower).
            benchmark (bool): If True, cudnn.benchmark = True (faster but less reproducible).
        """
        # Call BaseModel constructor to re-seed, so that each model starts identically.
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model = self.build_model(input_size, hidden_layers, output_size).to(self.device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Track training and fine-tuning metrics
        self.train_loss_history = []
        self.finetune_loss_history = []
        self.train_r2_history = []
        self.finetune_r2_history = []

        self.is_fine_tuning = False

    def build_model(self, input_size, hidden_layers, output_size):
        """
        Build a feed-forward neural network based on the specified layers.

        Args:
            input_size (int): Input feature dimension.
            hidden_layers (list of int): Sizes of hidden layers.
            output_size (int): Size of the output layer.

        Returns:
            nn.Sequential: The constructed neural network model.
        """
        layers = []
        in_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        return nn.Sequential(*layers)

    def train(self, X_train, y_train, X_eval, y_eval):
        """
        Train the model on given training data and evaluate metrics on evaluation data each epoch.

        Args:
            X_train (pd.DataFrame or pd.Series): Training features.
            y_train (pd.Series): Training targets.
            X_eval (pd.DataFrame or pd.Series): Evaluation features for computing R² each epoch.
            y_eval (pd.Series): Evaluation targets for computing R² each epoch.
        """
        self.is_fine_tuning = False
        self.train_loss_history.clear()
        self.train_r2_history.clear()

        X_train = pd.get_dummies(X_train)
        self.feature_columns = X_train.columns
        
        X_eval = pd.get_dummies(X_eval)
        X_eval = X_eval.reindex(columns=self.feature_columns, fill_value=0)

        self.model = self.build_model(X_train.shape[1], self.hidden_layers, self.output_size).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self._run_training_loop(X_train, y_train, X_eval, y_eval, self.train_loss_history, self.train_r2_history)

    def fine_tune(self, X_train, y_train, X_eval, y_eval, **kwargs):
        """
        Fine-tune the model on new training data (e.g., target domain) and evaluate metrics each epoch.

        Args:
            X_train (pd.DataFrame or pd.Series): Training features for fine-tuning.
            y_train (pd.Series): Training targets for fine-tuning.
            X_eval (pd.DataFrame or pd.Series): Evaluation features for computing R² each epoch during fine-tuning.
            y_eval (pd.Series): Evaluation targets for computing R² each epoch during fine-tuning.
        """
        self.is_fine_tuning = True
        self.finetune_loss_history.clear()
        self.finetune_r2_history.clear()
        
        X_train = pd.get_dummies(X_train)
        X_eval = pd.get_dummies(X_eval)
        
        # Align both to the feature space learned during training
        X_train = X_train.reindex(columns=self.feature_columns, fill_value=0)
        X_eval = X_eval.reindex(columns=self.feature_columns, fill_value=0)
        
        self.epochs = kwargs["additional_epochs"]
        self._run_training_loop(X_train, y_train, X_eval, y_eval, self.finetune_loss_history, self.finetune_r2_history)

    def _run_training_loop(self, X_train, y_train, X_eval, y_eval, loss_history, r2_history):
        """
        Internal method to run the training or fine-tuning loop.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_eval: Evaluation features.
            y_eval: Evaluation targets.
            loss_history (list): A list to record the loss per epoch.
            r2_history (list): A list to record the R² per epoch.
        """
        

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(X_batch)

            # Compute average epoch loss
            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)

            # Compute R² on the evaluation set
            r2 = self.compute_r2(X_eval, y_eval)
            r2_history.append(r2)

    def compute_r2(self, X, y):
        """
        Compute R² score on given evaluation data.

        Args:
            X: Evaluation features.
            y: Evaluation targets.

        Returns:
            float: R² score on the given evaluation data.
        """
        self.model.eval()
        inputs = torch.from_numpy(X.values.astype(np.float32)).to(self.device)
        with torch.no_grad():
            predictions = self.model(inputs).cpu().numpy().flatten()
        return r2_score(y, predictions)

    def predict(self, X_test):
        """
        Predict using the trained/fine-tuned model.

        Args:
            X_test (pd.DataFrame or pd.Series): Test features.

        Returns:
            np.ndarray: Predictions for the given features.
        """
        X_test = pd.get_dummies(X_test)
        X_test = X_test.reindex(columns=self.feature_columns, fill_value=0)
        
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            return outputs.cpu().numpy().flatten()

    def save_model(self, filepath):
        """
        Save the model state_dict to a file.

        Args:
            filepath (str): Path to save the model state.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """
        Load the model state_dict from a file.

        Args:
            filepath (str): Path to the model state file.
        """
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.to(self.device)
        else:
            print(f"Model file {filepath} does not exist.")

    def get_epoch_metrics(self):
        """
        Get epoch-wise metrics recorded during training and/or fine-tuning.

        Returns:
            dict: A dictionary with keys:
                'loss_pretrain', 'r2_pretrain' for the initial training phase.
                'loss_finetune', 'r2_finetune' for the fine-tuning phase.
        """
        metrics = {}
        # Pre-training metrics
        if self.train_loss_history:
            metrics['loss_pretrain'] = self.train_loss_history
        if self.train_r2_history:
            metrics['r2_pretrain'] = self.train_r2_history
        # Fine-tuning metrics
        if self.finetune_loss_history:
            metrics['loss_finetune'] = self.finetune_loss_history
        if self.finetune_r2_history:
            metrics['r2_finetune'] = self.finetune_r2_history

        return metrics