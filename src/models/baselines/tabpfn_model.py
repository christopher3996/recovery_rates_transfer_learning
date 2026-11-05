import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

from models.base_model import BaseModel

class TabPFNRegressorModel(BaseModel):
    """
    A class that wraps TabPFNRegressor to match the BaseModel interface.
    Since TabPFN typically does a single-pass fit, we treat each call to .train() or .fine_tune()
    as one pass and store a "pretrain" or "finetune" metric in lists.
    """

    def __init__(
        self,
        seed=42,
        deterministic=True,
        benchmark=False,
        **tabpfn_kwargs
    ):
        """
        Initialize the TabPFN regressor, respecting BaseModel's environment seeding.

        Args:
            seed (int): Random seed for reproducibility.
            deterministic (bool): Whether to enforce deterministic environment.
            benchmark (bool): Whether to enable cudnn.benchmark if PyTorch is in use.
            tabpfn_kwargs: Additional kwargs directly passed to TabPFNRegressor.
        """
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)

        # Create the TabPFN regressor
        self.model = TabPFNRegressor(**tabpfn_kwargs)

        # We'll track MSE and R² for each train/fine-tune call
        self.train_loss_history = []      # MSE after pre-training
        self.train_r2_history = []
        self.finetune_loss_history = []   # MSE after fine-tuning
        self.finetune_r2_history = []

        self.is_fine_tuning = False

    def train(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """
        Train the TabPFN regressor on (X_train, y_train).
        Since TabPFN doesn't do multi-epoch fitting, we do a single pass, 
        then optionally measure MSE & R² on X_eval/y_eval.
        """
        self.is_fine_tuning = False

        # Clear old training metric lists
        self.train_loss_history.clear()
        self.train_r2_history.clear()

        # Fit the model in one go
        self.model.fit(X_train, y_train)

        # If we have eval data, compute metrics
        if X_eval is not None and y_eval is not None:
            preds = self.model.predict(X_eval)
            mse_val = mean_squared_error(y_eval, preds)
            r2_val = r2_score(y_eval, preds)

            self.train_loss_history.append(mse_val)
            self.train_r2_history.append(r2_val)

    def fine_tune(self, X_train, y_train, X_eval=None, y_eval=None):
        """
        "Fine-tune" by simply re-fitting TabPFNRegressor on new data.
        (TabPFN doesn't support incremental training natively, so we just overwrite.)
        """
        self.is_fine_tuning = True

        # Clear old fine-tune metric lists
        self.finetune_loss_history.clear()
        self.finetune_r2_history.clear()

        # Re-fit model on new data
        self.model.fit(X_train, y_train)

        # Evaluate on optional eval set
        if X_eval is not None and y_eval is not None:
            preds = self.model.predict(X_eval)
            mse_val = mean_squared_error(y_eval, preds)
            r2_val = r2_score(y_eval, preds)

            self.finetune_loss_history.append(mse_val)
            self.finetune_r2_history.append(r2_val)

    def predict(self, X_test):
        """
        Predict regression outputs for X_test.
        """
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """
        Save the TabPFNRegressor via pickle (TabPFN doesn't provide built-in save).
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        """
        Load the TabPFNRegressor from file.
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

    def get_epoch_metrics(self):
        """
        Return a dict of "loss" (MSE) and R² for pretrain vs. finetune.
        Since there's no real multi-epoch loop, these lists typically hold 0 or 1 values.
        """
        metrics = {}
        # Pre-train
        if self.train_loss_history:
            metrics['loss_pretrain'] = self.train_loss_history
        if self.train_r2_history:
            metrics['r2_pretrain'] = self.train_r2_history
        # Fine-tune
        if self.finetune_loss_history:
            metrics['loss_finetune'] = self.finetune_loss_history
        if self.finetune_r2_history:
            metrics['r2_finetune'] = self.finetune_r2_history

        return metrics
