import os
import joblib
import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from models.base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    A Random Forest regression model supporting a two-phase approach:
      1) Pre-train on source data
      2) Fine-tune on target data (by adding more trees)

    Unlike XGBoost or PyTorch, scikit-learn's Random Forest does not provide
    iteration-wise metrics. We record only a single RMSE and R² each time
    we call 'train' or 'fine_tune'.
    """

    def __init__(self,
                 rf_params=None,
                 fine_tune_trees=50,
                 seed=42,
                 deterministic=True,
                 benchmark=False,
                 **kwargs):
        """
        Initialize the RandomForestModel.

        Args:
            rf_params (dict): Hyperparameters for the RandomForestRegressor.
                              e.g., {'n_estimators': 100, 'max_depth': 5, ...}
            fine_tune_trees (int): Number of additional trees to add during the fine-tuning phase.
            seed (int): Random seed for reproducibility.
            deterministic (bool): Passed to BaseModel (affects PyTorch, not scikit-learn).
            benchmark (bool): Passed to BaseModel (affects PyTorch, not scikit-learn).
        """
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)

        if rf_params is None:
            rf_params = {}
        # If user didn't specify random_state, set it
        if 'random_state' not in rf_params:
            rf_params['random_state'] = seed
        
        self.rf_params = rf_params
        self.fine_tune_trees = fine_tune_trees

        # Create the underlying RandomForestRegressor
        # (By default, warm_start=False => each .fit starts a fresh model.)
        self.model = RandomForestRegressor(**self.rf_params)

        # Buffers for storing single-phase metrics (not epoch-by-epoch)
        self.train_loss_history = []       # "loss_pretrain"
        self.train_r2_history = []         # "r2_pretrain"
        self.finetune_loss_history = []    # "loss_finetune"
        self.finetune_r2_history = []      # "r2_finetune"

    def train(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """
        Pre-train on source data (or initial train).
        
        We'll produce ONE entry in train_loss_history / train_r2_history
        corresponding to the entire random forest after training.
        """
        # Clear old pretrain histories
        self.train_loss_history.clear()
        self.train_r2_history.clear()

        # Train from scratch (warm_start=False)
        # If the user had set 'warm_start=True' in self.rf_params, we still
        # override it here to ensure a fresh start for the pre-training phase.
        fresh_params = {**self.rf_params, 'warm_start': False}
        self.model = RandomForestRegressor(**fresh_params)
        self.model.fit(X_train, y_train)

        if X_eval is not None and y_eval is not None:
            y_pred = self.model.predict(X_eval)
            # For consistency with your other models, we'll store RMSE as "loss"
            rmse = sqrt(mean_squared_error(y_eval, y_pred))
            r2 = r2_score(y_eval, y_pred)
            self.train_loss_history.append(rmse)
            self.train_r2_history.append(r2)

    def fine_tune(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """
        Fine-tune on target data by adding 'fine_tune_trees' more trees
        to the existing forest (warm_start=True).
        
        We'll produce ONE entry in finetune_loss_history / finetune_r2_history
        after this call.
        """
        self.finetune_loss_history.clear()
        self.finetune_r2_history.clear()

        # Increase n_estimators by 'fine_tune_trees'
        old_n = self.model.n_estimators
        new_n = old_n + self.fine_tune_trees

        # Re-fit with warm_start=True => keeps old trees, just adds new ones
        self.model.set_params(warm_start=True, n_estimators=new_n)
        self.model.fit(X_train, y_train)

        if X_eval is not None and y_eval is not None:
            y_pred = self.model.predict(X_eval)
            rmse = sqrt(mean_squared_error(y_eval, y_pred))
            r2 = r2_score(y_eval, y_pred)
            self.finetune_loss_history.append(rmse)
            self.finetune_r2_history.append(r2)

    def predict(self, X_test):
        """
        Predict using the trained/fine-tuned model.
        """
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """
        Save the entire random forest to a file.
        """
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """
        Load the model from a file.
        """
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
        else:
            print(f"Model file {filepath} does not exist.")

    def get_epoch_metrics(self):
        """
        Return the metrics dictionary, consistent with MonteCarloSimulation usage:
          - 'loss_pretrain'  => one-element list containing RMSE after the pre-train phase
          - 'r2_pretrain'    => one-element list containing R² after the pre-train phase
          - 'loss_finetune'  => one-element list containing RMSE after the fine-tune phase
          - 'r2_finetune'    => one-element list containing R² after the fine-tune phase
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
