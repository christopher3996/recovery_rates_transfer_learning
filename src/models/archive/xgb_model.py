import os
import joblib
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    """
    An XGBoost regression model supporting a two-phase approach:
      1) Pre-train on source data
      2) Fine-tune on target data

    Each phase tracks epoch-wise training RMSE and validation R².
    """

    def __init__(self,
                 xgb_params=None,
                 epochs=50,
                 seed=42,
                 deterministic=True,
                 benchmark=False,
                 **kwargs):
        """
        Initialize the XGBoost model.

        Args:
            xgb_params (dict): Additional XGBoost hyperparameters (e.g. 'max_depth', 'learning_rate', etc.).
            epochs (int): Number of boosting rounds for each training/fine-tuning call.
            seed (int): Random seed for reproducibility (passed to BaseModel and XGBoost).
            deterministic (bool): If True, sets PyTorch (not directly used here) in deterministic mode.
            benchmark (bool): If True, enables cudnn.benchmark in PyTorch (no effect here, but keeps signature).
        """
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)

        if xgb_params is None:
            xgb_params = {}

        # If the user hasn't specified 'n_estimators', we set it from epochs
        if 'n_estimators' not in xgb_params:
            xgb_params['n_estimators'] = epochs

        # Ensure we pass seed to XGB
        if 'random_state' not in xgb_params:
            xgb_params['random_state'] = seed

        self.epochs = epochs  # We'll store this in case we need it again
        self.xgb_params = xgb_params

        # Create the underlying XGBRegressor model
        self.model = XGBRegressor(enable_categorical=True, **self.xgb_params)

        # Buffers for storing epoch-wise metrics
        # "train_loss_history"  = RMSE on training set (source or target)
        # "train_r2_history"    = R² on evaluation set (source_test or target_test) during pre-training
        # "finetune_loss_history" = RMSE on training set (target) during fine-tuning
        # "finetune_r2_history"   = R² on evaluation set (target_test) during fine-tuning
        self.train_loss_history = []
        self.train_r2_history = []
        self.finetune_loss_history = []
        self.finetune_r2_history = []

    def _r2_metric(self, y_pred, dtrain):
        """
        A custom evaluation metric for XGBoost to calculate R² at each iteration.
        We return (metric_name, metric_value, is_higher_better).
        """
        y_true = dtrain.get_label()
        return ('r2', r2_score(y_true, y_pred), True)

    def train(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """
        Pre-train (or initial train) on source data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Source training features.
            y_train (pd.Series or np.ndarray): Source training targets.
            X_eval (pd.DataFrame or np.ndarray): Evaluation features (source_test or target_test).
            y_eval (pd.Series or np.ndarray): Evaluation targets.
        """
        # Clear old histories in case we're re-using this model
        self.train_loss_history.clear()
        self.train_r2_history.clear()

        # Build eval_set for XGBoost
        eval_set = []
        if X_eval is not None and y_eval is not None:
            # We'll track both training-set metrics and validation-set metrics
            eval_set = [(X_train, y_train), (X_eval, y_eval)]

        # Fit from scratch (i.e., no xgb_model=...)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            #eval_metric=["rmse", self._r2_metric],
            verbose=False
        )

        # After training, retrieve the per-iteration logs
        evals_result = self.model.evals_result()

        # The dictionary structure is:
        # {
        #   'validation_0': {'rmse': [...], 'r2': [...]},
        #   'validation_1': {'rmse': [...], 'r2': [...]}
        # }
        # We'll treat 'validation_0' as the *training set*, 'validation_1' as the eval set.

        if 'validation_0' in evals_result and 'rmse' in evals_result['validation_0']:
            self.train_loss_history = evals_result['validation_0']['rmse']
        if 'validation_1' in evals_result and 'r2' in evals_result['validation_1']:
            self.train_r2_history = evals_result['validation_1']['r2']

    def fine_tune(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """
        Fine-tune the model on new training data (typically target domain).

        Args:
            X_train (pd.DataFrame or np.ndarray): Target training features.
            y_train (pd.Series or np.ndarray): Target training targets.
            X_eval (pd.DataFrame or np.ndarray): Evaluation features (target_test).
            y_eval (pd.Series or np.ndarray): Evaluation targets (target_test).
        """
        # Clear old fine-tune histories
        self.finetune_loss_history.clear()
        self.finetune_r2_history.clear()

        eval_set = []
        if X_eval is not None and y_eval is not None:
            eval_set = [(X_train, y_train), (X_eval, y_eval)]

        # Continue training from the existing booster
        booster = self.model.get_booster()  # The underlying trained booster
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            #eval_metric=["rmse", self._r2_metric],
            verbose=False,
            xgb_model=booster  # This keeps adding trees
        )
        for col, importance in zip(X_train.columns, self.model.feature_importances_):
            print(f"{col}: {importance:.4f}")

        evals_result = self.model.evals_result()
        if 'validation_0' in evals_result and 'rmse' in evals_result['validation_0']:
            self.finetune_loss_history = evals_result['validation_0']['rmse']
        if 'validation_1' in evals_result and 'r2' in evals_result['validation_1']:
            self.finetune_r2_history = evals_result['validation_1']['r2']

    def predict(self, X_test):
        """
        Predict using the trained/fine-tuned model.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.

        Returns:
            np.ndarray: Predictions.
        """
        return self.model.predict(X_test)

    def save_model(self, filepath):
        """
        Save the model (entire XGBoost booster) to a file.
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
        Return the epoch-wise metrics as a dict consistent with the MonteCarloSimulation usage:
            - 'loss_pretrain':  training-set RMSE across pre-training epochs
            - 'r2_pretrain':    validation-set R² across pre-training epochs
            - 'loss_finetune':  training-set RMSE across fine-tuning epochs
            - 'r2_finetune':    validation-set R² across fine-tuning epochs
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
