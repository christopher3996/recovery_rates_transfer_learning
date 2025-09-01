import os
import torch
import numpy as np
from sklearn.metrics import r2_score
from models.base_model import BaseModel
from tqdm.notebook import tqdm as notebook_tqdm


# Import WideDeep components from pytorch-widedeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep.training import Trainer


class WideDeepRegression(BaseModel):
    """
    A regression model built using pytorch-widedeep.

    This class uses only the deeptabular component (TabMlp) to model continuous data.
    It supports a pretraining phase and fine-tuning on a new (target) dataset.

    The model extracts the continuous feature names directly from X_train,
    so you do not need to supply them in the parameter dictionary.

    Parameters:
        input_size (int): Number of input features (for compatibility; not used for column names).
        mlp_hidden_dims (list): Hidden layer dimensions for TabMlp.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        seed (int): Random seed.
        deterministic (bool): Enforces deterministic behavior if True.
        benchmark (bool): Enables cudnn.benchmark if True.
    """
    def __init__(self,
                 input_size,
                 mlp_hidden_dims=[64, 32],
                 learning_rate=0.001,
                 epochs=50,
                 batch_size=32,
                 seed=42,
                 deterministic=True,
                 benchmark=False):
        super().__init__(seed=seed, deterministic=deterministic, benchmark=benchmark)
        print(f"[INIT] WideDeepRegression initialized with input_size={input_size}, "
              f"mlp_hidden_dims={mlp_hidden_dims}, learning_rate={learning_rate}, "
              f"epochs={epochs}, batch_size={batch_size}")
        self.input_size = input_size  
        self.mlp_hidden_dims = mlp_hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # These will be created when training begins:
        self.tab_preprocessor = None
        self.tab_mlp = None
        self.model = None
        self.trainer = None

        self.history_pretrain = None
        self.history_finetune = None
        Trainer.tqdm = notebook_tqdm
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INIT] Using device: {self.device}")
    
    def _build_model(self, X_train):
        continuous_cols = list(X_train.columns)

        # Tell TabPreprocessor we only have continuous columns
        # so cat_embed_input should end up empty or None.
        self.tab_preprocessor = TabPreprocessor(
            embed_cols=None, 
            continuous_cols=continuous_cols
        )
        _ = self.tab_preprocessor.fit_transform(X_train)

        # If cat_embed_input is empty, set it to None
        cat_embed_input = getattr(self.tab_preprocessor, "cat_embed_input", None)
        if cat_embed_input and len(cat_embed_input) == 0:
            cat_embed_input = None

        self.tab_mlp = TabMlp(
            column_idx=self.tab_preprocessor.column_idx,
            cat_embed_input=cat_embed_input,  # <= might be None
            continuous_cols=continuous_cols,
            mlp_hidden_dims=self.mlp_hidden_dims,
        )
        self.model = WideDeep(deeptabular=self.tab_mlp, pred_dim=1)
        self.model.to(self.device)

    
    def train(self, X_train, y_train, X_eval, y_eval):
        print("[train] Starting training phase")
        if self.model is None:
            print("[train] Model not built yet; building model...")
            self._build_model(X_train)
        else:
            print("[train] Model already built; skipping _build_model()")

        # Transform
        X_tab_train = self.tab_preprocessor.transform(X_train)
        X_tab_val   = self.tab_preprocessor.transform(X_eval)

        self.trainer = Trainer(self.model, objective="regression")

        # Dictionary mode
        train_dict = {"X_tab": X_tab_train, "target": y_train.values}
        val_dict   = {"X_tab": X_tab_val,   "target": y_eval.values}

        # IMPORTANT: Use X_train=... not X_tab=...
        self.history_pretrain = self.trainer.fit(
            X_train=train_dict,
            X_val=val_dict,
            n_epochs=self.epochs,
            batch_size=self.batch_size
        )
        print("[train] Training complete.")

    
    def fine_tune(self, X_train, y_train, X_eval, y_eval):
        print("[fine_tune] Starting fine-tuning phase")
        X_tab_train = self.tab_preprocessor.transform(X_train)
        X_tab_val   = self.tab_preprocessor.transform(X_eval)

        self.trainer = Trainer(self.model, objective="regression")

        train_dict = {"X_tab": X_tab_train, "target": y_train.values}
        val_dict   = {"X_tab": X_tab_val,   "target": y_eval.values}

        # again, use X_train=..., not X_tab=...
        self.history_finetune = self.trainer.fit(
            X_train=train_dict,
            X_val=val_dict,
            n_epochs=self.epochs,
            batch_size=self.batch_size
        )
        print("[fine_tune] Fine-tuning complete.")

    
    def predict(self, X_test):
        """
        Generate predictions on new data.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predictions.
        """
        print("[predict] Transforming test data...")
        print(type(X_test))
        print("-----")
        X_tab_test = self.tab_preprocessor.transform(X_test.to_numpy())
        print(f"[predict] Test data transformed. Type: {type(X_tab_test)}")
        print("[predict] Forcing dictionary mode for prediction.")
        test_dict = {"X_tab": X_tab_test}
        preds = self.trainer.predict(X_tab=test_dict, batch_size=self.batch_size)
        print("[predict] Prediction complete.")
        return preds
    
    def save_model(self, filepath):
        """
        Save the model state to the given filepath.
        """
        print(f"[save_model] Saving model to {filepath}")
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """
        Load a model state from the given filepath.
        """
        print(f"[load_model] Loading model from {filepath}")
        if os.path.exists(filepath):
            state = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            print("[load_model] Model loaded successfully.")
        else:
            print(f"[load_model] Model file {filepath} does not exist.")
    
    def get_epoch_metrics(self):
        """
        Return training history for pretraining and fine-tuning if available.

        Returns:
            dict: Dictionary with keys 'pretrain' and/or 'finetune'.
        """
        metrics = {}
        if self.history_pretrain is not None:
            metrics['pretrain'] = self.history_pretrain
        if self.history_finetune is not None:
            metrics['finetune'] = self.history_finetune
        print(f"[get_epoch_metrics] Returning metrics: {metrics}")
        return metrics
