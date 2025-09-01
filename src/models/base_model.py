from abc import ABC, abstractmethod
from utils.seed import set_seed    

class BaseModel(ABC):
    def __init__(self, seed=42, deterministic=True, benchmark=False, **kwargs):
        """
        Initialize the base model, potentially re-seeding each time.

        Args:
            seed (int): Random seed for reproducibility.
            deterministic (bool): Whether to enforce PyTorch's deterministic mode.
            benchmark (bool): Whether to enable cudnn.benchmark for potential speed-ups.
        """
        # Re-seed the entire environment each time a model is created
        set_seed(seed=seed, deterministic=deterministic, benchmark=benchmark)

    
    @abstractmethod
    def train(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """
        Train the model on the training data.
        """
        pass

    def fine_tune(self, X_train, y_train, X_eval=None, y_eval=None, **kwargs):
        """
        Fine-tune the model on new training data.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict using the model.
        """
        pass

    def save_model(self, filepath):
        """
        Save the model to a file.
        """
        pass

    def load_model(self, filepath):
        """
        Load the model from a file.
        """
        pass

    def get_epoch_metrics(self):
        """
        Return a dictionary of epoch-wise metrics.
        For example: {"loss": [loss_per_epoch], "r2": [r2_per_epoch]}
        If no epoch-wise tracking is implemented, return an empty dict.
        """
        return {}
