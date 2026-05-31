from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseAnomalyDetector(ABC):
    """
    Common interface for all anomaly detection models (Strategy pattern).
    Ensures every model — LSTM, GRU, CNN1D, Automata — can be swapped
    without changing the experiment loop in main.py.
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> "BaseAnomalyDetector":
        """Train the model. Returns self for method chaining."""
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary predictions array (0 = normal, 1 = anomaly)."""
        ...

    @abstractmethod
    def get_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Run predict() and return a dict with at minimum:
            accuracy, precision, recall, f1, train_time_sec, inference_time_sec
        """
        ...
