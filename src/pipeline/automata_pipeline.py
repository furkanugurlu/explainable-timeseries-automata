import logging
import time
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.base_model import BaseAnomalyDetector
from src.models.automata_transform import SAXTransformer
from src.models.automata_model import ProbabilisticAutomata

logger = logging.getLogger(__name__)


class AutomataPipeline(BaseAnomalyDetector):
    """
    Pipeline pattern: SAXTransformer → ProbabilisticAutomata as a single unit.

    Implements BaseAnomalyDetector so it can be swapped with DL models
    in the experiment loop without any changes to main.py (Strategy pattern).

    All parameters come from config — no hardcoded values (spec VIII).
    """

    def __init__(self, window_size: int, alphabet_size: int):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.transformer = SAXTransformer(window_size=window_size, alphabet_size=alphabet_size)
        self.model = ProbabilisticAutomata()
        self._train_time: float = 0.0
        self._y_pred: np.ndarray = None
        self._fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> "AutomataPipeline":
        t0 = time.time()

        train_sym = self.transformer.fit_transform(X_train)
        train_patterns = self.transformer.extract_patterns(train_sym, pattern_length=self.window_size)
        self.model.fit(train_patterns, window_len=self.window_size)

        self._train_time = time.time() - t0
        self._fitted = True
        logger.info(
            f"AutomataPipeline fitted (W={self.window_size}, A={self.alphabet_size}) "
            f"in {self._train_time:.3f}s — {len(self.model.states)} states"
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "AutomataPipeline.predict() called before fit(). Call fit(X_train, y_train) first."
            )
        test_sym = self.transformer.transform(X_test)
        test_patterns = self.transformer.extract_patterns(test_sym, pattern_length=self.window_size)
        _, labels = self.model.predict_anomalies(test_patterns, window_len=self.window_size)
        self._y_pred = np.array(labels)
        return self._y_pred

    def get_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        t1 = time.time()
        preds = self.predict(X_test)
        inference_time = time.time() - t1

        # Align label length — windowing shrinks the prediction array
        aligned_y = y_test[len(y_test) - len(preds):]

        return {
            "accuracy":          float(accuracy_score(aligned_y, preds)),
            "precision":         float(precision_score(aligned_y, preds, zero_division=0)),
            "recall":            float(recall_score(aligned_y, preds, zero_division=0)),
            "f1":                float(f1_score(aligned_y, preds, zero_division=0)),
            "state_count":       int(len(self.model.states)),
            "density":           self._transition_density(),
            "train_time_sec":    self._train_time,
            "inference_time_sec": inference_time,
        }

    def _transition_density(self) -> float:
        n = len(self.model.states)
        if n == 0:
            return 0.0
        actual = sum(len(v) for v in self.model.probabilities.values())
        return actual / (n ** 2)
