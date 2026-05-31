import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.base_model import BaseAnomalyDetector
from src.models.automata_transform import SAXTransformer
from src.models.automata_model import ProbabilisticAutomata
from src.models.explainability import AutomataExplainer

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
        self._y_pred: Optional[np.ndarray] = None
        self._unseen_flags: Optional[List[bool]] = None
        self._path_probs: Optional[List[float]] = None
        self._test_patterns: Optional[List[str]] = None
        self._explanations: Optional[List[Dict]] = None
        self._fitted: bool = False
        self._predicted: bool = False

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
        self._predicted = False
        self._path_probs = None
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

        path_probs, labels, unseen_flags = self.model.predict_anomalies(test_patterns, window_len=self.window_size)

        self._test_patterns = test_patterns
        self._unseen_flags = unseen_flags
        self._path_probs = path_probs  # stored for ROC/PR curves (Day 4)
        self._y_pred = np.array(labels)
        self._explanations = None  # invalidate cached explanations on new predict
        self._predicted = True
        return self._y_pred

    def get_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        t1 = time.time()
        preds = self.predict(X_test)
        inference_time = time.time() - t1

        # Align label length — windowing shrinks the prediction array
        aligned_y = y_test[len(y_test) - len(preds):]
        unseen_flags = np.array(self._unseen_flags)

        # --- Base metrics (all windows) ---
        base = {
            "accuracy":           float(accuracy_score(aligned_y, preds)),
            "precision":          float(precision_score(aligned_y, preds, zero_division=0)),
            "recall":             float(recall_score(aligned_y, preds, zero_division=0)),
            "f1":                 float(f1_score(aligned_y, preds, zero_division=0)),
            "state_count":        int(len(self.model.states)),
            "density":            self._transition_density(),
            "train_time_sec":     self._train_time,
            "inference_time_sec": inference_time,
        }

        # --- Unseen-window metrics (Tablo 2 — spec VI) ---
        unseen_count = int(unseen_flags.sum())
        base["unseen_window_count"] = unseen_count

        if unseen_count > 0:
            mask = unseen_flags.astype(bool)
            sub_y = aligned_y[mask]
            sub_p = preds[mask]
            base["detection_rate"]   = float(recall_score(sub_y, sub_p, zero_division=0))
            base["mapping_accuracy"] = float(accuracy_score(sub_y, sub_p))
        else:
            base["detection_rate"]   = 0.0
            base["mapping_accuracy"] = 0.0

        logger.debug(
            f"Unseen windows: {unseen_count} | "
            f"detection_rate={base['detection_rate']:.3f} | "
            f"mapping_accuracy={base['mapping_accuracy']:.3f}"
        )
        return base

    def get_path_probabilities(self) -> List[float]:
        """
        Returns raw path probabilities from the last predict() call.
        Lower probability = higher anomaly score — used as anomaly scores for ROC/PR curves.
        Must be called after predict().
        """
        if not self._predicted or self._path_probs is None:
            raise RuntimeError(
                "AutomataPipeline.get_path_probabilities() called before predict(). "
                "Call predict(X_test) first."
            )
        return self._path_probs

    def get_explanations(self) -> List[Dict]:
        """
        Returns explanations for anomalous windows (rubric Kriter3 — Açıklanabilirlik, 20pt).
        Must be called after predict().
        """
        if not self._predicted or self._test_patterns is None:
            raise RuntimeError(
                "AutomataPipeline.get_explanations() called before predict(). "
                "Call predict(X_test) first."
            )
        if self._explanations is None:
            explainer = AutomataExplainer(self.model)
            self._explanations = explainer.explain_anomalies(
                self._test_patterns, window_len=self.window_size, json_output=False
            )
        return self._explanations

    def save_explanations(self, path: str) -> None:
        """Saves anomaly explanations as a JSON file to the given path."""
        explanations = self.get_explanations()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=2)
        logger.info(f"Saved {len(explanations)} explanations to {path}")

    def _transition_density(self) -> float:
        n = len(self.model.states)
        if n == 0:
            return 0.0
        actual = sum(len(v) for v in self.model.probabilities.values())
        return actual / (n ** 2)
