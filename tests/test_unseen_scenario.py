import sys
import unittest
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))


class TestUnseenPatternTracking(unittest.TestCase):
    """ProbabilisticAutomata must track which patterns are unseen during prediction."""

    def setUp(self):
        from src.models.automata_model import ProbabilisticAutomata
        self.model = ProbabilisticAutomata()
        # Train on a limited pattern set so test patterns are 'unseen'
        train_patterns = ['ab', 'bc', 'ca', 'ab', 'bc', 'ca', 'ab', 'bc', 'ca'] * 5
        self.model.fit(train_patterns, window_len=3)

    def test_predict_anomalies_returns_unseen_flags(self):
        """predict_anomalies() must return a third element: list of bools marking unseen windows."""
        test_patterns = ['ab', 'bc', 'zz', 'ab']  # 'zz' is unseen
        result = self.model.predict_anomalies(test_patterns, window_len=3)
        self.assertEqual(len(result), 3,
            "predict_anomalies() must return (probs, labels, unseen_flags) — 3 elements")

    def test_unseen_flags_correct_length(self):
        """unseen_flags list length must match labels list length."""
        test_patterns = ['ab', 'bc', 'zz', 'ab', 'bc']
        probs, labels, unseen_flags = self.model.predict_anomalies(test_patterns, window_len=3)
        self.assertEqual(len(unseen_flags), len(labels),
            "unseen_flags length must match labels length")

    def test_known_patterns_not_flagged_unseen(self):
        """Windows containing only known patterns must have unseen_flag=False."""
        # All patterns known from training
        test_patterns = ['ab', 'bc', 'ca', 'ab', 'bc']
        _, _, unseen_flags = self.model.predict_anomalies(test_patterns, window_len=3)
        self.assertFalse(any(unseen_flags),
            "No window should be flagged unseen when all patterns are known")

    def test_unseen_pattern_triggers_flag(self):
        """A window containing an unseen pattern must have unseen_flag=True."""
        # 'zz' is not in training — Levenshtein maps it, but it's still 'unseen'
        test_patterns = ['ab', 'zz', 'ca']
        _, _, unseen_flags = self.model.predict_anomalies(test_patterns, window_len=3)
        self.assertTrue(unseen_flags[0],
            "Window containing unseen pattern 'zz' must be flagged as unseen=True")


class TestDetectionRateAndMappingAccuracy(unittest.TestCase):
    """AutomataPipeline.get_metrics() must compute Detection Rate + Mapping Accuracy
    for windows that contain unseen patterns (Tablo 2 requirement)."""

    def _make_pipeline(self):
        from src.pipeline.automata_pipeline import AutomataPipeline
        rng = np.random.RandomState(42)
        X_tr = rng.rand(300, 1)
        y_tr = rng.randint(0, 2, 300)
        pipeline = AutomataPipeline(window_size=4, alphabet_size=3)
        pipeline.fit(X_tr, y_tr)
        return pipeline, rng

    def test_get_metrics_returns_detection_rate(self):
        """get_metrics() must include 'detection_rate' key (Tablo 2 — spec VI)."""
        pipeline, rng = self._make_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        metrics = pipeline.get_metrics(X_te, y_te)
        self.assertIn('detection_rate', metrics,
            "'detection_rate' missing from get_metrics() — required for Tablo 2")

    def test_get_metrics_returns_mapping_accuracy(self):
        """get_metrics() must include 'mapping_accuracy' key (Tablo 2 — spec VI)."""
        pipeline, rng = self._make_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        metrics = pipeline.get_metrics(X_te, y_te)
        self.assertIn('mapping_accuracy', metrics,
            "'mapping_accuracy' missing from get_metrics() — required for Tablo 2")

    def test_get_metrics_returns_unseen_window_count(self):
        """get_metrics() must include 'unseen_window_count' for reporting."""
        pipeline, rng = self._make_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        metrics = pipeline.get_metrics(X_te, y_te)
        self.assertIn('unseen_window_count', metrics)

    def test_detection_rate_is_float_between_0_and_1(self):
        """detection_rate must be a float in [0, 1]."""
        pipeline, rng = self._make_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        metrics = pipeline.get_metrics(X_te, y_te)
        dr = metrics['detection_rate']
        self.assertIsInstance(dr, float)
        self.assertGreaterEqual(dr, 0.0)
        self.assertLessEqual(dr, 1.0)

    def test_mapping_accuracy_is_float_between_0_and_1(self):
        """mapping_accuracy must be a float in [0, 1], or None if no unseen windows exist."""
        pipeline, rng = self._make_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        metrics = pipeline.get_metrics(X_te, y_te)
        ma = metrics['mapping_accuracy']
        if ma is not None:
            self.assertIsInstance(ma, float)
            self.assertGreaterEqual(ma, 0.0)
            self.assertLessEqual(ma, 1.0)

    def test_unseen_window_count_is_non_negative_int(self):
        """unseen_window_count must be a non-negative integer."""
        pipeline, rng = self._make_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        metrics = pipeline.get_metrics(X_te, y_te)
        count = metrics['unseen_window_count']
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
