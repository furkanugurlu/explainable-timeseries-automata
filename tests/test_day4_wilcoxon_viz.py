import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent
root_dir    = current_dir.parent
sys.path.append(str(root_dir))


class TestWilcoxonFunction(unittest.TestCase):
    """perform_wilcoxon_test must handle all edge cases correctly."""

    def _wt(self):
        from src.visualization.visualization_and_stats import perform_wilcoxon_test
        return perform_wilcoxon_test

    def test_returns_dict_with_required_keys(self):
        result = self._wt()(
            [0.6, 0.7, 0.8, 0.5, 0.9],
            [0.4, 0.3, 0.2, 0.5, 0.1],
            "ModelA", "ModelB",
        )
        self.assertIsNotNone(result)
        for key in ('stat', 'p_value', 'significant'):
            self.assertIn(key, result, f"Missing key: '{key}'")

    def test_identical_groups_returns_p1(self):
        group = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = self._wt()(group, group, "A", "B")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['p_value'], 1.0, places=5)
        self.assertFalse(result['significant'])

    def test_clearly_different_groups_significant(self):
        # Wilcoxon minimum achievable p with n=5 is 0.0625 > 0.05, so we need n>=8
        result = self._wt()(
            [0.90, 0.95, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87],
            [0.10, 0.05, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13],
            "High", "Low",
        )
        self.assertIsNotNone(result)
        self.assertTrue(result['significant'], "Clearly different groups (n=8) must be significant")

    def test_size_mismatch_returns_none(self):
        result = self._wt()([0.5, 0.6], [0.5, 0.6, 0.7], "A", "B")
        self.assertIsNone(result)

    def test_single_element_returns_none(self):
        result = self._wt()([0.5], [0.6], "A", "B")
        self.assertIsNone(result)

    def test_p_value_in_valid_range(self):
        result = self._wt()(
            [0.80, 0.75, 0.82, 0.79, 0.85],
            [0.60, 0.55, 0.62, 0.59, 0.65],
            "A", "B",
        )
        if result is not None:
            self.assertGreaterEqual(result['p_value'], 0.0)
            self.assertLessEqual(result['p_value'], 1.0)

    def test_significant_flag_matches_p_threshold(self):
        result = self._wt()(
            [0.90, 0.95, 0.92, 0.88, 0.91],
            [0.10, 0.05, 0.08, 0.12, 0.09],
            "A", "B",
        )
        if result is not None:
            expected = result['p_value'] < 0.05
            self.assertEqual(result['significant'], expected)


class TestVisualizationSmoke(unittest.TestCase):
    """All 5 visualization functions must complete without exceptions."""

    def test_plot_confusion_matrix_smoke(self):
        from src.visualization.visualization_and_stats import plot_confusion_matrix
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        try:
            plot_confusion_matrix(y_true, y_pred, "day4_smoke")
        except Exception as exc:
            self.fail(f"plot_confusion_matrix raised: {exc}")

    def test_plot_roc_pr_two_class_smoke(self):
        from src.visualization.visualization_and_stats import plot_roc_pr_curves
        rng    = np.random.RandomState(0)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = rng.rand(10)
        try:
            plot_roc_pr_curves(y_true, y_prob, "day4_smoke")
        except Exception as exc:
            self.fail(f"plot_roc_pr_curves raised: {exc}")

    def test_plot_roc_pr_single_class_no_crash(self):
        """Guard: single-class y_true must return early, not raise."""
        from src.visualization.visualization_and_stats import plot_roc_pr_curves
        y_true = np.zeros(10, dtype=int)
        y_prob = np.random.rand(10)
        try:
            plot_roc_pr_curves(y_true, y_prob, "single_class_smoke")
        except Exception as exc:
            self.fail(f"Single-class ROC raised instead of returning early: {exc}")

    def test_plot_hyperparameter_heatmap_smoke(self):
        from src.visualization.visualization_and_stats import plot_hyperparameter_heatmap
        df = pd.DataFrame({
            'window':   [3, 3, 4, 4],
            'alphabet': [3, 4, 3, 4],
            'f1':       [0.70, 0.75, 0.80, 0.85],
        })
        try:
            plot_hyperparameter_heatmap(df, metric='f1')
        except Exception as exc:
            self.fail(f"plot_hyperparameter_heatmap raised: {exc}")

    def test_plot_transition_heatmap_smoke(self):
        from src.visualization.visualization_and_stats import plot_transition_heatmap
        probs = {'aaa': {'aab': 0.8, 'aaa': 0.2}, 'aab': {'aba': 1.0}}
        try:
            plot_transition_heatmap(probs)
        except Exception as exc:
            self.fail(f"plot_transition_heatmap raised: {exc}")

    def test_draw_automata_graph_smoke(self):
        from src.visualization.visualization_and_stats import draw_automata_graph
        probs = {'aaa': {'aab': 0.8, 'aaa': 0.2}, 'aab': {'aba': 1.0}}
        try:
            draw_automata_graph(probs)
        except Exception as exc:
            self.fail(f"draw_automata_graph raised: {exc}")


if __name__ == '__main__':
    unittest.main()
