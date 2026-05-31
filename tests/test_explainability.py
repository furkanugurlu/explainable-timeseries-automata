import sys
import json
import unittest
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))


class TestAutomataExplainerUnit(unittest.TestCase):
    """Unit tests for AutomataExplainer standalone behavior."""

    def setUp(self):
        from src.models.automata_model import ProbabilisticAutomata
        from src.models.explainability import AutomataExplainer
        self.model = ProbabilisticAutomata()
        train = ['ab', 'bc', 'ca', 'ab', 'bc', 'ca', 'ab', 'bc', 'ca'] * 5
        self.model.fit(train, window_len=3)
        self.explainer = AutomataExplainer(self.model)

    def test_explain_anomalies_returns_list(self):
        """explain_anomalies(json_output=False) must return a list."""
        seq = ['ab', 'bc', 'ca', 'ab', 'bc', 'ca']
        result = self.explainer.explain_anomalies(seq, window_len=3, json_output=False)
        self.assertIsInstance(result, list,
            "explain_anomalies() must return a list when json_output=False")

    def test_explanation_has_required_keys(self):
        """Each explanation entry must have the required keys."""
        from src.models.automata_model import ProbabilisticAutomata
        from src.models.explainability import AutomataExplainer
        model = ProbabilisticAutomata()
        # Force anomaly: set a very high threshold
        train = ['ab', 'bc', 'ca'] * 5
        model.fit(train, window_len=3)
        model.anomaly_threshold = 1.0  # every window will be anomalous
        explainer = AutomataExplainer(model)
        seq = ['ab', 'bc', 'ca', 'ab']
        results = explainer.explain_anomalies(seq, window_len=3, json_output=False)
        required = {'time_step', 'window_sequence', 'pattern', 'status',
                    'mapped_to', 'transitions', 'path_probability',
                    'decision', 'confidence_score'}
        for entry in results:
            for key in required:
                self.assertIn(key, entry,
                    f"Explanation entry missing required key: '{key}'")

    def test_explain_anomalies_json_output_is_valid_json(self):
        """explain_anomalies(json_output=True) must return valid JSON string."""
        from src.models.automata_model import ProbabilisticAutomata
        from src.models.explainability import AutomataExplainer
        model = ProbabilisticAutomata()
        train = ['ab', 'bc', 'ca'] * 5
        model.fit(train, window_len=3)
        model.anomaly_threshold = 1.0
        explainer = AutomataExplainer(model)
        seq = ['ab', 'bc', 'ca', 'ab']
        json_str = explainer.explain_anomalies(seq, window_len=3, json_output=True)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            self.fail("explain_anomalies(json_output=True) returned invalid JSON")
        self.assertIsInstance(parsed, list)


class TestAutomataPipelineExplainability(unittest.TestCase):
    """AutomataPipeline must expose explanations and save them to disk."""

    def _fitted_pipeline(self):
        from src.pipeline.automata_pipeline import AutomataPipeline
        rng = np.random.RandomState(99)
        X_tr = rng.rand(300, 1)
        y_tr = rng.randint(0, 2, 300)
        p = AutomataPipeline(window_size=4, alphabet_size=3)
        p.fit(X_tr, y_tr)
        return p, rng

    def test_pipeline_has_get_explanations_method(self):
        """AutomataPipeline must have a get_explanations() method (rubric Kriter3 — 20pt)."""
        from src.pipeline.automata_pipeline import AutomataPipeline
        p = AutomataPipeline(window_size=4, alphabet_size=3)
        self.assertTrue(hasattr(p, 'get_explanations'),
            "AutomataPipeline must expose get_explanations() for the explainability module")

    def test_get_explanations_returns_list(self):
        """get_explanations() must return a list of explanation dicts."""
        pipeline, rng = self._fitted_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        pipeline.predict(X_te)
        result = pipeline.get_explanations()
        self.assertIsInstance(result, list,
            "get_explanations() must return a list")

    def test_get_explanations_before_predict_raises_runtime_error(self):
        """get_explanations() called before predict() must raise RuntimeError."""
        from src.pipeline.automata_pipeline import AutomataPipeline
        rng = np.random.RandomState(42)
        X_tr = rng.rand(200, 1)
        y_tr = rng.randint(0, 2, 200)
        pipeline = AutomataPipeline(window_size=4, alphabet_size=3)
        pipeline.fit(X_tr, y_tr)
        with self.assertRaises(RuntimeError,
                msg="get_explanations() before predict() must raise RuntimeError"):
            pipeline.get_explanations()

    def test_save_explanations_creates_json_file(self):
        """save_explanations() must write a JSON file to the given path."""
        from src.pipeline.automata_pipeline import AutomataPipeline
        import tempfile
        pipeline, rng = self._fitted_pipeline()
        X_te = rng.rand(100, 1)
        y_te = rng.randint(0, 2, 100)
        pipeline.predict(X_te)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_explanations.json"
            pipeline.save_explanations(str(out_path))
            self.assertTrue(out_path.exists(),
                "save_explanations() must create the JSON file")
            with open(out_path) as f:
                data = json.load(f)
            self.assertIsInstance(data, list,
                "Saved JSON must be a list of explanations")


if __name__ == '__main__':
    unittest.main()
