import sys
import unittest
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))


class TestInfrastructure(unittest.TestCase):

    def test_setup_experiment_dirs_creates_required_paths(self):
        """setup_experiment_dirs() must create results/, results/figures/, and logs/."""
        from main import setup_experiment_dirs
        setup_experiment_dirs()
        self.assertTrue(Path("results").exists(), "results/ must be created")
        self.assertTrue(Path("results/figures").exists(), "results/figures/ must be created")
        self.assertTrue(Path("logs").exists(), "logs/ must be created")


class TestSKABLoader(unittest.TestCase):

    def test_get_folds_yields_exactly_five(self):
        """SKABDataLoader.get_folds(n_splits=5) must yield exactly 5 complete folds, not just 1."""
        from src.data.data_loader_skab import SKABDataLoader
        loader = SKABDataLoader()
        folds = list(loader.get_folds(n_splits=5))
        self.assertEqual(len(folds), 5, "All 5 GroupKFold splits must be produced")

    def test_each_fold_returns_four_arrays(self):
        """Each fold must unpack as (X_tr, X_te, y_tr, y_te)."""
        from src.data.data_loader_skab import SKABDataLoader
        loader = SKABDataLoader()
        fold = next(loader.get_folds(n_splits=5))
        self.assertEqual(len(fold), 4, "Fold must return exactly 4 arrays")


class TestAutomataMetrics(unittest.TestCase):

    def test_run_automata_pipeline_returns_precision(self):
        """run_automata_pipeline() result dict must contain 'precision' key (required by spec IX)."""
        from main import run_automata_pipeline
        rng = np.random.RandomState(42)
        X = rng.rand(300, 1)
        y = rng.randint(0, 2, 300)
        result = run_automata_pipeline(X[:200], y[:200], X[200:], y[200:], w_size=4, a_size=3)
        self.assertIn('precision', result)

    def test_run_automata_pipeline_returns_recall(self):
        """run_automata_pipeline() result dict must contain 'recall' key (required by spec IX)."""
        from main import run_automata_pipeline
        rng = np.random.RandomState(42)
        X = rng.rand(300, 1)
        y = rng.randint(0, 2, 300)
        result = run_automata_pipeline(X[:200], y[:200], X[200:], y[200:], w_size=4, a_size=3)
        self.assertIn('recall', result)


class TestAutomataModel(unittest.TestCase):

    def test_fit_accepts_window_len_parameter(self):
        """ProbabilisticAutomata.fit() must accept window_len — no hardcoded values (spec VIII)."""
        from src.models.automata_model import ProbabilisticAutomata
        automata = ProbabilisticAutomata()
        patterns = ['ab', 'bc', 'ca'] * 10
        try:
            automata.fit(patterns, window_len=4)
        except TypeError:
            self.fail("fit() raised TypeError — window_len parameter is missing (hardcoded value violation)")
        self.assertGreater(automata.anomaly_threshold, 0)

    def test_fit_window_len_affects_threshold(self):
        """Different window_len values must produce different anomaly thresholds."""
        from src.models.automata_model import ProbabilisticAutomata
        patterns = ['ab', 'bc', 'ca', 'ab', 'ba', 'cc', 'ab', 'bc', 'ca'] * 5

        m1 = ProbabilisticAutomata()
        m1.fit(patterns, window_len=3)

        m2 = ProbabilisticAutomata()
        m2.fit(patterns, window_len=6)

        self.assertNotEqual(m1.anomaly_threshold, m2.anomaly_threshold,
                            "window_len=3 and window_len=6 must produce different thresholds")


class TestAutomataPipeline(unittest.TestCase):

    def test_pipeline_is_instance_of_base_detector(self):
        """AutomataPipeline must implement BaseAnomalyDetector (Strategy pattern contract)."""
        from src.pipeline.automata_pipeline import AutomataPipeline
        from src.models.base_model import BaseAnomalyDetector
        pipeline = AutomataPipeline(window_size=4, alphabet_size=3)
        self.assertIsInstance(pipeline, BaseAnomalyDetector)

    def test_pipeline_predict_before_fit_raises_runtime_error(self):
        """predict() called before fit() must raise RuntimeError, not a cryptic crash."""
        from src.pipeline.automata_pipeline import AutomataPipeline
        pipeline = AutomataPipeline(window_size=4, alphabet_size=3)
        X = np.random.rand(50, 1)
        with self.assertRaises(RuntimeError):
            pipeline.predict(X)

    def test_pipeline_get_metrics_returns_all_four_metric_keys(self):
        """get_metrics() must return accuracy, precision, recall, f1 (spec IX)."""
        from src.pipeline.automata_pipeline import AutomataPipeline
        rng = np.random.RandomState(42)
        X = rng.rand(300, 1)
        y = rng.randint(0, 2, 300)
        pipeline = AutomataPipeline(window_size=4, alphabet_size=3)
        pipeline.fit(X[:200], y[:200])
        metrics = pipeline.get_metrics(X[200:], y[200:])
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            self.assertIn(key, metrics, f"'{key}' missing from get_metrics() output")

    def test_pipeline_get_metrics_also_returns_timing_keys(self):
        """get_metrics() must return train_time_sec and inference_time_sec (Tablo 5)."""
        from src.pipeline.automata_pipeline import AutomataPipeline
        rng = np.random.RandomState(42)
        X = rng.rand(300, 1)
        y = rng.randint(0, 2, 300)
        pipeline = AutomataPipeline(window_size=4, alphabet_size=3)
        pipeline.fit(X[:200], y[:200])
        metrics = pipeline.get_metrics(X[200:], y[200:])
        self.assertIn('train_time_sec', metrics)
        self.assertIn('inference_time_sec', metrics)


class TestDLAnomalyDetector(unittest.TestCase):

    def test_dl_detector_is_instance_of_base_detector(self):
        """DLAnomalyDetector must implement BaseAnomalyDetector (Strategy pattern contract)."""
        from src.models.dl_models import DLAnomalyDetector, LSTMModel
        from src.models.base_model import BaseAnomalyDetector
        detector = DLAnomalyDetector(LSTMModel)
        self.assertIsInstance(detector, BaseAnomalyDetector)

    def test_dl_detector_get_metrics_before_fit_raises_runtime_error(self):
        """get_metrics() called before fit() must raise RuntimeError."""
        from src.models.dl_models import DLAnomalyDetector, LSTMModel
        detector = DLAnomalyDetector(LSTMModel)
        rng = np.random.RandomState(42)
        X = rng.rand(50, 1)
        y = rng.randint(0, 2, 50)
        with self.assertRaises(RuntimeError):
            detector.get_metrics(X, y)

    def test_dl_detector_fit_returns_self(self):
        """fit() must return self for method chaining."""
        from src.models.dl_models import DLAnomalyDetector, LSTMModel
        detector = DLAnomalyDetector(LSTMModel)
        rng = np.random.RandomState(42)
        X = rng.rand(100, 1)
        y = rng.randint(0, 2, 100)
        result = detector.fit(X, y)
        self.assertIs(result, detector)

    def test_dl_detector_predict_raises_not_implemented(self):
        """predict() must raise NotImplementedError — DL training is inseparable from eval."""
        from src.models.dl_models import DLAnomalyDetector, LSTMModel
        detector = DLAnomalyDetector(LSTMModel)
        rng = np.random.RandomState(42)
        X = rng.rand(100, 1)
        y = rng.randint(0, 2, 100)
        detector.fit(X, y)
        with self.assertRaises(NotImplementedError):
            detector.predict(X)


class TestModelRegistry(unittest.TestCase):

    def test_get_model_class_returns_lstm(self):
        """get_model_class('LSTM') must return LSTMModel class."""
        from src.models.model_registry import get_model_class
        from src.models.dl_models import LSTMModel
        self.assertIs(get_model_class('LSTM'), LSTMModel)

    def test_get_model_class_returns_gru(self):
        """get_model_class('GRU') must return GRUModel class."""
        from src.models.model_registry import get_model_class
        from src.models.dl_models import GRUModel
        self.assertIs(get_model_class('GRU'), GRUModel)

    def test_get_model_class_raises_for_unknown_name(self):
        """get_model_class('Transformer') must raise KeyError — unknown model."""
        from src.models.model_registry import get_model_class
        with self.assertRaises(KeyError):
            get_model_class('Transformer')

    def test_config_dl_models_all_in_registry(self):
        """Every model name listed in config models.dl must exist in the registry."""
        from src.utils.config_loader import load_config
        from src.models.model_registry import get_model_class
        cfg = load_config()
        for name in cfg['models']['dl']:
            try:
                get_model_class(name)
            except KeyError:
                self.fail(f"Model '{name}' is in config but missing from MODEL_REGISTRY")


if __name__ == '__main__':
    unittest.main()
