import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))


def _fast_cfg():
    """Returns config with minimal grid so cross-dataset tests complete quickly."""
    from src.utils.config_loader import load_config
    cfg = load_config()
    cfg['automata']['parameter_variations'] = {
        'window_sizes': [4],
        'alphabet_sizes': [3],
    }
    return cfg


def _synthetic_pair(n=300, seed=0):
    """Returns (X_tr, y_tr, X_te, y_te) with shape (N,1) each."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 1)
    y = rng.randint(0, 2, n)
    split = n * 2 // 3
    return X[:split], y[:split], X[split:], y[split:]


class TestCrossDatasetFunction(unittest.TestCase):
    """run_cross_dataset_experiment must exist, return a DataFrame, and populate Tablo 3."""

    def test_function_is_importable(self):
        """run_cross_dataset_experiment must be importable from main."""
        from main import run_cross_dataset_experiment
        self.assertTrue(callable(run_cross_dataset_experiment))

    def test_returns_dataframe(self):
        """Function must return a pandas DataFrame."""
        from main import run_cross_dataset_experiment
        skab = _synthetic_pair(n=300, seed=1)
        batadal = _synthetic_pair(n=300, seed=2)
        result = run_cross_dataset_experiment(
            _fast_cfg(), seeds=[42],
            skab_data=skab, batadal_data=batadal
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_df_has_required_columns(self):
        """Result DataFrame must contain direction, train_dataset, test_dataset, f1, seed."""
        from main import run_cross_dataset_experiment
        skab = _synthetic_pair(n=300, seed=1)
        batadal = _synthetic_pair(n=300, seed=2)
        df = run_cross_dataset_experiment(
            _fast_cfg(), seeds=[42],
            skab_data=skab, batadal_data=batadal
        )
        for col in ['direction', 'train_dataset', 'test_dataset', 'f1', 'seed']:
            self.assertIn(col, df.columns, f"Missing required column: '{col}'")

    def test_df_contains_both_directions(self):
        """DataFrame must include both SKAB_to_BATADAL and BATADAL_to_SKAB directions."""
        from main import run_cross_dataset_experiment
        skab = _synthetic_pair(n=300, seed=1)
        batadal = _synthetic_pair(n=300, seed=2)
        df = run_cross_dataset_experiment(
            _fast_cfg(), seeds=[42],
            skab_data=skab, batadal_data=batadal
        )
        directions = set(df['direction'].unique())
        self.assertIn('SKAB_to_BATADAL', directions,
            "SKAB_to_BATADAL direction missing from cross-dataset results")
        self.assertIn('BATADAL_to_SKAB', directions,
            "BATADAL_to_SKAB direction missing from cross-dataset results")

    def test_f1_values_in_valid_range(self):
        """Every f1 value in the result DataFrame must be in [0.0, 1.0]."""
        from main import run_cross_dataset_experiment
        skab = _synthetic_pair(n=300, seed=1)
        batadal = _synthetic_pair(n=300, seed=2)
        df = run_cross_dataset_experiment(
            _fast_cfg(), seeds=[42],
            skab_data=skab, batadal_data=batadal
        )
        for val in df['f1']:
            self.assertGreaterEqual(val, 0.0, "f1 must be >= 0")
            self.assertLessEqual(val, 1.0, "f1 must be <= 1")

    def test_multi_seed_produces_multiple_rows_per_direction(self):
        """Running with 2 seeds must produce at least 2 rows per direction."""
        from main import run_cross_dataset_experiment
        skab = _synthetic_pair(n=300, seed=1)
        batadal = _synthetic_pair(n=300, seed=2)
        df = run_cross_dataset_experiment(
            _fast_cfg(), seeds=[42, 123],
            skab_data=skab, batadal_data=batadal
        )
        for direction in ['SKAB_to_BATADAL', 'BATADAL_to_SKAB']:
            count = len(df[df['direction'] == direction])
            self.assertGreaterEqual(count, 2,
                f"Direction '{direction}' must have >=2 rows for 2 seeds")

    def test_returns_empty_df_when_both_datasets_none(self):
        """When no data is provided (disk unavailable), must return empty DataFrame gracefully."""
        from main import run_cross_dataset_experiment
        # Pass a config that points to non-existent paths so disk loading fails
        from src.utils.config_loader import load_config
        cfg = load_config()
        cfg['data']['datasets']['skab']['path'] = 'data/raw/nonexistent_skab'
        cfg['data']['datasets']['batadal']['path'] = 'data/raw/nonexistent_batadal'
        cfg['automata']['parameter_variations'] = {'window_sizes': [4], 'alphabet_sizes': [3]}
        df = run_cross_dataset_experiment(cfg, seeds=[42])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0, "Empty DataFrame expected when datasets are unavailable")


if __name__ == '__main__':
    unittest.main()
