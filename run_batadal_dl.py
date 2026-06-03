"""
Runs only the missing BATADAL DL experiments (LSTM/GRU/CNN1D).
Appends results to experiment_full_raw.csv and regenerates experiment_summary.csv.
Fix for: dl_models.py squeeze() → squeeze(1) for single-sample last batch.
"""
import random
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.utils.config_loader import load_config
from src.data.data_loader_batadal import BATADALDataLoader
from src.models.dl_models import DLAnomalyDetector
from src.models.model_registry import get_model_class


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_gaussian_noise(data: np.ndarray, scale: float = 0.1) -> np.ndarray:
    noise = np.random.normal(loc=0, scale=scale, size=data.shape)
    return data + noise


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/batadal_dl.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    cfg = load_config()
    seeds          = cfg['training']['seeds']
    noise_scale    = cfg['experiments']['noise_scale']
    dl_model_names = cfg['models']['dl']

    logging.info("Loading BATADAL splits...")
    X_tr, X_val, X_te, y_tr, y_val, y_te = BATADALDataLoader(config=cfg).get_processed_splits()
    logging.info(f"BATADAL — Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    all_results = []

    for seed in seeds:
        set_seed(seed)

        scenarios = {
            "Original":       X_te,
            "Gaussian_Noise": add_gaussian_noise(X_te, scale=noise_scale),
            "Unseen":         X_te,
        }

        for scenario_name, X_test in scenarios.items():
            logging.info(f">>> BATADAL_CHRON | {scenario_name} | seed={seed}")

            for m_name in dl_model_names:
                m_class = get_model_class(m_name)
                try:
                    detector = DLAnomalyDetector(m_class, config=cfg)
                    detector.fit(X_tr, y_tr, X_val, y_val)
                    metrics = detector.get_metrics(X_test, y_te)
                    metrics.update({
                        "dataset":  "BATADAL_CHRON",
                        "scenario": scenario_name,
                        "model":    m_name,
                        "seed":     seed,
                    })
                    all_results.append(metrics)
                    logging.info(f"  {m_name} F1={metrics['f1']:.4f}")
                except Exception as e:
                    logging.error(f"{m_name} failed: {e}")

    if not all_results:
        logging.error("No results collected.")
        return

    new_df = pd.DataFrame(all_results)
    raw_path = Path("results/experiment_full_raw.csv")

    # Append to existing raw CSV
    if raw_path.exists():
        existing = pd.read_csv(raw_path)
        # Remove any stale BATADAL DL rows (in case of partial previous run)
        mask = ~(
            (existing['dataset'] == 'BATADAL_CHRON') &
            (existing['model'].isin(dl_model_names))
        )
        existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(raw_path, index=False)
    logging.info(f"Updated {raw_path} ({len(combined)} total rows)")

    # Regenerate summary CSV
    agg_cols = {}
    for col in ['f1', 'accuracy', 'precision', 'recall']:
        if col in combined.columns:
            agg_cols[col] = ['mean', 'std']
    for col in ['detection_rate', 'mapping_accuracy']:
        if col in combined.columns:
            agg_cols[col] = ['mean', 'std']
    for col in ['unseen_window_count', 'train_time_sec', 'inference_time_sec', 'state_count', 'density']:
        if col in combined.columns:
            agg_cols[col] = ['mean']

    summary = combined.groupby(["dataset", "scenario", "model"]).agg(agg_cols).reset_index()
    summary_path = Path("results/experiment_summary.csv")
    summary.to_csv(summary_path)
    logging.info(f"Regenerated {summary_path} ({len(summary)} rows)")

    # Quick check
    bat_dl = summary[
        (summary['dataset'] == 'BATADAL_CHRON') &
        summary['model'].isin(dl_model_names)
    ]
    logging.info(f"\nBATADAL DL results:\n{bat_dl[['scenario','model',('f1','mean'),('f1','std')]].to_string()}")


if __name__ == "__main__":
    main()
