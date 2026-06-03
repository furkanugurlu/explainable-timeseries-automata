"""
Regenerates all explanation JSON files without re-running DL training.
Run after fixing explainability.py (Anomaly -> anomaly case sensitivity bug).
"""
import random
import logging
import numpy as np
import torch
from pathlib import Path

from src.utils.config_loader import load_config
from src.data.data_loader_skab import SKABDataLoader
from src.data.data_loader_batadal import BATADALDataLoader
from src.pipeline.automata_pipeline import AutomataPipeline


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    cfg = load_config()
    seeds   = cfg['training']['seeds']
    w_sizes = cfg['automata']['parameter_variations']['window_sizes']
    a_sizes = cfg['automata']['parameter_variations']['alphabet_sizes']

    Path("results/explanations").mkdir(parents=True, exist_ok=True)
    total = 0

    # ---- SKAB ----
    logging.info("Loading SKAB folds...")
    folds = list(SKABDataLoader(config=cfg).get_folds(n_splits=5))

    for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(folds, 1):
        ds_name = f"SKAB_Fold{fold_idx}"
        for seed in seeds:
            set_seed(seed)
            for w in w_sizes:
                for a in a_sizes:
                    path = f"results/explanations/{ds_name}_Unseen_W{w}_A{a}_seed{seed}.json"
                    try:
                        pipeline = AutomataPipeline(window_size=w, alphabet_size=a)
                        pipeline.fit(X_tr, y_tr)
                        pipeline.predict(X_te)
                        pipeline.save_explanations(path)
                        total += 1
                    except Exception as e:
                        logging.warning(f"Failed {path}: {e}")

    # ---- BATADAL ----
    logging.info("Loading BATADAL splits...")
    try:
        X_tr, X_val, X_te, y_tr, y_val, y_te = BATADALDataLoader(config=cfg).get_processed_splits()
        ds_name = "BATADAL_CHRON"
        for seed in seeds:
            set_seed(seed)
            for w in w_sizes:
                for a in a_sizes:
                    path = f"results/explanations/{ds_name}_Unseen_W{w}_A{a}_seed{seed}.json"
                    try:
                        pipeline = AutomataPipeline(window_size=w, alphabet_size=a)
                        pipeline.fit(X_tr, y_tr)
                        pipeline.predict(X_te)
                        pipeline.save_explanations(path)
                        total += 1
                    except Exception as e:
                        logging.warning(f"Failed {path}: {e}")
    except Exception as e:
        logging.warning(f"BATADAL skipped: {e}")

    logging.info(f"Done — {total} explanation files written.")


if __name__ == "__main__":
    main()
