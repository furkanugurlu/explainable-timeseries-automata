"""
Standalone experiment: BATADAL DL with class-imbalance handling.

BATADAL's training set is only ~4% anomaly, so plain BCELoss + 0.5 threshold
collapses every DL model into an all-normal predictor (F1=0). This script
re-runs ONLY BATADAL with (a) positive-class weighting in the loss and
(b) validation-based threshold tuning, and compares against the plain baseline.
SKAB results and the main pipeline are untouched.

Output: results/batadal_imbalance.csv
"""
import logging
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.data.data_loader_batadal import BATADALDataLoader
from src.models.dl_models import train_evaluate_dl
from src.models.model_registry import get_model_class
from main import set_seed

logging.basicConfig(level=logging.WARNING)

cfg = load_config()
dl_models = cfg['models']['dl']
seeds = cfg['training']['seeds']

loader = BATADALDataLoader()
Xtr, Xval, Xte, ytr, yval, yte = loader.get_processed_splits(apply_pca=False)
print(f"BATADAL multivariate: train{Xtr.shape} val{Xval.shape} test{Xte.shape}")
print(f"anomaly rate -> train={ytr.mean():.3f}  val={yval.mean():.3f}  test={yte.mean():.3f}\n")

variants = [
    ("baseline", dict()),  # plain BCELoss + 0.5 threshold (reproduces F1=0)
    ("balanced", dict(use_class_weight=True, tune_threshold=True)),
]

rows = []
for m_name in dl_models:
    mclass = get_model_class(m_name)
    for label, kw in variants:
        for seed in seeds:
            set_seed(seed)
            met = train_evaluate_dl(mclass, Xtr, ytr, Xval, yval, Xte, yte, config=cfg, **kw)
            rows.append({
                "model": m_name, "variant": label, "seed": seed,
                "f1": met['f1'], "precision": met['precision'],
                "recall": met['recall'], "accuracy": met['accuracy'],
                "threshold": met['threshold'], "pos_weight": met['pos_weight'],
            })
        sub = [r for r in rows if r['model'] == m_name and r['variant'] == label]
        f1s = [r['f1'] for r in sub]
        print(f"{m_name:6s} {label:9s} F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

df = pd.DataFrame(rows)
df.to_csv("results/batadal_imbalance.csv", index=False)

print("\n=== BATADAL: baseline vs balanced (5 seed ortalaması) ===")
g = df.groupby(['model', 'variant']).agg(
    f1=('f1', 'mean'), f1std=('f1', 'std'),
    precision=('precision', 'mean'), recall=('recall', 'mean'),
    accuracy=('accuracy', 'mean'), thr=('threshold', 'mean'),
    pos_weight=('pos_weight', 'mean')).round(4)
print(g.to_string())
print("\nSaved -> results/batadal_imbalance.csv")
