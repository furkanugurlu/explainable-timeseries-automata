"""
Standalone experiment: reconstruction-based anomaly detection on BATADAL.

Supervised LSTM/GRU/CNN classifiers cannot detect BATADAL's test attacks
(chronological distribution shift -> F1 ~ 0, even with class weighting).
An LSTM autoencoder trained ONLY on normal windows flags windows it cannot
reconstruct, which does not require having seen the specific attack pattern.

Output: results/batadal_autoencoder.csv
"""
import logging
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.data.data_loader_batadal import BATADALDataLoader
from src.models.dl_models import train_evaluate_autoencoder
from main import set_seed

logging.basicConfig(level=logging.WARNING)

cfg = load_config()
seeds = cfg['training']['seeds']

loader = BATADALDataLoader()
Xtr, Xval, Xte, ytr, yval, yte = loader.get_processed_splits(apply_pca=False)
print(f"BATADAL multivariate: train{Xtr.shape} val{Xval.shape} test{Xte.shape}")
print(f"anomaly rate -> train={ytr.mean():.3f}  val={yval.mean():.3f}  test={yte.mean():.3f}\n")

rows = []
for seed in seeds:
    set_seed(seed)
    m = train_evaluate_autoencoder(Xtr, ytr, Xval, yval, Xte, yte, config=cfg)
    rows.append({"model": "LSTM-AE", "seed": seed, "f1": m['f1'],
                 "precision": m['precision'], "recall": m['recall'],
                 "accuracy": m['accuracy'], "threshold": m['threshold']})
    print(f"seed {seed:4d}: F1={m['f1']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} acc={m['accuracy']:.4f}")

df = pd.DataFrame(rows)
df.to_csv("results/batadal_autoencoder.csv", index=False)

print("\n=== BATADAL LSTM-Autoencoder (5 seed ortalamasi) ===")
print(f"F1   = {df.f1.mean():.4f} +/- {df.f1.std():.4f}")
print(f"Prec = {df.precision.mean():.4f}   Rec = {df.recall.mean():.4f}   Acc = {df.accuracy.mean():.4f}")
print("\nSaved -> results/batadal_autoencoder.csv")
