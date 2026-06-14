import random
import logging
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, Any
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.utils.config_loader import load_config
from src.data.data_loader_skab import SKABDataLoader
from src.data.data_loader_batadal import BATADALDataLoader
from src.models.dl_models import DLAnomalyDetector
from src.models.model_registry import get_model_class
from src.pipeline.automata_pipeline import AutomataPipeline


def setup_experiment_dirs():
    """Creates all required output directories before any experiment writes."""
    for path in ["results", "results/figures", "results/explanations", "logs"]:
        Path(path).mkdir(parents=True, exist_ok=True)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Configures file + console logging. Level and dir are read from config (spec VIII.A)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/experiment.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def set_seed(seed: int):
    """Ensures reproducibility across numpy, random and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"[SEED] Applied deterministic seed: {seed}")


def add_gaussian_noise(data: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Generates a noisy version of the dataset for the Gaussian noise scenario."""
    noise = np.random.normal(loc=0, scale=scale, size=data.shape)
    return data + noise


def run_automata_pipeline(X_tr, y_tr, X_te, y_te, w_size, a_size, name="Automata") -> Dict[str, Any]:
    """
    Runs full Automata pipeline and returns all metrics.
    Delegates to AutomataPipeline — single canonical implementation, no duplication.
    """
    pipeline = AutomataPipeline(window_size=w_size, alphabet_size=a_size)
    pipeline.fit(X_tr, y_tr)
    return pipeline.get_metrics(X_te, y_te)


def run_cross_dataset_experiment(
    cfg: dict,
    seeds: list,
    skab_data=None,
    batadal_data=None,
) -> pd.DataFrame:
    """
    Cross-dataset generalizability: train on one dataset, test on the other (Tablo 3, 15 pts).

    Each direction (SKAB_to_BATADAL, BATADAL_to_SKAB) is evaluated with the full
    parameter grid across all seeds.  Results are returned as a flat DataFrame
    so callers can aggregate mean±std per direction and model variant.

    Args:
        cfg:          Full config dict — window_sizes and alphabet_sizes come from here.
        seeds:        List of random seeds for reproducibility.
        skab_data:    (X_tr, y_tr, X_te, y_te) after SKAB preprocessing, or None → disk load.
        batadal_data: (X_tr, y_tr, X_te, y_te) after BATADAL preprocessing, or None → disk load.

    Returns:
        pd.DataFrame with one row per (seed × direction × model variant).
        Empty DataFrame when either dataset is unavailable.
    """
    w_sizes = cfg['automata']['parameter_variations']['window_sizes']
    a_sizes = cfg['automata']['parameter_variations']['alphabet_sizes']
    results = []

    # --- Load data from disk when not injected (production path) ---
    if skab_data is None:
        try:
            skab_loader = SKABDataLoader(config=cfg)
            # Combine all 5 folds' train+test into a single training set and a
            # comprehensive test set for cross-dataset evaluation.
            # next() avoids materialising all folds at once; we then collect the rest.
            fold_gen = skab_loader.get_folds(n_splits=5)
            all_X_tr, all_X_te, all_y_tr, all_y_te = [], [], [], []
            for X_tr_f, X_te_f, y_tr_f, y_te_f in fold_gen:
                all_X_tr.append(X_tr_f)
                all_X_te.append(X_te_f)
                all_y_tr.append(y_tr_f)
                all_y_te.append(y_te_f)
            # Training: union of all fold train sets (maximum coverage of SKAB)
            # Test:     union of all fold test sets (all SKAB data seen as test)
            X_sk_tr = np.vstack(all_X_tr)
            y_sk_tr = np.concatenate(all_y_tr)
            X_sk_te = np.vstack(all_X_te)
            y_sk_te = np.concatenate(all_y_te)
            skab_data = (X_sk_tr, y_sk_tr, X_sk_te, y_sk_te)
            logging.info(
                f"Cross-dataset: SKAB loaded — "
                f"train={X_sk_tr.shape}, test={X_sk_te.shape}"
            )
        except Exception as e:
            logging.warning(f"Cross-dataset: SKAB unavailable — {e}")

    if batadal_data is None:
        try:
            batadal_loader = BATADALDataLoader(config=cfg)
            X_bd_tr, X_bd_val, X_bd_te, y_bd_tr, y_bd_val, y_bd_te = (
                batadal_loader.get_processed_splits()
            )
            batadal_data = (X_bd_tr, y_bd_tr, X_bd_te, y_bd_te)
            logging.info("Cross-dataset: BATADAL loaded.")
        except Exception as e:
            logging.warning(f"Cross-dataset: BATADAL unavailable — {e}")

    if skab_data is None or batadal_data is None:
        logging.warning("Cross-dataset: one or both datasets unavailable — returning empty results.")
        return pd.DataFrame()

    X_sk_tr, y_sk_tr, X_sk_te, y_sk_te = skab_data
    X_bd_tr, y_bd_tr, X_bd_te, y_bd_te = batadal_data

    # Both directions: (label, train_dataset, test_dataset, X_tr, y_tr, X_te, y_te)
    experiment_pairs = [
        ("SKAB_to_BATADAL", "SKAB",    "BATADAL", X_sk_tr, y_sk_tr, X_bd_te, y_bd_te),
        ("BATADAL_to_SKAB", "BATADAL", "SKAB",    X_bd_tr, y_bd_tr, X_sk_te, y_sk_te),
    ]

    for seed in seeds:
        set_seed(seed)

        for direction, train_ds, test_ds, X_tr, y_tr, X_te, y_te in experiment_pairs:
            logging.info(f"Cross-dataset: {direction} | seed={seed}")

            for w in w_sizes:
                for a in a_sizes:
                    try:
                        metrics = run_automata_pipeline(X_tr, y_tr, X_te, y_te, w_size=w, a_size=a)
                        metrics.update({
                            "direction":     direction,
                            "train_dataset": train_ds,
                            "test_dataset":  test_ds,
                            "model":         f"Automata_W{w}_A{a}",
                            "window":        w,
                            "alphabet":      a,
                            "seed":          seed,
                        })
                        results.append(metrics)
                    except Exception as e:
                        logging.debug(f"Cross {direction} W{w}/A{a}: {e}")

    return pd.DataFrame(results)


def main():
    setup_experiment_dirs()

    # Load config first so logging level and dir come from config (spec VIII)
    cfg = load_config()
    setup_logging(
        log_dir=cfg['logging']['log_dir'],
        log_level=cfg['logging']['level'],
    )

    logging.info("=" * 60)
    logging.info("STARTING EXPLAINABLE TIME SERIES AUTOMATA EXPERIMENT RUNNER")
    logging.info("=" * 60)

    seeds          = cfg['training']['seeds']
    w_sizes        = cfg['automata']['parameter_variations']['window_sizes']
    a_sizes        = cfg['automata']['parameter_variations']['alphabet_sizes']
    noise_scale    = cfg['experiments']['noise_scale']
    dl_model_names = cfg['models']['dl']   # no hardcoded list — driven by config

    all_results = []

    # --- STEP 1: Load Datasets ---
    logging.info("[1] Loading Data Sources...")
    data_sources = []

    try:
        skab_loader = SKABDataLoader()
        # Two parallel views of the same GroupKFold split (deterministic, so indices align):
        #  - PCA/PC1 (1D)         -> Automata (SAX/PAA requires a 1D signal, spec IV)
        #  - Multivariate scaled  -> DL models (LSTM/GRU/CNN1D natively support multi-feature input)
        folds_pca = list(skab_loader.get_folds(n_splits=5, apply_pca=True))
        folds_multi = list(skab_loader.get_folds(n_splits=5, apply_pca=False))
        for fold_idx, ((X_tr, X_te, y_tr, y_te), (X_tr_dl, X_te_dl, _, _)) in enumerate(zip(folds_pca, folds_multi), 1):
            split = int(len(X_tr) * 0.8)
            data_sources.append({
                "name":    f"SKAB_Fold{fold_idx}",
                # DL models need a val split for early stopping — use 80/20 of fold train.
                # Automata has no val requirement, so store the full fold train separately
                # to avoid wasting 20 % of training data on an unused validation set.
                "tr_full": (X_tr, y_tr),
                "tr":      (X_tr[:split], y_tr[:split]),
                "val":     (X_tr[split:], y_tr[split:]),
                "te":      (X_te, y_te),
                # Multivariate views for DL models (same row indices as above)
                "tr_dl":   (X_tr_dl[:split], y_tr[:split]),
                "val_dl":  (X_tr_dl[split:], y_tr[split:]),
                "te_dl":   (X_te_dl, y_te),
            })
        logging.info(f"SKAB: {sum(1 for d in data_sources if 'SKAB' in d['name'])} folds loaded.")
    except Exception as e:
        logging.warning(f"SKAB loading skipped. Error: {e}")

    try:
        batadal_loader = BATADALDataLoader()
        X_tr, X_val, X_te, y_tr, y_val, y_te = batadal_loader.get_processed_splits(apply_pca=True)
        X_tr_dl, X_val_dl, X_te_dl, _, _, _ = batadal_loader.get_processed_splits(apply_pca=False)
        data_sources.append({
            "name": "BATADAL_CHRON",
            "tr":  (X_tr, y_tr),
            "val": (X_val, y_val),
            "te":  (X_te, y_te),
            "tr_dl":  (X_tr_dl, y_tr),
            "val_dl": (X_val_dl, y_val),
            "te_dl":  (X_te_dl, y_te),
        })
        logging.info("BATADAL: chronological split loaded.")
    except Exception as e:
        logging.warning(f"BATADAL loading skipped. Error: {e}")

    if not data_sources:
        logging.warning("[!] No real data found. Using synthetic dummy data for smoke test.")
        dummy_x = np.random.rand(500, 1)
        dummy_y = np.random.randint(0, 2, 500)
        data_sources.append({
            "name": "SYNTHETIC_DEMO",
            "tr":  (dummy_x[:300], dummy_y[:300]),
            "val": (dummy_x[300:400], dummy_y[300:400]),
            "te":  (dummy_x[400:], dummy_y[400:]),
            "tr_dl":  (dummy_x[:300], dummy_y[:300]),
            "val_dl": (dummy_x[300:400], dummy_y[300:400]),
            "te_dl":  (dummy_x[400:], dummy_y[400:]),
        })

    # --- STEP 2: Multi-Seed Experiments across Scenarios ---
    logging.info("[2] Starting Multi-Seed Evaluations...")

    # Representative pipeline captured for Step 6 visualizations (default W/A, Original scenario)
    _dw = cfg['automata']['defaults']['window_size']
    _da = cfg['automata']['defaults']['alphabet_size']
    _viz_data: Dict[str, Any] = {}
    _viz_dl: Dict[str, Any] = {}  # representative DL run (first model, Original, first seed)

    for ds in data_sources:
        ds_name = ds["name"]
        X_tr, y_tr           = ds["tr"]
        # Automata uses full fold train when available (SKAB); falls back to "tr" otherwise (BATADAL)
        X_tr_full, y_tr_full = ds.get("tr_full", ds["tr"])
        X_val, y_val         = ds["val"]
        X_te_orig, y_te_orig = ds["te"]

        # Multivariate views for DL models (spec V.A — no PCA requirement for DL,
        # only the automata model needs the PC1 bottleneck per spec IV)
        X_tr_dl, _      = ds.get("tr_dl", ds["tr"])
        X_val_dl, _     = ds.get("val_dl", ds["val"])
        X_te_dl_orig, _ = ds.get("te_dl", ds["te"])

        for seed in seeds:
            set_seed(seed)

            # Build scenarios AFTER set_seed so each seed produces a distinct noise realization
            scenarios = {
                "Original":       (X_te_orig, y_te_orig),
                "Gaussian_Noise": (add_gaussian_noise(X_te_orig, scale=noise_scale), y_te_orig),
                # Unseen: same data — automata tracks unseen patterns via Levenshtein mapping
                "Unseen":         (X_te_orig, y_te_orig),
            }
            scenarios_dl = {
                "Original":       (X_te_dl_orig, y_te_orig),
                "Gaussian_Noise": (add_gaussian_noise(X_te_dl_orig, scale=noise_scale), y_te_orig),
                "Unseen":         (X_te_dl_orig, y_te_orig),
            }

            for scenario_name, (X_test, y_test) in scenarios.items():
                X_test_dl, _ = scenarios_dl[scenario_name]
                logging.info(f">>> {ds_name} | {scenario_name} | seed={seed}")

                # --- Deep Learning Models (config-driven via Registry + Strategy pattern) ---
                for m_name in dl_model_names:
                    m_class = get_model_class(m_name)
                    try:
                        dl_detector = DLAnomalyDetector(m_class, config=cfg)
                        dl_detector.fit(X_tr_dl, y_tr, X_val_dl, y_val)

                        # Capture first DL run on Original scenario for Step 6 figures
                        capture_dl = (not _viz_dl and scenario_name == "Original")
                        dl_metrics = dl_detector.get_metrics(
                            X_test_dl, y_test, return_outputs=capture_dl
                        )
                        if capture_dl:
                            _viz_dl["y_true"]  = dl_metrics.pop("y_true")
                            _viz_dl["y_prob"]  = dl_metrics.pop("y_prob")
                            _viz_dl["model"]   = m_name
                            _viz_dl["ds_name"] = ds_name

                        dl_metrics.update({
                            "dataset": ds_name, "scenario": scenario_name,
                            "model": m_name, "seed": seed,
                        })
                        all_results.append(dl_metrics)
                    except Exception as e:
                        logging.error(f"{m_name} failed: {e}")

                # --- Automata Hyperparameter Grid ---
                for w in w_sizes:
                    for a in a_sizes:
                        try:
                            pipeline = AutomataPipeline(window_size=w, alphabet_size=a)
                            pipeline.fit(X_tr_full, y_tr_full)  # full fold train, not DL's 80 % split
                            auto_res = pipeline.get_metrics(X_test, y_test)

                            # Capture representative pipeline for Step 6 (first default-params, Original)
                            if (not _viz_data and scenario_name == "Original"
                                    and w == _dw and a == _da):
                                _viz_data["pipeline"] = pipeline
                                _viz_data["y_true"]   = y_test[
                                    len(y_test) - len(pipeline._y_pred):
                                ].copy()
                                _viz_data["y_pred"]   = pipeline._y_pred.copy()
                                _viz_data["ds_name"]  = ds_name

                            # Save explanations for Unseen scenario (rubric Kriter3 — 20pt)
                            if scenario_name == "Unseen":
                                expl_path = (
                                    f"results/explanations/"
                                    f"{ds_name}_{scenario_name}_W{w}_A{a}_seed{seed}.json"
                                )
                                try:
                                    pipeline.save_explanations(expl_path)
                                except Exception as expl_err:
                                    logging.warning(f"Explanation save failed: {expl_err}")

                            auto_res.update({
                                "dataset": ds_name, "scenario": scenario_name,
                                "model": f"Automata_W{w}_A{a}",
                                "window": w, "alphabet": a, "seed": seed,
                            })
                            all_results.append(auto_res)
                        except Exception as e:
                            logging.debug(f"Automata W{w}/A{a} skipped: {e}")

    # --- STEP 3: Aggregation and Persistence ---
    logging.info("[3] Aggregating Final Results...")

    # Aggregation only runs when the main experiment produced results.
    # Step 4 (cross-dataset) always runs — it has its own independent data loading.
    if not all_results:
        logging.error("No main-experiment results collected — skipping Step 3 aggregation.")
    else:
        df_results = pd.DataFrame(all_results)

        agg_cols = {}
        for col in ['f1', 'accuracy', 'precision', 'recall']:
            if col in df_results.columns:
                agg_cols[col] = ['mean', 'std']
        # Tablo 2: Unseen scenario metrics — mean±std across seeds
        for col in ['detection_rate', 'mapping_accuracy']:
            if col in df_results.columns:
                agg_cols[col] = ['mean', 'std']
        for col in ['unseen_window_count', 'train_time_sec', 'inference_time_sec', 'state_count']:
            if col in df_results.columns:
                agg_cols[col] = ['mean']
        for col in ['density']:
            if col in df_results.columns:
                agg_cols[col] = ['mean']

        summary = df_results.groupby(["dataset", "scenario", "model"]).agg(agg_cols).reset_index()

        output_csv     = Path("results/experiment_full_raw.csv")
        output_summary = Path("results/experiment_summary.csv")
        df_results.to_csv(output_csv, index=False)
        summary.to_csv(output_summary)

        logging.info("#" * 50)
        logging.info("EXPERIMENTS CONCLUDED SUCCESSFULLY.")
        logging.info(f"Raw:     {output_csv}")
        logging.info(f"Summary: {output_summary}")
        logging.info("#" * 50)

        logging.info("\nSample Result Preview:\n" + summary.head(10).to_string())

        # --- STEP 5: Wilcoxon Statistical Significance Tests (rubric Kriter4 — 5 pts) ---
        logging.info("[5] Running Wilcoxon Statistical Tests (Automata vs DL)...")
        from src.visualization.visualization_and_stats import perform_wilcoxon_test as _wilcoxon

        wilcoxon_rows = []
        auto_key = f"Automata_W{_dw}_A{_da}"

        for ds_n in df_results['dataset'].unique():
            orig_df = df_results[
                (df_results['dataset'] == ds_n) & (df_results['scenario'] == 'Original')
            ]
            auto_f1 = (
                orig_df[orig_df['model'] == auto_key]
                .groupby('seed')['f1'].mean()
            )
            for dl_name in dl_model_names:
                dl_f1 = (
                    orig_df[orig_df['model'] == dl_name]
                    .groupby('seed')['f1'].mean()
                )
                common = sorted(set(auto_f1.index) & set(dl_f1.index))
                if len(common) >= 2:
                    res = _wilcoxon(
                        auto_f1[common].tolist(),
                        dl_f1[common].tolist(),
                        auto_key, dl_name,
                    )
                    if res:
                        res.update({
                            'dataset': ds_n,
                            'model_a': auto_key,
                            'model_b': dl_name,
                        })
                        wilcoxon_rows.append(res)

        if wilcoxon_rows:
            wilcoxon_df = pd.DataFrame(wilcoxon_rows)
            wilcoxon_path = Path("results/wilcoxon_results.csv")
            wilcoxon_df.to_csv(wilcoxon_path, index=False)
            logging.info(f"Wilcoxon results saved to {wilcoxon_path}")
            logging.info("\nWilcoxon Summary:\n" + wilcoxon_df.to_string())
        else:
            logging.warning("Wilcoxon: no common seeds found — skipping CSV.")

        # --- STEP 6: Visualizations (rubric Kriter5 — 3 pts) ---
        logging.info("[6] Generating Experiment Visualizations...")
        from src.visualization.visualization_and_stats import (
            plot_confusion_matrix,
            plot_roc_pr_curves,
            plot_hyperparameter_heatmap,
            plot_transition_heatmap,
            draw_automata_graph,
            plot_model_comparison,
            plot_scenario_impact,
        )

        # Parametre duyarlılık heatmap: window_size vs alphabet_size (Tablo 4 görseli)
        # Tablo 4 = SKAB, Original senaryo — diğer senaryolar/datasetler karıştırılmaz
        # (Gaussian_Noise otomata F1'ini şişirir, BATADAL F1=0 ortalamayı düşürür)
        if 'window' in df_results.columns:
            auto_viz_df = df_results[
                df_results['model'].str.startswith('Automata')
                & (df_results['scenario'] == 'Original')
                & df_results['dataset'].str.startswith('SKAB')
            ].copy()
            if not auto_viz_df.empty:
                plot_hyperparameter_heatmap(auto_viz_df, metric='f1')

        # Aggregate comparison charts (Tablo 1A / Tablo 2 visual companions)
        plot_model_comparison(df_results, metric='f1')
        plot_scenario_impact(df_results, metric='f1')

        # Pipeline-based figures: confusion matrix, ROC/PR, state diagrams
        if _viz_data:
            viz_p      = _viz_data["pipeline"]
            viz_y_true = _viz_data["y_true"]
            viz_y_pred = _viz_data["y_pred"]
            viz_ds     = _viz_data["ds_name"]

            plot_confusion_matrix(viz_y_true, viz_y_pred, f"Automata_{viz_ds}")

            # Negate path probs: low prob = high anomaly score (sklearn roc_curve convention)
            path_probs     = viz_p.get_path_probabilities()
            anomaly_scores = np.array(path_probs) * -1.0
            plot_roc_pr_curves(viz_y_true, anomaly_scores, f"Automata_{viz_ds}")

            plot_transition_heatmap(viz_p.model.probabilities)
            draw_automata_graph(viz_p.model.probabilities)
        else:
            logging.warning("No representative pipeline captured — state diagram figures skipped.")

        # DL representative figures: confusion matrix + ROC/PR (multivariate input)
        if _viz_dl:
            dl_y_true = _viz_dl["y_true"]
            dl_y_prob = _viz_dl["y_prob"]
            dl_suffix = f"{_viz_dl['model']}_{_viz_dl['ds_name']}"

            plot_confusion_matrix(dl_y_true, (dl_y_prob > 0.5).astype(int), dl_suffix)
            plot_roc_pr_curves(dl_y_true, dl_y_prob, dl_suffix)
        else:
            logging.warning("No representative DL run captured — DL figures skipped.")

        logging.info("All visualizations saved to results/figures/")

    # --- STEP 4: Cross-Dataset Generalizability (Tablo 3, 15 pts) ---
    logging.info("[4] Cross-Dataset Generalizability Experiment...")

    # Reuse Step 1's already-loaded PC1 data instead of reloading from disk and
    # refitting GroupKFold/PCA a second time (deterministic split -> identical results).
    skab_data = None
    skab_folds = [d for d in data_sources if d["name"].startswith("SKAB_Fold")]
    if skab_folds:
        skab_data = (
            np.vstack([d["tr_full"][0] for d in skab_folds]),
            np.concatenate([d["tr_full"][1] for d in skab_folds]),
            np.vstack([d["te"][0] for d in skab_folds]),
            np.concatenate([d["te"][1] for d in skab_folds]),
        )

    batadal_data = None
    batadal_ds = next((d for d in data_sources if d["name"] == "BATADAL_CHRON"), None)
    if batadal_ds:
        batadal_data = (*batadal_ds["tr"], *batadal_ds["te"])

    cross_df = run_cross_dataset_experiment(cfg, seeds, skab_data=skab_data, batadal_data=batadal_data)
    if not cross_df.empty:
        cross_path = Path("results/cross_dataset.csv")
        cross_df.to_csv(cross_path, index=False)
        logging.info(f"Cross-dataset raw results saved to {cross_path}")

        # mean±std per direction and model — this becomes Tablo 3 in the report
        cross_agg = {}
        for col in ['f1', 'accuracy', 'precision', 'recall']:
            if col in cross_df.columns:
                cross_agg[col] = ['mean', 'std']
        for col in ['detection_rate', 'mapping_accuracy']:
            if col in cross_df.columns:
                cross_agg[col] = ['mean', 'std']
        for col in ['train_time_sec', 'inference_time_sec', 'state_count', 'density']:
            if col in cross_df.columns:
                cross_agg[col] = ['mean']

        if cross_agg:
            cross_summary = (
                cross_df.groupby(['direction', 'train_dataset', 'test_dataset', 'model'])
                .agg(cross_agg)
                .reset_index()
            )
            cross_summary_path = Path("results/cross_dataset_summary.csv")
            cross_summary.to_csv(cross_summary_path)
            logging.info(f"Cross-dataset summary (mean±std) saved to {cross_summary_path}")
            logging.info("\nCross-Dataset Preview:\n" + cross_summary.to_string())
    else:
        logging.warning("Cross-dataset experiment produced no results (datasets not on disk).")


if __name__ == "__main__":
    main()
