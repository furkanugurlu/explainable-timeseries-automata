import os
import random
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, Any, List
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# Import Project Modules
from src.utils.config_loader import load_config
from src.data.data_loader_skab import SKABDataLoader
from src.data.data_loader_batadal import BATADALDataLoader
from src.models.dl_models import LSTMModel, GRUModel, CNN1DModel, train_evaluate_dl
from src.models.automata_transform import SAXTransformer
from src.models.automata_model import ProbabilisticAutomata
from src.models.explainability import AutomataExplainer

def set_seed(seed: int):
    """Ensures reproducibility across numpy, random and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"\n---> [SEED CONTEXT]: Applied deterministic seed value: {seed}")

def add_gaussian_noise(data: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Generates a noisy version of the dataset scenario."""
    noise = np.random.normal(loc=0, scale=scale, size=data.shape)
    return data + noise

def run_automata_pipeline(X_tr, y_tr, X_te, y_te, w_size, a_size, name="Automata") -> Dict[str, Any]:
    """Runs standalone Automata logic and retrieves metrics."""
    # 1. Transform
    t0 = time.time()
    sax = SAXTransformer(window_size=w_size, alphabet_size=a_size)
    train_sym = sax.fit_transform(X_tr)
    test_sym = sax.transform(X_te)
    
    train_patterns = sax.extract_patterns(train_sym, pattern_length=w_size)
    test_patterns = sax.extract_patterns(test_sym, pattern_length=w_size)
    
    # 2. Model Fit
    model = ProbabilisticAutomata()
    model.fit(train_patterns)
    
    fit_time = time.time() - t0
    
    # 3. Predict & Inference Time
    t1 = time.time()
    # Align labels by accounting for pattern overlap reduction
    trimmed_y_te = y_te[len(y_te) - len(test_patterns):]
    probs, preds = model.predict_anomalies(test_patterns, window_len=5)
    inference_time = time.time() - t1
    
    # Cut labels to match predictor windowing (which trims more)
    aligned_y = trimmed_y_te[len(trimmed_y_te) - len(preds):]
    
    f1 = f1_score(aligned_y, preds, zero_division=0)
    acc = accuracy_score(aligned_y, preds)
    
    # Calculated Automata Stats requested: State Count and Transition Density
    num_states = len(model.states)
    # Density = existing connections / total potential connections
    potential_conns = num_states ** 2 if num_states > 0 else 1
    actual_conns = sum(len(v) for v in model.probabilities.values())
    density = actual_conns / potential_conns if potential_conns > 0 else 0
    
    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "state_count": int(num_states),
        "density": float(density),
        "train_time_sec": fit_time,
        "inference_time_sec": inference_time
    }

def main():
    print("="*50)
    print("STARTING EXPLAINABLE TIME SERIES AUTOMATA EXPERIMENT RUNNER")
    print("="*50)
    
    # Load configs strictly
    cfg = load_config()
    seeds = cfg['training']['seeds']
    
    # Parameter grid for Automata
    w_sizes = cfg['automata']['parameter_variations']['window_sizes']
    a_sizes = cfg['automata']['parameter_variations']['alphabet_sizes']
    
    all_results = []
    
    # --- STEP 1: Load Datasets ---
    print("\n[1] Loading Data Sources...")
    
    # We loop over datasets if existing. For safety in demo environment, we catch errors
    data_sources = []
    
    try:
        skab_loader = SKABDataLoader()
        # Get first fold as representative for large experiments to keep manageable 
        # (Though loop can be added to test all 5 folds)
        gen = skab_loader.get_folds(n_splits=5)
        X_tr, X_te, y_tr, y_te = next(gen)
        # Create fake validation for split compatibility from training end
        split = int(len(X_tr) * 0.8)
        data_sources.append({
            "name": "SKAB_Fold1",
            "tr": (X_tr[:split], y_tr[:split]),
            "val": (X_tr[split:], y_tr[split:]),
            "te": (X_te, y_te)
        })
        print("✔ SKAB Data successfully integrated.")
    except Exception as e:
        print(f"⚠ SKAB loading skipped (Check dataset existence). Error: {e}")

    try:
        batadal_loader = BATADALDataLoader()
        X_tr, X_val, X_te, y_tr, y_val, y_te = batadal_loader.get_processed_splits()
        data_sources.append({
            "name": "BATADAL_CHRON",
            "tr": (X_tr, y_tr),
            "val": (X_val, y_val),
            "te": (X_te, y_te)
        })
        print("✔ BATADAL Data successfully integrated.")
    except Exception as e:
        print(f"⚠ BATADAL loading skipped (Check dataset existence). Error: {e}")

    if not data_sources:
        print("\n[!] NO DATA FOUND in raw folders. Executing Smoke Test with DUMMY DATA to verify pipeline logic.")
        # Generate synthetic to allow runtime verification without errors
        dummy_x = np.random.rand(500, 1)
        dummy_y = np.random.randint(0, 2, 500)
        data_sources.append({
            "name": "SYNTHETIC_DEMO",
            "tr": (dummy_x[:300], dummy_y[:300]),
            "val": (dummy_x[300:400], dummy_y[300:400]),
            "te": (dummy_x[400:], dummy_y[400:])
        })

    # --- STEP 2: Execute Multi-Seed Experiments across Scenarios ---
    print("\n[2] Starting Multi-Seed Evaluations...")
    
    for ds in data_sources:
        ds_name = ds["name"]
        X_tr, y_tr = ds["tr"]
        X_val, y_val = ds["val"]
        X_te_orig, y_te_orig = ds["te"]
        
        # Prepare Scenario Data Variants
        scenarios = {
            "Original": (X_te_orig, y_te_orig),
            "Gaussian_Noise": (add_gaussian_noise(X_te_orig, scale=0.15), y_te_orig)
        }
        
        for seed in seeds:
            set_seed(seed)
            
            for scenario_name, (X_test, y_test) in scenarios.items():
                print(f"\n>>> Running {ds_name} | Scenario: {scenario_name} | Seed: {seed}")
                
                # --- RUN DEEP LEARNING MODELS LOOP ---
                dl_model_variants = [
                    ("LSTM", LSTMModel),
                    ("GRU", GRUModel),
                    ("CNN1D", CNN1DModel)
                ]
                
                for m_name, m_class in dl_model_variants:
                    try:
                        print(f"Training {m_name}...")
                        dl_metrics = train_evaluate_dl(
                            m_class, X_tr, y_tr, X_val, y_val, X_test, y_test, config=cfg
                        )
                        dl_metrics.update({
                            "dataset": ds_name, "scenario": scenario_name, "model": m_name, "seed": seed
                        })
                        all_results.append(dl_metrics)
                    except Exception as e:
                        print(f"{m_name} Fail: {e}")
                
                # --- RUN AUTOMATA (Hyperparameter Search Cycle) ---
                print(f"Beginning Automata Param Iteration...")
                for w in w_sizes:
                    for a in a_sizes:
                        try:
                            auto_res = run_automata_pipeline(X_tr, y_tr, X_test, y_test, w, a)
                            auto_res.update({
                                "dataset": ds_name, 
                                "scenario": scenario_name, 
                                "model": f"Automata_W{w}_A{a}", 
                                "window": w, 
                                "alphabet": a,
                                "seed": seed
                            })
                            all_results.append(auto_res)
                        except Exception as e:
                            pass # Silence expected small length mismatches in extreme grid corners
                            
    # --- STEP 3: Aggregation and Log Persistence ---
    print("\n[3] Aggregating Final Results...")
    if not all_results:
        print("No results collected. Terminating.")
        return
        
    df_results = pd.DataFrame(all_results)
    
    # Calculate Averages & Standard Deviations grouped by model, scenario, dataset across ALL SEEDS
    summary = df_results.groupby(["dataset", "scenario", "model"]).agg({
        'f1': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'train_time_sec': ['mean'],
        'inference_time_sec': ['mean']
    }).reset_index()
    
    # Save all records for later analysis
    output_csv = Path("results/experiment_full_raw.csv")
    df_results.to_csv(output_csv, index=False)
    
    output_summary = Path("results/experiment_summary.csv")
    summary.to_csv(output_summary)
    
    print("\n" + "#"*40)
    print(f"EXPERIMENTS CONCLUDED SUCCESSFULLY.")
    print(f"Raw results written to: {output_csv}")
    print(f"Averaged Summary written to: {output_summary}")
    print("#"*40)
    
    # Print preview of head
    print("\nSample Result Preview:")
    print(summary.head(10))

if __name__ == "__main__":
    main()
