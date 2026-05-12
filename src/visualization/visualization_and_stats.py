import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from pathlib import Path
from typing import Dict, List, Any

# Ensure storage exists
SAVE_DIR = Path("results/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def perform_wilcoxon_test(group_a: List[float], group_b: List[float], label_a: str, label_b: str):
    """
    Computes Wilcoxon signed-rank test to determine statistical difference significance.
    Used when comparing pairs of performance metrics (e.g., across various seeds).
    """
    print(f"\n=== Statistical Significance Test: {label_a} vs {label_b} ===")
    
    if len(group_a) != len(group_b):
        print("Error: Group sizes must be identical for paired test.")
        return
        
    if len(group_a) < 2:
        print("Notice: Not enough samples for reliable p-value.")
        return
        
    # If perfect match, will throw error, so handle identical inputs
    if np.array_equal(group_a, group_b):
        print("P-value: 1.0 (Identical datasets)")
        return

    try:
        stat, p = wilcoxon(group_a, group_b)
        print(f"Statistic: {stat:.4f}")
        print(f"P-value:   {p:.6f}")
        if p < 0.05:
            print("Result: STATISTICALLY SIGNIFICANT difference (p < 0.05)")
        else:
            print("Result: NOT statistically significant (p >= 0.05)")
    except Exception as e:
        print(f"Test could not compute: {e}")

def plot_confusion_matrix(y_true, y_pred, title_suffix=""):
    """Plots a labeled confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title(f"Confusion Matrix {title_suffix}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    filename = SAVE_DIR / f"conf_matrix_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to {filename}")

def plot_roc_pr_curves(y_true, y_prob, title_suffix=""):
    """Plots both ROC and Precision-Recall side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='darkorange', lw=2)
    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, label=f'AUC = {pr_auc:.3f}', color='green', lw=2)
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    plt.suptitle(f"Performance Curves: {title_suffix}")
    plt.tight_layout()
    
    filename = SAVE_DIR / f"curves_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

def plot_hyperparameter_heatmap(results_df: pd.DataFrame, metric="f1"):
    """
    Takes comprehensive experiments DataFrame and graphs 
    window_size vs alphabet_size performance as a heatmap.
    """
    if 'window' not in results_df.columns or 'alphabet' not in results_df.columns:
        print("Required dataframe columns ('window', 'alphabet') missing for hyperparam plot.")
        return
        
    # Pivot to grid format suitable for seaborn
    pivot = results_df.pivot_table(index='window', columns='alphabet', values=metric, aggfunc='mean')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f"Automata Performance Grid (Mean {metric.upper()})")
    plt.xlabel("Alphabet Size")
    plt.ylabel("Window Size")
    
    filename = SAVE_DIR / f"param_heatmap_{metric}.png"
    plt.savefig(filename)
    plt.close()

def plot_transition_heatmap(model_probabilities: Dict[str, Dict[str, float]]):
    """
    Converts nested dict to a visual grid of state transition probabilities.
    """
    # Extract sorted union of states to ensure matrix alignment
    states = sorted(list(set(model_probabilities.keys()) | {to_s for to_vals in model_probabilities.values() for to_s in to_vals}))
    
    if len(states) > 30:
        print("Notice: Automata size is extremely large, heatmap might be unreadable. Clipping to top 20 states.")
        states = states[:20]
        
    # Build matrix
    size = len(states)
    matrix = np.zeros((size, size))
    
    state_to_idx = {s: i for i, s in enumerate(states)}
    for from_s, links in model_probabilities.items():
        if from_s not in state_to_idx: continue
        for to_s, prob in links.items():
            if to_s not in state_to_idx: continue
            matrix[state_to_idx[from_s], state_to_idx[to_s]] = prob
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=states, yticklabels=states, cmap="Reds", annot=len(states)<15)
    plt.title("Transition Probability Density Map")
    plt.xlabel("Target State (To)")
    plt.ylabel("Source State (From)")
    plt.tight_layout()
    
    filename = SAVE_DIR / "transition_density_heatmap.png"
    plt.savefig(filename)
    plt.close()

def draw_automata_graph(model_probabilities: Dict[str, Dict[str, float]]):
    """
    Leverages NetworkX to generate the actual directional visual graph model.
    Filters for stronger edges to prevent messy spaghetti visualizations.
    """
    G = nx.DiGraph()
    
    threshold = 0.05 # minimum probability to show link visually
    
    for src, dests in model_probabilities.items():
        for dst, prob in dests.items():
            if prob >= threshold:
                # Scale weight for visual thickness or keep raw for labeling
                G.add_edge(src, dst, weight=prob)
                
    plt.figure(figsize=(12, 10))
    
    # Use simple spring layout for clean node spacing
    pos = nx.spring_layout(G, k=0.6)
    
    # Edge widths correlated to likelihood
    edges = G.edges(data=True)
    weights = [edata['weight'] * 5 for _, _, edata in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=weights, alpha=0.6)
    
    plt.title("Probabilistic Automata State Diagram")
    plt.axis("off")
    plt.tight_layout()
    
    filename = SAVE_DIR / "automata_state_diagram.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved graphical state diagram to {filename}")

if __name__ == "__main__":
    # Generating mock test outputs to confirm code path integrity
    print("Testing Visualization Package pipeline integrity...")
    
    # Mock transition dict
    mock_probs = {
        'aaa': {'aab': 0.8, 'aaa': 0.2},
        'aab': {'aba': 0.9, 'bbb': 0.1},
        'aba': {'aaa': 1.0}
    }
    
    try:
        plot_transition_heatmap(mock_probs)
        draw_automata_graph(mock_probs)
        print("Mock graph components generated successfully.")
    except Exception as e:
        print(f"Visualization integration test failed: {e}")
