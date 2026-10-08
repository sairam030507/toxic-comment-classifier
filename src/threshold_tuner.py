"""
Threshold Tuning Module (Proposal Section 5.3).
Tunes decision thresholds independently for each of the 6 labels on the validation set
by sweeping candidate thresholds to maximize per-label F1.
Freezes tuned thresholds and evaluates on test set, comparing against flat 0.5 baseline.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def tune_thresholds_for_labels(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    threshold_range: Tuple[float, float, float] = (0.05, 0.90, 0.01)
) -> Dict[str, Any]:
    """
    Sweeps candidate thresholds independently for each label on validation set
    to maximize F1 score.
    
    Returns dictionary with:
    - best_thresholds: Dict[str, float]
    - val_metrics: Dict[str, Dict[str, float]]
    - curves: Dict[str, List[Dict[str, float]]] (threshold sweep curves for visualization)
    """
    min_t, max_t, step = threshold_range
    candidates = np.arange(min_t, max_t + step / 2, step)
    
    best_thresholds = {}
    val_results = {}
    curves = {}
    
    for idx, col in enumerate(LABEL_COLUMNS):
        y_true = y_val[:, idx]
        probs = probs_val[:, idx]
        
        best_f1 = -1.0
        best_t = 0.5
        best_prec = 0.0
        best_rec = 0.0
        
        curve_data = []
        
        for t in candidates:
            preds = (probs >= t).astype(int)
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            
            curve_data.append({
                'threshold': round(float(t), 2),
                'precision': round(float(p), 4),
                'recall': round(float(r), 4),
                'f1': round(float(f1), 4)
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
                best_prec = float(p)
                best_rec = float(r)
                
        best_thresholds[col] = round(best_t, 3)
        curves[col] = curve_data
        val_results[col] = {
            'optimal_threshold': round(best_t, 3),
            'val_f1': round(best_f1, 4),
            'val_precision': round(best_prec, 4),
            'val_recall': round(best_rec, 4)
        }
        print(f"Label '{col:13s}': Tuned Threshold = {best_t:.2f} | Val F1 = {best_f1:.4f} (P={best_prec:.4f}, R={best_rec:.4f})")
        
    return {
        'best_thresholds': best_thresholds,
        'val_results': val_results,
        'curves': curves
    }


def compare_flat_vs_tuned(
    y_test: np.ndarray,
    probs_test: np.ndarray,
    tuned_thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Evaluates test set under flat 0.5 cutoff vs tuned thresholds.
    Generates measurable before-and-after comparison as requested in Proposal Section 5.3.
    """
    comparison = {}
    
    flat_macro_f1 = []
    tuned_macro_f1 = []
    
    for idx, col in enumerate(LABEL_COLUMNS):
        y_true = y_test[:, idx]
        probs = probs_test[:, idx]
        tuned_t = tuned_thresholds.get(col, 0.5)
        
        # Flat 0.5 cutoff
        preds_flat = (probs >= 0.5).astype(int)
        p_flat = precision_score(y_true, preds_flat, zero_division=0)
        r_flat = recall_score(y_true, preds_flat, zero_division=0)
        f1_flat = f1_score(y_true, preds_flat, zero_division=0)
        
        # Tuned cutoff
        preds_tuned = (probs >= tuned_t).astype(int)
        p_tuned = precision_score(y_true, preds_tuned, zero_division=0)
        r_tuned = recall_score(y_true, preds_tuned, zero_division=0)
        f1_tuned = f1_score(y_true, preds_tuned, zero_division=0)
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.5
            
        flat_macro_f1.append(f1_flat)
        tuned_macro_f1.append(f1_tuned)
        
        gain_pct = ((f1_tuned - f1_flat) / max(f1_flat, 1e-4)) * 100.0 if f1_flat > 0 else 0.0
        
        comparison[col] = {
            'roc_auc': round(float(auc), 4),
            'flat_0_5': {
                'threshold': 0.5,
                'precision': round(float(p_flat), 4),
                'recall': round(float(r_flat), 4),
                'f1': round(float(f1_flat), 4)
            },
            'tuned': {
                'threshold': round(float(tuned_t), 3),
                'precision': round(float(p_tuned), 4),
                'recall': round(float(r_tuned), 4),
                'f1': round(float(f1_tuned), 4)
            },
            'f1_gain_absolute': round(float(f1_tuned - f1_flat), 4),
            'f1_gain_percent': round(float(gain_pct), 2)
        }
        
    macro_summary = {
        'flat_macro_f1': round(float(np.mean(flat_macro_f1)), 4),
        'tuned_macro_f1': round(float(np.mean(tuned_macro_f1)), 4),
        'macro_gain_absolute': round(float(np.mean(tuned_macro_f1) - np.mean(flat_macro_f1)), 4)
    }
    
    return {
        'per_label': comparison,
        'macro_summary': macro_summary
    }


def save_thresholds(tuning_data: Dict[str, Any], filepath: str = "models/thresholds.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(tuning_data, f, indent=2)
    print(f"Saved tuned thresholds to {filepath}")


def load_thresholds(filepath: str = "models/thresholds.json") -> Dict[str, float]:
    if not os.path.exists(filepath):
        # Fallback to recommended defaults if file doesn't exist yet
        return {
            'toxic': 0.45,
            'severe_toxic': 0.25,
            'obscene': 0.40,
            'threat': 0.20,
            'insult': 0.40,
            'identity_hate': 0.25
        }
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get('best_thresholds', data)
