"""
Comprehensive Multi-Label Evaluation Module.
Computes per-label precision, recall, F1, ROC-AUC, Hamming Loss, and confusion matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    hamming_loss,
    confusion_matrix,
    classification_report
)

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Computes complete evaluation metrics across all 6 labels using tuned thresholds.
    """
    metrics_per_label = {}
    y_pred_binary = np.zeros_like(y_true)
    
    macro_p, macro_r, macro_f1, macro_auc = [], [], [], []
    
    for idx, col in enumerate(LABEL_COLUMNS):
        t = thresholds.get(col, 0.5)
        y_t = y_true[:, idx]
        p_col = probs[:, idx]
        preds = (p_col >= t).astype(int)
        y_pred_binary[:, idx] = preds
        
        p, r, f1, _ = precision_recall_fscore_support(y_t, preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_t, p_col)
        except Exception:
            auc = 0.5
            
        cm = confusion_matrix(y_t, preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        macro_p.append(p)
        macro_r.append(r)
        macro_f1.append(f1)
        macro_auc.append(auc)
        
        metrics_per_label[col] = {
            'threshold': round(float(t), 3),
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'support_positives': int(y_t.sum()),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
    h_loss = float(hamming_loss(y_true, y_pred_binary))
    exact_match = float(np.mean(np.all(y_true == y_pred_binary, axis=1)))
    
    overall = {
        'macro_precision': round(float(np.mean(macro_p)), 4),
        'macro_recall': round(float(np.mean(macro_r)), 4),
        'macro_f1': round(float(np.mean(macro_f1)), 4),
        'macro_roc_auc': round(float(np.mean(macro_auc)), 4),
        'hamming_loss': round(h_loss, 4),
        'exact_match_ratio': round(exact_match, 4)
    }
    
    return {
        'per_label': metrics_per_label,
        'overall': overall
    }
