import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def compare_groups(correct_scores: List[float], hallucinated_scores: List[float]) -> Dict[str, float]:
    """Compare RSI between correct and hallucinated groups using Welch's t-test."""
    if not correct_scores or not hallucinated_scores:
        return {}
        
    correct = np.array(correct_scores)
    hallucinated = np.array(hallucinated_scores)
    
    t_stat, p_val = stats.ttest_ind(hallucinated, correct, equal_var=False)
    
    # Cohen's d
    n1, n2 = len(hallucinated), len(correct)
    var1, var2 = np.var(hallucinated, ddof=1), np.var(correct, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(hallucinated) - np.mean(correct)) / pooled_sd if pooled_sd > 0 else 0
    
    return {
        "mean_correct": float(np.mean(correct)),
        "std_correct": float(np.std(correct)),
        "n_correct": n2,
        "mean_hallucinated": float(np.mean(hallucinated)),
        "std_hallucinated": float(np.std(hallucinated)),
        "n_hallucinated": n1,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d)
    }

def compute_correlations(rsi_values: List[float], metric_values: List[float]) -> Dict[str, float]:
    """Compute Pearson and Spearman correlations between RSI and continuous quality metrics (e.g., F1)."""
    if len(rsi_values) < 2 or len(metric_values) < 2:
        return {}
        
    pearson_r, p_pearson = stats.pearsonr(rsi_values, metric_values)
    spearman_rho, p_spearman = stats.spearmanr(rsi_values, metric_values)
    
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(p_pearson),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(p_spearman)
    }

def compute_roc_auc(labels: List[bool], scores: List[float]) -> Dict[str, Any]:
    """Compute ROC and AUC for a given predictor."""
    if len(set(labels)) < 2:
        return {"auc": 0.5, "fpr": [], "tpr": []}
        
    # Standardize labels (True = hallucination = positive class)
    y_true = np.array([1 if l else 0 for l in labels])
    y_score = np.array(scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # Youden's J for optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        "auc": float(roc_auc),
        "ap": float(ap),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "optimal_threshold": float(optimal_threshold)
    }

def bootstrap_auc(labels: List[bool], scores: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute bootstrap confidence intervals for AUC."""
    y_true = np.array([1 if l else 0 for l in labels])
    y_score = np.array(scores)
    
    n = len(y_true)
    aucs = []
    
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        curr_auc = roc_auc_score(y_true[indices], y_score[indices])
        aucs.append(curr_auc)
        
    if not aucs:
        return 0.5, 0.5, 0.5
        
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    
    return float(np.mean(aucs)), float(lower), float(upper)

def roc_auc_score(y_true, y_score):
    """Helper for bootstrap"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)
