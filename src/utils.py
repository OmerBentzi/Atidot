"""
Utility functions for the Atidot Decision Assistant.
Includes seed management, metrics, and helper functions.
"""
import os
import random
import numpy as np
from typing import Tuple
from sklearn.metrics import average_precision_score


def set_seeds(seed: int = 42) -> None:
    """
    Set all random seeds deterministically.
    
    Args:
        seed: Random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_ratio: float) -> float:
    """
    Compute precision at top k% of predictions (by score descending).
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities/scores
        k_ratio: Fraction of top predictions to consider (e.g., 0.01 for top 1%)
    
    Returns:
        Precision at top k%
    """
    if len(y_true) == 0 or len(y_score) == 0:
        return 0.0
    
    n_top = max(1, int(len(y_true) * k_ratio))
    top_indices = np.argsort(y_score)[::-1][:n_top]
    top_labels = y_true[top_indices]
    
    if len(top_labels) == 0:
        return 0.0
    
    return float(np.sum(top_labels) / len(top_labels))


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve (Average Precision).
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities
    
    Returns:
        AUC-PR score
    """
    return float(average_precision_score(y_true, y_score))


def month_to_ordinal(month_str: str) -> int:
    """
    Convert month string 'YYYY-MM' to ordinal integer.
    
    Args:
        month_str: Month string in format 'YYYY-MM'
    
    Returns:
        Ordinal integer (e.g., '2023-01' -> 202301)
    
    Examples:
        >>> month_to_ordinal('2023-01')
        202301
        >>> month_to_ordinal('2023-12')
        202312
    """
    year, month = month_str.split('-')
    return int(year) * 100 + int(month)
