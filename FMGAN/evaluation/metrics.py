"""
Evaluation metrics for time series imputation.
Computes errors ONLY on artificially masked positions (indicating_mask).
"""

import numpy as np


def mae(pred, target, indicating_mask):
    """Mean Absolute Error on masked positions."""
    diff = np.abs(pred - target)
    return (diff * indicating_mask).sum() / indicating_mask.sum()


def mse(pred, target, indicating_mask):
    """Mean Squared Error on masked positions."""
    diff = (pred - target) ** 2
    return (diff * indicating_mask).sum() / indicating_mask.sum()


def rmse(pred, target, indicating_mask):
    """Root Mean Squared Error on masked positions."""
    return np.sqrt(mse(pred, target, indicating_mask))


def mre(pred, target, indicating_mask):
    """Mean Relative Error on masked positions."""
    diff = np.abs(pred - target)
    abs_target = np.abs(target) + 1e-8
    return ((diff / abs_target) * indicating_mask).sum() / indicating_mask.sum()


def compute_all_metrics(pred, target, indicating_mask):
    """
    Compute all standard imputation metrics.

    Args:
        pred: (N, T, F) predicted values
        target: (N, T, F) ground truth values
        indicating_mask: (N, T, F) binary mask for evaluation positions

    Returns:
        dict of metric_name -> value
    """
    return {
        'MAE': float(mae(pred, target, indicating_mask)),
        'MSE': float(mse(pred, target, indicating_mask)),
        'RMSE': float(rmse(pred, target, indicating_mask)),
        'MRE': float(mre(pred, target, indicating_mask)),
    }
