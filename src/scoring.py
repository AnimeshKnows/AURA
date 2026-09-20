"""Anomaly scoring utilities for AUROC evaluation."""

from typing import Optional, Sequence, Union

import numpy as np
from sklearn.metrics import roc_auc_score


def image_anomaly_score(img: np.ndarray, recon_img: np.ndarray) -> float:
    """Mean absolute reconstruction error over all pixels (scalar anomaly score)."""
    return float(np.mean(np.abs(img - recon_img)))


def compute_auroc(
    y_true: Sequence[Union[int, float]],
    y_scores: Sequence[float],
) -> Optional[float]:
    """Compute ROC AUROC, or None if undefined (empty / single-class labels)."""
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    if y_true_arr.size == 0 or y_scores_arr.size == 0:
        return None
    if y_true_arr.shape[0] != y_scores_arr.shape[0]:
        raise ValueError(
            f"y_true and y_scores length mismatch: "
            f"{y_true_arr.shape[0]} vs {y_scores_arr.shape[0]}"
        )
    if np.unique(y_true_arr).size < 2:
        return None

    return float(roc_auc_score(y_true_arr, y_scores_arr))
