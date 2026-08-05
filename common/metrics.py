"""Common error metrics for NNLCI predictions."""

from __future__ import annotations

import numpy as np


def relative_l1(prediction: np.ndarray, target: np.ndarray, axis=0, eps: float = 1e-8) -> np.ndarray:
    """Return ||prediction-target||_1 / ||target||_1 along ``axis``."""
    numerator = np.sum(np.abs(prediction - target), axis=axis)
    denominator = np.sum(np.abs(target), axis=axis)
    return numerator / (denominator + eps)


def relative_l2(prediction: np.ndarray, target: np.ndarray, axis=0, eps: float = 1e-8) -> np.ndarray:
    """Return ||prediction-target||_2 / ||target||_2 along ``axis``."""
    numerator = np.sqrt(np.sum((prediction - target) ** 2, axis=axis))
    denominator = np.sqrt(np.sum(target**2, axis=axis))
    return numerator / (denominator + eps)

