"""Reusable data transforms shared by NNLCI problem workflows."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view


def z_score(values: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize values with a numerically safe denominator."""
    return (values - mean) / (std + eps)


def minmax_minus_one_to_one(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Scale feature columns to [-1, 1]."""
    return 2.0 * (values - lower) / (upper - lower + eps) - 1.0


def interpolate_1d(values: np.ndarray, target_size: int) -> np.ndarray:
    """Linearly interpolate an (N, C) field to ``target_size`` points."""
    source_x = np.linspace(0.0, 1.0, values.shape[0])
    target_x = np.linspace(0.0, 1.0, target_size)
    result = np.empty((target_size, values.shape[1]), dtype=values.dtype)
    for channel in range(values.shape[1]):
        result[:, channel] = interp1d(
            source_x, values[:, channel], kind="linear", fill_value="extrapolate"
        )(target_x)
    return result


def local_windows(values: np.ndarray, window_size: int, axis: int = 0) -> np.ndarray:
    """Return flattened local windows while preserving leading batch axes."""
    windows = sliding_window_view(values, window_shape=window_size, axis=axis)
    return windows.reshape(*windows.shape[:-2], -1)
