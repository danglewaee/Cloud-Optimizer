from __future__ import annotations

import numpy as np


def build_sequences(values: np.ndarray, window_size: int, horizon_steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
    x_list = []
    y_list = []

    n = len(values)
    max_i = n - window_size - horizon_steps + 1
    for i in range(max_i):
        x_list.append(values[i : i + window_size])
        y_idx = i + window_size + horizon_steps - 1
        y_list.append(values[y_idx])

    if not x_list:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32), np.empty((0, values.shape[1]), dtype=np.float32)

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def normalize_features(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    normed = (values - mean) / std
    return normed, mean, std


def apply_norm(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = std.copy()
    safe_std[safe_std < 1e-6] = 1.0
    return (values - mean) / safe_std


def undo_norm(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std + mean
