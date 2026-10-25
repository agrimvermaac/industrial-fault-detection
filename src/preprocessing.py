"""
Preprocessing Module for Vibration Signals.

Handles:
- Signal slicing and window segmentation.
- Split-before-segment or recording-level partitioning to strictly prevent data leakage.
- Normalization (StandardScaler fitted strictly on training data).
- Reproducible random seeding.
"""

from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import scipy.signal

from src.data_loader import load_cwru_mat


RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility across libraries."""
    np.random.seed(seed)


def load_and_preprocess_recording(
    file_path: Path,
    original_fs: int = 12000,
    target_fs: int = 12000
) -> Tuple[np.ndarray, int]:
    """
    Loads raw CWRU MATLAB file and performs anti-aliasing decimation if needed.

    Args:
        file_path: Path to .mat file.
        original_fs: Original sampling rate (e.g. 48000 for Normal baseline, 12000 for fault data).
        target_fs: Standard target sampling frequency (12000 Hz).

    Returns:
        Tuple[np.ndarray, int]: (preprocessed 1D signal, effective sampling rate)
    """
    record = load_cwru_mat(file_path)
    raw_sig = record["de_signal"]
    if raw_sig is None:
        raise ValueError(f"Drive End (DE) signal not found in {file_path}")

    raw_sig = np.asarray(raw_sig, dtype=np.float64)

    if original_fs > target_fs:
        decimate_factor = original_fs // target_fs
        # scipy.signal.decimate uses an 8th-order Chebyshev type I filter with anti-aliasing cutoff
        processed_sig = scipy.signal.decimate(raw_sig, q=decimate_factor, ftype="iir")
        effective_fs = target_fs
    else:
        processed_sig = raw_sig.copy()
        effective_fs = original_fs

    return processed_sig.astype(np.float64), effective_fs



def segment_signal(
    signal: np.ndarray,
    window_size: int = 2048,
    step_size: int = 1024
) -> np.ndarray:
    """
    Slices a continuous 1D vibration signal array into fixed-length windows.

    Args:
        signal: 1D numpy array of raw vibration signal.
        window_size: Number of samples per window (e.g., 2048 samples).
        step_size: Stride between successive windows. If equal to window_size, windows are non-overlapping.

    Returns:
        np.ndarray: 2D array of shape (num_windows, window_size).
    """
    if signal is None or len(signal) < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    num_windows = (len(signal) - window_size) // step_size + 1
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::step_size]
    return windows[:num_windows].astype(np.float32)


def temporal_train_test_split(
    signal: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a continuous time-series recording chronologically before windowing.

    CRITICAL DATA LEAKAGE PREVENTION:
    Splitting continuous signals BEFORE windowing prevents overlapping windows
    from the same timestamp from being distributed across train and test sets,
    which would otherwise lead to artificially inflated evaluation metrics.

    Args:
        signal: Continuous 1D vibration array.
        train_ratio: Proportion of signal for training.
        val_ratio: Proportion of signal for validation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (train_signal, val_signal, test_signal)
    """
    n = len(signal)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_sig = signal[:train_end]
    val_sig = signal[train_end:val_end]
    test_sig = signal[val_end:]

    return train_sig, val_sig, test_sig


def normalize_splits(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Z-score normalizes data splits using statistics computed strictly on the training set.

    Args:
        train_data: Training array.
        val_data: Validation array.
        test_data: Test array.

    Returns:
        Tuple: Normalized (train, val, test) and computed (mean, std).
    """
    mean = float(np.mean(train_data))
    std = float(np.std(train_data)) + 1e-8

    norm_train = (train_data - mean) / std
    norm_val = (val_data - mean) / std
    norm_test = (test_data - mean) / std

    return norm_train, norm_val, norm_test, mean, std
