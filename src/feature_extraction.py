"""
Feature Extraction Module for Vibration Signals.

Extracts time-domain statistical indicators and frequency-domain spectral features
from vibration signal windows for traditional machine learning classification.
"""

from typing import Dict, List, Optional
import numpy as np
import scipy.stats


def extract_statistical_features(window: np.ndarray) -> Dict[str, float]:
    """
    Computes the 7 core statistical features specified for baseline machine learning:
    - Mean
    - Standard deviation
    - Root Mean Square (RMS)
    - Peak amplitude
    - Peak-to-peak amplitude
    - Kurtosis (Fisher's definition, normal=0)
    - Crest factor (Peak / RMS)

    Safely handles numerical edge cases such as near-zero division and non-finite values.

    Args:
        window: 1D numpy array representing a single vibration window.

    Returns:
        Dict[str, float]: The 7 core statistical features.
    """
    window = np.asarray(window, dtype=np.float64)
    if len(window) == 0:
        return {
            "mean": 0.0, "std": 0.0, "rms": 0.0,
            "peak": 0.0, "peak_to_peak": 0.0,
            "kurtosis": 0.0, "crest_factor": 0.0
        }

    abs_window = np.abs(window)
    mean_val = float(np.mean(window))
    std_val = float(np.std(window))
    rms_val = float(np.sqrt(np.mean(window ** 2)))
    peak_val = float(np.max(abs_window))
    peak_to_peak = float(np.ptp(window))

    # Crest factor with epsilon to guard against division by zero
    crest_factor = float(peak_val / (rms_val + 1e-12))

    # Kurtosis (Fisher's definition: normal distribution = 0.0)
    # If standard deviation is zero (constant signal), kurtosis is 0.0
    if std_val < 1e-12:
        kurtosis_val = 0.0
    else:
        kurtosis_val = float(scipy.stats.kurtosis(window, fisher=True))

    # Ensure all return values are strictly finite
    res = {
        "mean": mean_val if np.isfinite(mean_val) else 0.0,
        "std": std_val if np.isfinite(std_val) else 0.0,
        "rms": rms_val if np.isfinite(rms_val) else 0.0,
        "peak": peak_val if np.isfinite(peak_val) else 0.0,
        "peak_to_peak": peak_to_peak if np.isfinite(peak_to_peak) else 0.0,
        "kurtosis": kurtosis_val if np.isfinite(kurtosis_val) else 0.0,
        "crest_factor": crest_factor if np.isfinite(crest_factor) else 0.0
    }
    return res


def extract_time_domain_features(window: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical time-domain features from a vibration signal window.

    Metrics include:
        - RMS (Root Mean Square): Indicates overall vibration energy.
        - Peak Amplitude: Maximum absolute deflection.
        - Peak-to-Peak: Total dynamic excursion.
        - Crest Factor: Peak / RMS (sensitive to impulsive shocks from early defects).
        - Kurtosis: 4th central moment (sharpness of peaks, high in rolling element faults).
        - Skewness: 3rd central moment (asymmetry of distribution).
        - Shape Factor: RMS / Rectified Mean.
        - Impulse Factor: Peak / Rectified Mean.
        - Margin Factor: Peak / (Mean(sqrt(|x|)))^2.

    Args:
        window: 1D numpy array representing a single window of vibration measurements.

    Returns:
        Dict[str, float]: Extracted time-domain features.
    """
    abs_window = np.abs(window)
    mean_val = float(np.mean(window))
    std_val = float(np.std(window))
    rectified_mean = float(np.mean(abs_window)) + 1e-8
    rms_val = float(np.sqrt(np.mean(window ** 2)))
    peak_val = float(np.max(abs_window))
    peak_to_peak = float(np.ptp(window))

    # Crest factor
    crest_factor = peak_val / (rms_val + 1e-8)

    # Kurtosis and Skewness
    kurtosis_val = float(scipy.stats.kurtosis(window))
    skewness_val = float(scipy.stats.skew(window))

    # Shape, Impulse, and Margin factors
    shape_factor = rms_val / rectified_mean
    impulse_factor = peak_val / rectified_mean
    margin_factor = peak_val / (np.mean(np.sqrt(abs_window)) ** 2 + 1e-8)

    return {
        "mean": mean_val,
        "std": std_val,
        "rms": rms_val,
        "peak": peak_val,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis_val,
        "skewness": skewness_val,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "margin_factor": margin_factor
    }


def extract_frequency_domain_features(window: np.ndarray, sampling_rate: int = 12000) -> Dict[str, float]:
    """
    Computes spectral features using Fast Fourier Transform (FFT).

    Args:
        window: 1D numpy array representing a single window.
        sampling_rate: Sampling frequency in Hz (e.g. 12000 Hz).

    Returns:
        Dict[str, float]: Extracted frequency-domain features (spectral centroid, peak frequency, spectral energy).
    """
    n = len(window)
    # Compute one-sided FFT
    fft_vals = np.fft.rfft(window)
    fft_magnitude = np.abs(fft_vals) / n
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    peak_freq_idx = np.argmax(fft_magnitude)
    peak_frequency = float(fft_freqs[peak_freq_idx])
    peak_magnitude = float(fft_magnitude[peak_freq_idx])

    # Spectral centroid (center of mass of spectrum)
    total_power = np.sum(fft_magnitude) + 1e-8
    spectral_centroid = float(np.sum(fft_freqs * fft_magnitude) / total_power)

    # Spectral energy
    spectral_energy = float(np.sum(fft_magnitude ** 2))

    return {
        "peak_frequency_hz": peak_frequency,
        "peak_magnitude": peak_magnitude,
        "spectral_centroid": spectral_centroid,
        "spectral_energy": spectral_energy
    }


def extract_window_features(window: np.ndarray, sampling_rate: int = 12000) -> Dict[str, float]:
    """
    Combines both time-domain and frequency-domain features for a window.

    Args:
        window: 1D numpy array.
        sampling_rate: Sampling frequency in Hz.

    Returns:
        Dict[str, float]: Combined feature dictionary.
    """
    features = extract_time_domain_features(window)
    features.update(extract_frequency_domain_features(window, sampling_rate))
    return features
