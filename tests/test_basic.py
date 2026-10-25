"""
Basic Automated Tests for Industrial Fault Intelligence Project Structure and Modules.
"""

from pathlib import Path
import numpy as np
import pytest

from src.preprocessing import segment_signal, temporal_train_test_split, normalize_splits, set_seed
from src.feature_extraction import extract_time_domain_features, extract_frequency_domain_features, extract_window_features
from src.prioritization import compute_maintenance_priority


def test_directory_structure():
    """Verifies that all foundational project directories exist."""
    base_dir = Path(__file__).resolve().parent.parent
    expected_dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "notebooks",
        base_dir / "src",
        base_dir / "models",
        base_dir / "app",
        base_dir / "tests",
    ]
    for d in expected_dirs:
        assert d.exists(), f"Expected directory {d} does not exist"
        assert d.is_dir(), f"Path {d} is not a directory"


def test_core_dependencies_importable():
    """Verifies that all required third-party libraries import correctly."""
    import numpy
    import pandas
    import scipy
    import sklearn
    import torch
    import streamlit

    assert numpy.__version__ is not None
    assert pandas.__version__ is not None
    assert scipy.__version__ is not None
    assert sklearn.__version__ is not None
    assert torch.__version__ is not None
    assert streamlit.__version__ is not None


def test_segment_signal():
    """Tests signal window slicing functionality."""
    signal = np.arange(10000, dtype=np.float32)
    window_size = 1024
    step_size = 512

    windows = segment_signal(signal, window_size=window_size, step_size=step_size)
    assert windows.ndim == 2
    assert windows.shape[1] == window_size
    expected_windows = (len(signal) - window_size) // step_size + 1
    assert windows.shape[0] == expected_windows


def test_temporal_train_test_split_no_leakage():
    """
    Verifies temporal chronological splitting ensures no overlap
    or shared timestamps across train, validation, and test splits.
    """
    signal = np.arange(1000, dtype=np.float32)
    train_sig, val_sig, test_sig = temporal_train_test_split(signal, train_ratio=0.7, val_ratio=0.15)

    assert len(train_sig) == 700
    assert len(val_sig) == 150
    assert len(test_sig) == 150

    # Ensure strictly partitioned index ranges (no overlap / no leakage)
    assert train_sig[-1] < val_sig[0]
    assert val_sig[-1] < test_sig[0]


def test_feature_extraction_keys():
    """Verifies that feature extraction produces all required time and spectral indicators."""
    set_seed(42)
    # Synthetic test signal (pure sine wave + small noise) to test numerical pipeline
    t = np.linspace(0, 1, 2048)
    test_window = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(2048)

    features = extract_window_features(test_window, sampling_rate=12000)

    required_keys = [
        "mean", "std", "rms", "peak", "crest_factor",
        "kurtosis", "skewness", "shape_factor", "impulse_factor",
        "peak_frequency_hz", "peak_magnitude", "spectral_centroid", "spectral_energy"
    ]
    for key in required_keys:
        assert key in features, f"Missing feature key: {key}"
        assert not np.isnan(features[key]), f"Feature {key} is NaN"


def test_maintenance_prioritization_logic():
    """Verifies that higher fault severity and higher RMS result in higher priority scores."""
    normal_prio = compute_maintenance_priority(
        predicted_class="Normal",
        confidence=0.95,
        vibration_rms=0.08,
        baseline_rms=0.10,
        motor_load_hp=1.0
    )
    critical_prio = compute_maintenance_priority(
        predicted_class="Inner Race Fault",
        confidence=0.95,
        vibration_rms=0.35,
        baseline_rms=0.10,
        motor_load_hp=1.0
    )

    assert normal_prio["priority_tier"] == "Low"
    assert critical_prio["priority_tier"] in ["High", "Critical"]
    assert critical_prio["priority_score"] > normal_prio["priority_score"]


def test_cwru_raw_files_integrity():
    """Verifies that downloaded real CWRU .mat files load properly with scipy.io."""
    from src.data_loader import load_cwru_mat
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    expected_files = ["98.mat", "106.mat", "119.mat", "131.mat"]

    for fname in expected_files:
        fpath = raw_dir / fname
        assert fpath.exists(), f"Raw file {fname} not found in {raw_dir}"
        record = load_cwru_mat(fpath)
        sig = record["de_signal"]
        assert sig is not None, f"DE signal missing from {fname}"
        assert len(sig) > 100000, f"Signal in {fname} unexpectedly short: {len(sig)}"
        assert not np.isnan(sig).any(), f"Signal in {fname} contains NaNs"
        assert not np.isinf(sig).any(), f"Signal in {fname} contains Infs"


def test_statistical_features_edge_cases():
    """Tests numerical stability of extract_statistical_features against zeros and constants."""
    from src.feature_extraction import extract_statistical_features

    # Constant signal (std = 0)
    constant_win = np.ones(2048, dtype=np.float64) * 5.0
    feat_const = extract_statistical_features(constant_win)
    assert feat_const["std"] == 0.0
    assert feat_const["kurtosis"] == 0.0
    assert not np.isnan(feat_const["crest_factor"])
    assert not np.isinf(feat_const["crest_factor"])

    # All-zero signal
    zero_win = np.zeros(2048, dtype=np.float64)
    feat_zero = extract_statistical_features(zero_win)
    for k, v in feat_zero.items():
        assert not np.isnan(v), f"Feature {k} is NaN for zero signal"
        assert not np.isinf(v), f"Feature {k} is Inf for zero signal"


def test_saved_baseline_model_artifacts():
    """Verifies that the trained Random Forest model and metadata exist and load correctly."""
    import joblib
    import json

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"

    model_path = models_dir / "random_forest_baseline.joblib"
    feat_path = models_dir / "feature_columns.json"
    metrics_path = models_dir / "rf_evaluation_metrics.json"

    assert model_path.exists(), f"Model artifact missing: {model_path}"
    assert feat_path.exists(), f"Feature columns missing: {feat_path}"
    assert metrics_path.exists(), f"Metrics file missing: {metrics_path}"

    clf = joblib.load(str(model_path))
    with open(feat_path, "r") as f:
        feature_cols = json.load(f)

    assert len(feature_cols) == 7
    # Predict with dummy input
    dummy_input = np.ones((1, len(feature_cols)))
    pred = clf.predict(dummy_input)
    assert pred.shape == (1,)
    assert pred[0] in [0, 1, 2, 3]


def test_saved_cnn_model_artifacts():
    """Verifies that the trained 1D-CNN model loads and runs inference on a vibration tensor."""
    import torch
    import json
    from src.train_cnn import load_1d_cnn_model

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"

    model_path = models_dir / "bearing_1d_cnn.pt"
    config_path = models_dir / "cnn_config.json"
    prep_path = models_dir / "cnn_preprocessing_config.json"
    metrics_path = models_dir / "cnn_evaluation_metrics.json"

    assert model_path.exists(), f"CNN weights missing: {model_path}"
    assert config_path.exists(), f"CNN config missing: {config_path}"
    assert prep_path.exists(), f"CNN preprocessing config missing: {prep_path}"
    assert metrics_path.exists(), f"CNN metrics missing: {metrics_path}"

    with open(config_path, "r") as f:
        cfg = json.load(f)
    assert cfg["num_classes"] == 4
    assert cfg["input_channels"] == 1

    model = load_1d_cnn_model(model_path, num_classes=4, input_channels=1, device="cpu")
    dummy_tensor = torch.randn(2, 1, 2048)
    with torch.no_grad():
        logits = model(dummy_tensor)

    assert logits.shape == (2, 4), f"Unexpected logits shape: {logits.shape}"


def test_model_comparison_artifacts():
    """Verifies that model comparison exports exist and have valid structure."""
    import pandas as pd
    base_dir = Path(__file__).resolve().parent.parent
    comp_csv = base_dir / "models" / "model_comparison.csv"
    assert comp_csv.exists(), f"Comparison CSV missing: {comp_csv}"
    df = pd.read_csv(comp_csv)
    assert len(df) == 2
    assert "Random Forest" in df["Model"].values
    assert "1D-CNN" in df["Model"].values
    assert "Test Accuracy" in df.columns
    assert "Macro F1" in df.columns



