"""
End-to-End Pipeline: Feature Extraction and Random Forest Baseline.

Executes:
1. Signal preprocessing and anti-aliasing decimation (48 kHz -> 12 kHz).
2. Chronological split-before-windowing (70% train / 15% val / 15% test).
3. Non-overlapping window segmentation (N = 2048 samples).
4. Statistical feature extraction (Mean, Std, RMS, Peak, Peak-to-Peak, Kurtosis, Crest Factor).
5. Random Forest model training, validation tuning, and final test set evaluation.
6. Generation and saving of model artifacts, metrics, and visualization figures.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import joblib

from src.data_loader import load_cwru_mat, FAULT_CLASSES
from src.preprocessing import (
    temporal_train_test_split,
    segment_signal,
    load_and_preprocess_recording,
    set_seed,
    RANDOM_SEED
)
from src.feature_extraction import extract_statistical_features
from src.train_baseline import (
    train_random_forest_baseline,
    tune_random_forest,
    save_baseline_model
)
from src.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_feature_distributions,
    plot_per_class_metrics
)


def run_pipeline():
    print("=" * 80)
    print(" INDUSTRIAL FAULT DETECTION: FEATURE EXTRACTION & RANDOM FOREST BASELINE")
    print("=" * 80)

    set_seed(RANDOM_SEED)

    # Output directories
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    figures_dir = PROJECT_ROOT / "notebooks" / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # PART 1: VERIFY PREPROCESSING & INGESTION
    # ---------------------------------------------------------
    print("\n--- PART 1: PREPROCESSING & SIGNAL VERIFICATION ---")
    manifest_path = PROJECT_ROOT / "data" / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    raw_dir = PROJECT_ROOT / "data" / "raw"

    print("Verifying 4 selected classes and source files:")
    preprocessed_signals = {}
    for _, row in manifest.iterrows():
        fpath = raw_dir / row["file_name"]
        sig, eff_fs = load_and_preprocess_recording(
            fpath,
            original_fs=int(row["sampling_rate_hz"]),
            target_fs=12000
        )
        assert not np.isnan(sig).any(), f"NaNs detected in {row['file_name']}"
        assert not np.isinf(sig).any(), f"Infs detected in {row['file_name']}"

        preprocessed_signals[row["class_id"]] = {
            "file_name": row["file_name"],
            "class_label": row["class_label"],
            "signal": sig,
            "sampling_rate": eff_fs,
            "raw_length": row["raw_signal_length"],
            "processed_length": len(sig)
        }
        print(f"  Class {row['class_id']} [{row['class_label']}]: {row['file_name']} -> "
              f"Raw={row['raw_signal_length']}, Processed={len(sig)} samples @ {eff_fs} Hz")

    # ---------------------------------------------------------
    # PART 2: CREATE SIGNAL WINDOWS (LEAKAGE-SAFE SPLITTING)
    # ---------------------------------------------------------
    print("\n--- PART 2: LEAKAGE-FREE CHRONOLOGICAL SPLITTING & WINDOWING ---")
    window_size = 2048  # Captures ~5 complete motor shaft revolutions at 1772 RPM
    step_size = 2048    # Non-overlapping windows
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    print(f"Window configuration: window_size={window_size}, step_size={step_size} (Non-overlapping)")
    print(f"Chronological split ratios: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")

    window_records = []
    feature_rows = []
    core_features = ["mean", "std", "rms", "peak", "peak_to_peak", "kurtosis", "crest_factor"]

    for class_id, info in preprocessed_signals.items():
        sig = info["signal"]
        # Split chronologically BEFORE windowing
        train_sig, val_sig, test_sig = temporal_train_test_split(
            sig, train_ratio=train_ratio, val_ratio=val_ratio
        )

        splits = [("train", train_sig), ("val", val_sig), ("test", test_sig)]

        for split_name, split_sig in splits:
            windows = segment_signal(split_sig, window_size=window_size, step_size=step_size)
            num_win = len(windows)
            window_records.append({
                "class_id": class_id,
                "class_label": info["class_label"],
                "file_name": info["file_name"],
                "split": split_name,
                "split_samples": len(split_sig),
                "num_windows": num_win
            })

            # ---------------------------------------------------------
            # PART 3: FEATURE EXTRACTION
            # ---------------------------------------------------------
            for w_idx, win in enumerate(windows):
                stats = extract_statistical_features(win)
                row_dict = {
                    "source_file": info["file_name"],
                    "class_id": class_id,
                    "class_label": info["class_label"],
                    "split": split_name,
                    "window_idx": w_idx,
                }
                row_dict.update(stats)
                feature_rows.append(row_dict)

    # Convert to DataFrames
    split_summary_df = pd.DataFrame(window_records)
    crosstab_summary = pd.crosstab(split_summary_df["class_label"], split_summary_df["split"],
                                   values=split_summary_df["num_windows"], aggfunc="sum")
    crosstab_summary = crosstab_summary[["train", "val", "test"]]
    crosstab_summary["Total"] = crosstab_summary.sum(axis=1)

    print("\nWindow Segmentation Summary per Class and Split:")
    print(crosstab_summary.to_string())

    feature_df = pd.DataFrame(feature_rows)
    print(f"\nExtracted total feature records: {len(feature_df)}")

    # Save feature tables
    train_features_df = feature_df[feature_df["split"] == "train"].copy()
    val_features_df = feature_df[feature_df["split"] == "val"].copy()
    test_features_df = feature_df[feature_df["split"] == "test"].copy()

    train_features_df.to_csv(processed_dir / "features_train.csv", index=False)
    val_features_df.to_csv(processed_dir / "features_val.csv", index=False)
    test_features_df.to_csv(processed_dir / "features_test.csv", index=False)
    feature_df.to_csv(processed_dir / "features_all.csv", index=False)
    print(f"Saved feature tables to {processed_dir}")

    # ---------------------------------------------------------
    # PART 4: RANDOM FOREST BASELINE TRAINING & TUNING
    # ---------------------------------------------------------
    print("\n--- PART 4: RANDOM FOREST TRAINING & MODEL SELECTION ---")
    X_train = train_features_df[core_features].values
    y_train = train_features_df["class_id"].values

    X_val = val_features_df[core_features].values
    y_val = val_features_df["class_id"].values

    X_test = test_features_df[core_features].values
    y_test = test_features_df["class_id"].values

    # Tune hyperparameters on validation split
    best_rf, best_params, val_f1 = tune_random_forest(
        X_train, y_train, X_val, y_val, random_state=RANDOM_SEED
    )

    # Evaluate on Train, Val, and Test splits
    target_names = [FAULT_CLASSES[i] for i in range(4)]

    train_metrics = compute_classification_metrics(y_train, best_rf.predict(X_train), target_names)
    val_metrics = compute_classification_metrics(y_val, best_rf.predict(X_val), target_names)
    test_metrics = compute_classification_metrics(y_test, best_rf.predict(X_test), target_names)

    print(f"\nTraining Set Accuracy   : {train_metrics['accuracy']:.4f} (Macro F1: {train_metrics['macro_f1']:.4f})")
    print(f"Validation Set Accuracy : {val_metrics['accuracy']:.4f} (Macro F1: {val_metrics['macro_f1']:.4f})")
    print(f"Test Set Accuracy       : {test_metrics['accuracy']:.4f} (Macro F1: {test_metrics['macro_f1']:.4f})")

    print("\nTest Set Classification Report:")
    report_df = pd.DataFrame(test_metrics["classification_report"]).transpose()
    print(report_df.round(4).to_string())

    print("\nTest Set Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    # Feature Importances
    importances = best_rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": core_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    print("\nFeature Importances:")
    print(feat_imp_df.to_string(index=False))

    # Save model artifacts
    save_baseline_model(best_rf, models_dir / "random_forest_baseline.joblib")

    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(core_features, f, indent=2)

    # Save metrics JSON (converting numpy types to python natives)
    evaluation_record = {
        "model_name": "RandomForestClassifier",
        "hyperparameters": best_params,
        "features": core_features,
        "random_seed": RANDOM_SEED,
        "window_size": window_size,
        "step_size": step_size,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "train_metrics": {
            "accuracy": train_metrics["accuracy"],
            "macro_precision": train_metrics["macro_precision"],
            "macro_recall": train_metrics["macro_recall"],
            "macro_f1": train_metrics["macro_f1"]
        },
        "val_metrics": {
            "accuracy": val_metrics["accuracy"],
            "macro_precision": val_metrics["macro_precision"],
            "macro_recall": val_metrics["macro_recall"],
            "macro_f1": val_metrics["macro_f1"]
        },
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            "classification_report": test_metrics["classification_report"]
        }
    }

    with open(models_dir / "rf_evaluation_metrics.json", "w") as f:
        json.dump(evaluation_record, f, indent=2)

    report_df.to_csv(models_dir / "rf_classification_report.csv")
    print(f"Saved model, feature order, and metrics to {models_dir}")

    # ---------------------------------------------------------
    # PART 5: VISUALIZATIONS
    # ---------------------------------------------------------
    print("\n--- PART 5: GENERATING VISUALIZATION FIGURES ---")

    # 1. Class Distribution Across Splits
    fig_dist = plot_class_distribution(
        feature_df,
        class_col="class_label",
        split_col="split",
        title="Class Distribution Across Dataset Splits (Non-Overlapping Windows)",
        save_path=figures_dir / "04_class_distribution_splits.png"
    )

    # 2. Feature Distributions by Class
    fig_feat = plot_feature_distributions(
        feature_df,
        feature_cols=core_features,
        class_col="class_label",
        title="Distribution of 7 Core Statistical Features by Bearing Condition",
        save_path=figures_dir / "05_feature_distributions.png"
    )

    # 3. Confusion Matrix (Test Set)
    fig_cm = plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=target_names,
        title="Random Forest Baseline: Test Set Confusion Matrix",
        save_path=figures_dir / "06_rf_confusion_matrix.png"
    )

    # 4. Per-Class Precision, Recall, F1
    fig_metrics = plot_per_class_metrics(
        test_metrics["classification_report"],
        class_names=target_names,
        title="Random Forest Baseline: Per-Class Performance (Test Set)",
        save_path=figures_dir / "07_rf_per_class_metrics.png"
    )

    print(f"Saved all 4 evaluation figures to {figures_dir}")
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    return evaluation_record


if __name__ == "__main__":
    run_pipeline()
