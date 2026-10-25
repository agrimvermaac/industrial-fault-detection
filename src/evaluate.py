"""
Evaluation Module for Fault Classification Models.

Computes comprehensive classification metrics (Accuracy, Precision, Recall, F1-score)
and generates confusion matrices to validate model performance without leakage.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics.

    Args:
        y_true: Ground truth integer class labels.
        y_pred: Predicted integer class labels.
        target_names: List of class names for reporting.

    Returns:
        Dict[str, Any]: Dictionary containing accuracy, macro F1, weighted F1,
                        classification report dict, and confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix
    )

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report": report
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Bearing Fault Confusion Matrix",
    save_path: Optional[Path] = None
):
    """
    Renders and optionally saves a Seaborn heatmap of the confusion matrix.

    Args:
        cm: 2D numpy confusion matrix.
        class_names: List of class labels for axes.
        title: Plot title.
        save_path: Optional path to save the generated figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Evaluate] Confusion matrix saved to {save_path}")

    return fig


def plot_class_distribution(
    df: Any,
    class_col: str = "class_label",
    split_col: str = "split",
    title: str = "Window Distribution Across Dataset Splits",
    save_path: Optional[Path] = None
):
    """
    Plots a grouped bar chart displaying window count for each class across splits.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    crosstab = pd.crosstab(df[class_col], df[split_col])
    cols = [c for c in ["train", "val", "test"] if c in crosstab.columns]
    crosstab = crosstab[cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    crosstab.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85, color=["#2b5c8f", "#7570b3", "#d95f02"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Bearing Condition Class", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Non-Overlapping Windows", fontsize=10, fontweight="bold")
    ax.legend(title="Dataset Split", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fontsize=9, padding=2)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Evaluate] Class distribution plot saved to {save_path}")
    return fig


def plot_feature_distributions(
    df: Any,
    feature_cols: List[str],
    class_col: str = "class_label",
    title: str = "Statistical Feature Distributions by Bearing Condition",
    save_path: Optional[Path] = None
):
    """
    Plots boxplots comparing extracted features across the bearing fault classes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_features = len(feature_cols)
    cols = min(4, n_features)
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    palette = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]

    for idx, feat in enumerate(feature_cols):
        ax = axes[idx]
        sns.boxplot(x=class_col, y=feat, data=df, ax=ax, palette=palette, hue=class_col, legend=False)
        ax.set_title(f"Feature: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Metric Value", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=25)

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Evaluate] Feature distribution plot saved to {save_path}")
    return fig


def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    class_names: List[str],
    title: str = "Random Forest: Per-Class Evaluation Metrics (Test Set)",
    save_path: Optional[Path] = None
):
    """
    Plots a grouped bar chart displaying Precision, Recall, and F1-score for each class.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    precisions = [report_dict[cls]["precision"] for cls in class_names]
    recalls = [report_dict[cls]["recall"] for cls in class_names]
    f1s = [report_dict[cls]["f1-score"] for cls in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, precisions, width, label="Precision", color="#2b5c8f", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x, recalls, width, label="Recall", color="#d95f02", edgecolor="black", alpha=0.85)
    b3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#7570b3", edgecolor="black", alpha=0.85)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    for bar_group in [b1, b2, b3]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.02, f"{h:.2f}",
                    ha='center', va='bottom', fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Evaluate] Per-class metrics plot saved to {save_path}")
    return fig


def plot_cnn_training_curves(
    history: Dict[str, List[float]],
    title: str = "1D-CNN Training & Validation Learning Curves",
    save_path: Optional[Path] = None
):
    """
    Renders loss and accuracy progression across training epochs.
    """
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], label="Training Loss", color="#2b5c8f", linewidth=1.8, marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="#d95f02", linewidth=1.8, linestyle="--", marker="s", markersize=3)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Training Accuracy", color="#2b5c8f", linewidth=1.8, marker="o", markersize=3)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation Accuracy", color="#2ca02c", linewidth=1.8, linestyle="--", marker="^", markersize=3)
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Evaluate] CNN learning curves saved to {save_path}")
    return fig


def plot_model_comparison(
    comparison_df: Any,
    metrics: List[str] = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
    title: str = "Performance Comparison: Random Forest Baseline vs. 1D-CNN",
    save_path: Optional[Path] = None
):
    """
    Renders a side-by-side grouped bar chart comparing Random Forest vs. 1D-CNN on the test set.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    models = comparison_df["Model"].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    rf_scores = [comparison_df.loc[comparison_df["Model"] == "Random Forest", m].values[0] for m in metrics]
    cnn_scores = [comparison_df.loc[comparison_df["Model"] == "1D-CNN", m].values[0] for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, rf_scores, width, label="Random Forest Baseline", color="#2b5c8f", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x + width/2, cnn_scores, width, label="1D-CNN Deep Learning", color="#d95f02", edgecolor="black", alpha=0.85)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    for bar_group in [b1, b2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.02, f"{h:.4f}",
                    ha='center', va='bottom', fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Evaluate] Model comparison plot saved to {save_path}")
    return fig


