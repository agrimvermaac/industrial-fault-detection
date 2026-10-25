"""
Baseline Machine Learning Training Module.

Trains classical machine learning models (e.g., Random Forest, Support Vector Machine)
on statistical time- and frequency-domain vibration features.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


def train_random_forest_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42
):
    """
    Trains a Random Forest classifier on engineered vibration features.

    Args:
        X_train: 2D feature matrix (samples, features).
        y_train: 1D ground-truth class labels.
        n_estimators: Number of decision trees.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split an internal node.
        random_state: Seed for reproducibility.

    Returns:
        Fitted RandomForestClassifier instance.
    """
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Performs grid selection over Random Forest hyperparameters using the validation set.

    Args:
        X_train: Training feature matrix.
        y_train: Training class labels.
        X_val: Validation feature matrix.
        y_val: Validation class labels.
        random_state: Random seed.

    Returns:
        Tuple: (best_fitted_model, best_params_dict, best_val_f1_score)
    """
    from sklearn.metrics import precision_recall_fscore_support

    param_grid = [
        {"n_estimators": 50, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 4},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    ]

    best_f1 = -1.0
    best_params = param_grid[0]
    best_model = None

    for params in param_grid:
        clf = train_random_forest_baseline(
            X_train, y_train,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=random_state
        )
        val_preds = clf.predict(X_val)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="macro", zero_division=0
        )
        if macro_f1 > best_f1:
            best_f1 = float(macro_f1)
            best_params = params
            best_model = clf

    print(f"[TrainBaseline] Best validation configuration: {best_params} (Macro F1: {best_f1:.4f})")
    return best_model, best_params, best_f1



def train_svm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = "rbf",
    random_state: int = 42
):
    """
    Trains a Support Vector Classifier on scaled vibration features.

    Args:
        X_train: Scaled feature matrix.
        y_train: Ground truth class labels.
        C: Regularization parameter.
        kernel: Kernel type ('rbf', 'linear', etc.).
        random_state: Seed for reproducibility.

    Returns:
        Fitted SVC instance.
    """
    from sklearn.svm import SVC

    clf = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def save_baseline_model(model: Any, output_path: Path) -> Path:
    """Saves a fitted baseline model to disk via joblib."""
    import joblib

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(output_path))
    print(f"[TrainBaseline] Model saved to {output_path}")
    return output_path
