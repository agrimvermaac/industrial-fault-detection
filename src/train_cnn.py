"""
Deep Learning Module: 1D-CNN for Vibration Signal Classification.

Defines a PyTorch 1D Convolutional Neural Network that accepts raw or normalized
vibration time-series windows directly and classifies into 4 bearing health states.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np


# Lazy import torch inside functions/classes to avoid import errors prior to complete package installation
def get_torch_modules():
    """Helper to lazily import PyTorch modules."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    return torch, nn, Dataset, DataLoader


def build_1d_cnn_model(num_classes: int = 4, input_channels: int = 1):
    """
    Constructs a 1D Convolutional Neural Network for end-to-end vibration signal classification.

    Architecture:
        - Block 1: Conv1D (kernel_size=64, stride=16) -> BatchNorm1d -> ReLU -> MaxPool1d
        - Block 2: Conv1D (kernel_size=16, stride=1) -> BatchNorm1d -> ReLU -> MaxPool1d
        - Block 3: Conv1D (kernel_size=3, stride=1) -> BatchNorm1d -> ReLU -> AdaptiveAvgPool1d(1)
        - Fully Connected head: Linear -> Dropout -> Linear(num_classes)
    """
    torch, nn, _, _ = get_torch_modules()

    class Bearing1DCNN(nn.Module):
        def __init__(self, in_ch: int = 1, n_classes: int = 4):
            super().__init__()
            self.features = nn.Sequential(
                # First conv layer uses large receptive field to capture low-frequency trends
                nn.Conv1d(in_ch, 32, kernel_size=64, stride=8, padding=28),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),

                # Second conv layer captures mid-range harmonic patterns
                nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=8),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),

                # Third conv layer captures localized high-frequency transients
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.classifier(x)
            return x

    return Bearing1DCNN(in_ch=input_channels, n_classes=num_classes)


def save_cnn_checkpoint(model, optimizer, epoch: int, metrics: Dict[str, float], output_path: Path):
    """Saves PyTorch model checkpoint to disk."""
    import torch
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }, str(output_path))
    print(f"[TrainCNN] Checkpoint saved to {output_path}")


def load_1d_cnn_model(
    weights_path: Path,
    num_classes: int = 4,
    input_channels: int = 1,
    device: str = "cpu"
):
    """
    Loads a trained 1D-CNN model from weights file for evaluation or Streamlit inference.

    Args:
        weights_path: Path to saved .pt checkpoint or state_dict.
        num_classes: Number of fault classes (4).
        input_channels: Number of signal channels (1).
        device: 'cpu' or 'cuda'.

    Returns:
        Loaded model in eval mode.
    """
    import torch
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"CNN model weights not found at {weights_path}")

    model = build_1d_cnn_model(num_classes=num_classes, input_channels=input_channels)
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unrecognized checkpoint format in {weights_path}")

    model.to(device)
    model.eval()
    return model


def train_cnn_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cpu"
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Trains the 1D-CNN model with validation loss tracking and early stopping.

    Args:
        model: PyTorch nn.Module.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        epochs: Maximum number of epochs.
        lr: Adam learning rate.
        weight_decay: L2 regularization penalty.
        patience: Early stopping patience epochs.
        device: 'cpu' or 'cuda'.

    Returns:
        Tuple: (best_fitted_model, training_history_dict)
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import precision_recall_fscore_support

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss_total, train_correct, train_samples = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_samples += len(y_batch)

        epoch_train_loss = train_loss_total / train_samples
        epoch_train_acc = train_correct / train_samples

        # Validation phase
        model.eval()
        val_loss_total, val_correct, val_samples = 0.0, 0, 0
        val_preds_all, val_targets_all = [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss_total += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_samples += len(y_batch)

                val_preds_all.extend(preds.cpu().numpy().tolist())
                val_targets_all.extend(y_batch.cpu().numpy().tolist())

        epoch_val_loss = val_loss_total / val_samples
        epoch_val_acc = val_correct / val_samples
        _, _, epoch_val_f1, _ = precision_recall_fscore_support(
            val_targets_all, val_preds_all, average="macro", zero_division=0
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(float(epoch_train_loss))
        history["val_loss"].append(float(epoch_val_loss))
        history["train_acc"].append(float(epoch_train_acc))
        history["val_acc"].append(float(epoch_val_acc))
        history["val_macro_f1"].append(float(epoch_val_f1))

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[TrainCNN] Epoch {epoch:2d}/{epochs:2d} | "
                  f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}")

        # Best model tracking
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[TrainCNN] Early stopping triggered after {epoch} epochs (patience={patience}).")
                break

    # Restore best validation weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"[TrainCNN] Restored best model weights with Validation Loss = {best_val_loss:.4f}")

    model.eval()
    return model, history


def evaluate_cnn_model(
    model,
    data_loader,
    target_names: Optional[List[str]] = None,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Evaluates CNN on a data loader and computes classification metrics.

    Args:
        model: Evaluated PyTorch model.
        data_loader: DataLoader (e.g., test_loader).
        target_names: Class labels.
        device: 'cpu' or 'cuda'.

    Returns:
        Tuple: (y_true_array, y_pred_array, metrics_dict)
    """
    import torch
    from src.evaluate import compute_classification_metrics

    model.eval()
    model.to(device)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_b, y_b in data_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y_b.numpy().tolist())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    metrics = compute_classification_metrics(y_true, y_pred, target_names=target_names)
    return y_true, y_pred, metrics

