"""
Training library
================

Trains the dual-branch ExoplanetCNN on preprocessed KOI lightcurve arrays,
validates after every epoch, checkpoints the best model, and produces a full
evaluation report on the held-out test set.

Training loop summary
---------------------
  • Loss      : BCEWithLogitsLoss with pos_weight (handles mild class imbalance)
  • Optimiser : Adam with L2 weight decay
  • Scheduler : ReduceLROnPlateau — halves the LR when val AUC stops improving
  • Stopping  : EarlyStopping on val AUC with configurable patience
  • Checkpoint: saves the epoch with the best val AUC to results/checkpoints/

Test-set evaluation (after training completes)
----------------------------------------------
  • AUC-ROC  — primary metric; threshold-independent ranking quality
  • Precision, Recall, F1  — at the threshold that maximises val-set F1
  • Confusion matrix
  • Training-curve plot  (loss + AUC vs epoch)
  • ROC curve plot
  • Confusion-matrix heat-map
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend — safe on any machine
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sibling module imports
# ---------------------------------------------------------------------------
# Scripts add the project root to sys.path before importing this module,
# so absolute src.* imports resolve correctly.

from src.data.dataset import load_splits, make_loaders
from src.models.model import ExoplanetCNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training set.

    Returns:
        mean training loss over all batches (weighted by batch size).
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for global_view, local_view, labels in pbar:
        global_view = global_view.to(device, non_blocking=True)
        local_view  = local_view.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        logits = model(global_view, local_view).squeeze(1)   # (B,)
        loss   = criterion(logits, labels)

        loss.backward()
        # Clamp gradients to prevent weights from growing large enough to
        # produce extreme logits, which cause non-finite sigmoid outputs on
        # the MPS backend.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        batch_loss = loss.item() * len(labels)
        total_loss += batch_loss
        n_samples  += len(labels)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_samples


# ---------------------------------------------------------------------------
# Evaluation pass
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run inference over *loader* with no gradient computation.

    Returns a dict with keys:
        loss     float   mean BCE loss (no pos_weight applied)
        auc      float   AUC-ROC over the full split
        labels   ndarray ground-truth labels
        probs    ndarray predicted probabilities in [0, 1]
    """
    # Run evaluation on CPU regardless of training device.
    #
    # The MPS backend has numerical instability during inference: BCEWithLogitsLoss
    # produces NaN/±inf, and sigmoid can overflow to NaN for large-but-finite
    # logits.  These corrupt val loss, val AUC, and therefore checkpoint selection,
    # LR scheduling, and early stopping — all the signals that guide training.
    # Symptom-level fixes (clamping, NaN sanitisation) don't reach the root cause.
    #
    # CPU arithmetic is always numerically stable.  Moving the model costs ~0.3 s
    # per eval call; on a 4-second epoch that's <10 % overhead and is worth it for
    # reliable, deterministic metrics.
    eval_device = torch.device("cpu") if device.type == "mps" else device
    model.to(eval_device)
    model.eval()
    total_loss = 0.0
    n_samples  = 0
    all_labels: list[np.ndarray] = []
    all_probs:  list[np.ndarray] = []

    with torch.no_grad():
        for global_view, local_view, labels in loader:
            global_view = global_view.to(eval_device)
            local_view  = local_view.to(eval_device)
            labels      = labels.to(eval_device)

            logits = model(global_view, local_view).squeeze(1)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            n_samples  += len(labels)

            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Restore model to training device before returning.
    if eval_device != device:
        model.to(device)

    labels_arr = np.concatenate(all_labels)
    probs_arr  = np.concatenate(all_probs)

    # Safety net: should never trigger on CPU, but keeps the pipeline robust.
    probs_arr = np.where(np.isfinite(probs_arr), probs_arr, 0.5)

    # Convert float32 labels (0.0 / 1.0) to binary int.
    # We use np.where rather than astype(int) because the MPS backend can produce
    # signaling-NaN bit patterns that np.isnan() misses and nan_to_num doesn't
    # catch; those survive to astype(int) and overflow to INT64_MIN (-2^63).
    # A boolean comparison is safe for *any* NaN variant: IEEE 754 guarantees
    # that NaN comparisons always return False, so NaN -> 0 with no cast involved.
    labels_int = np.where(labels_arr >= 0.5, 1, 0)
    try:
        auc = roc_auc_score(labels_int, probs_arr)
    except ValueError:
        auc = 0.5
    if np.isnan(auc):
        auc = 0.5

    return {
        "loss":   total_loss / n_samples,
        "auc":    auc,
        "labels": labels_int,
        "probs":  probs_arr,
    }


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """
    Find the decision threshold that maximises F1 on *labels* / *probs*.

    We scan 89 candidate thresholds in [0.1, 0.9] and return the one that
    gives the highest F1.  This is done on the *validation* set, then the
    chosen threshold is applied to the test set — never tune the threshold
    on the test set itself.

    Returns:
        (best_threshold, best_f1)
    """
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0

    thresholds = np.linspace(0.1, 0.9, 89)
    best_f1, best_t = 0.0, 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return best_t, best_f1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_losses: list[float],
    val_losses:   list[float],
    val_aucs:     list[float],
    save_path:    Path,
) -> None:
    """Two-panel plot: BCE loss and AUC-ROC vs epoch."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="train loss")
    ax1.plot(epochs, val_losses,   label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_aucs, color="tab:green", label="val AUC-ROC")
    ax2.axhline(max(val_aucs), color="tab:green", linestyle="--", alpha=0.5,
                label=f"best = {max(val_aucs):.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Validation AUC-ROC")
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved training curves -> {save_path}")


def plot_roc_curve(
    labels:    np.ndarray,
    probs:     np.ndarray,
    auc:       float,
    save_path: Path,
) -> None:
    """ROC curve with AUC annotation.  No-ops gracefully on one-class splits."""
    if len(np.unique(labels)) < 2:
        log.warning("Skipping ROC curve plot — test set contains only one class.")
        return
    if not np.all(np.isfinite(probs)):
        log.warning("Skipping ROC curve plot — probabilities contain non-finite values.")
        return
    fpr, tpr, _ = roc_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="random (0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved ROC curve -> {save_path}")


def plot_confusion_matrix(
    labels:    np.ndarray,
    preds:     np.ndarray,
    save_path: Path,
) -> None:
    """Colour-coded confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["False Positive", "Planet"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved confusion matrix -> {save_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(
    model_name: str,
    dataset_name: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    patience: int = 15,
    scheduler_patience: int = 5,
    workers: int = 0,
    resume: bool = False,
) -> None:
    """
    Train the ExoplanetCNN on pre-built train/val/test splits.

    Args:
        model_name:         Output directory name under results/; checkpoint and
                            plots are saved to results/{model_name}/.
        dataset_name:       Name of the pre-built dataset to load splits from;
                            reads from data/datasets/{dataset_name}/.
        epochs:             Maximum number of training epochs.
        batch_size:         Mini-batch size.
        lr:                 Initial Adam learning rate.
        weight_decay:       Adam L2 regularisation coefficient.
        dropout:            Dropout probability in the classifier head.
        patience:           Early stopping: epochs without val AUC improvement.
        scheduler_patience: ReduceLROnPlateau: epochs before halving the LR.
        workers:            DataLoader worker processes (0 = main process).
        resume:             If True, resume from results/{model_name}/best_model.pt.
    """
    device = get_device()
    log.info(f"Device: {device}")

    output_dir = ROOT / "results" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = ROOT / "data" / "datasets" / dataset_name

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    log.info(f"Loading train/val/test splits from {dataset_dir} ...")
    train_df, val_df, test_df = load_splits(datasets_dir=dataset_dir)

    # pin_memory is a CUDA-only optimisation; MPS and CPU don't support it.
    pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    n_train_pos = int(train_df["label"].sum())
    n_train_neg = len(train_df) - n_train_pos
    log.info(
        f"  Train: {len(train_df)} KOIs  ({n_train_pos} positive / {n_train_neg} negative)\n"
        f"  Val:   {len(val_df)}   KOIs\n"
        f"  Test:  {len(test_df)}  KOIs"
    )

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = ExoplanetCNN(dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: ExoplanetCNN  ({n_params:,} trainable parameters)")

    # ------------------------------------------------------------------
    # 3. Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    pos_weight = torch.tensor([n_train_neg / n_train_pos], device=device)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion  = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=0.5,
        patience=scheduler_patience,
    )

    # ------------------------------------------------------------------
    # 4. Optional: resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_auc = 0.0
    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_aucs:     list[float] = []

    checkpoint_path = output_dir / "best_model.pt"

    if resume and checkpoint_path.exists():
        log.info(f"Resuming from {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimiser_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_auc  = ckpt["best_val_auc"]
        train_losses  = ckpt.get("train_losses", [])
        val_losses    = ckpt.get("val_losses",   [])
        val_aucs      = ckpt.get("val_aucs",     [])
        log.info(f"  Resumed at epoch {start_epoch}, best val AUC = {best_val_auc:.4f}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    epochs_since_improvement = 0

    for epoch in range(start_epoch, epochs):
        log.info(f"Epoch {epoch + 1}/{epochs}  (lr={optimiser.param_groups[0]['lr']:.2e})")

        train_loss = train_epoch(model, train_loader, optimiser, train_criterion, device)

        val_metrics = evaluate(model, val_loader, eval_criterion, device)
        val_loss = val_metrics["loss"]
        val_auc  = val_metrics["auc"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        log.info(
            f"  train loss={train_loss:.4f}  "
            f"val loss={val_loss:.4f}  val AUC={val_auc:.4f}"
        )

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_since_improvement = 0

            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "best_val_auc":    best_val_auc,
                    "train_losses":    train_losses,
                    "val_losses":      val_losses,
                    "val_aucs":        val_aucs,
                },
                checkpoint_path,
            )
            log.info(f"  ✓ New best val AUC={best_val_auc:.4f}  — checkpoint saved.")
        else:
            epochs_since_improvement += 1
            log.info(
                f"  No improvement for {epochs_since_improvement}/{patience} epochs."
            )

        if epochs_since_improvement >= patience:
            log.info(
                f"Early stopping: val AUC has not improved for {patience} epochs."
            )
            break

    # ------------------------------------------------------------------
    # 6. Plots — training curves
    # ------------------------------------------------------------------
    if train_losses:
        plot_training_curves(
            train_losses, val_losses, val_aucs,
            output_dir / "training_curves.png",
        )

    # ------------------------------------------------------------------
    # 7. Test-set evaluation
    # ------------------------------------------------------------------
    log.info("Loading best checkpoint for test evaluation ...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    val_metrics = evaluate(model, val_loader, eval_criterion, device)
    best_thresh, val_f1 = find_best_threshold(val_metrics["labels"], val_metrics["probs"])
    log.info(
        f"Optimal threshold (from val set): {best_thresh:.2f}  "
        f"(val F1 = {val_f1:.4f})"
    )

    # Persist the threshold into the checkpoint so scripts/predict.py can
    # load it without re-running the full validation set.
    ckpt["threshold"] = best_thresh
    torch.save(ckpt, checkpoint_path)

    test_metrics = evaluate(model, test_loader, eval_criterion, device)
    test_labels = test_metrics["labels"]
    test_probs  = test_metrics["probs"]
    test_preds  = (test_probs >= best_thresh).astype(int)

    log.info("=" * 60)
    log.info("TEST SET RESULTS")
    log.info(f"  AUC-ROC  : {test_metrics['auc']:.4f}")
    log.info(f"  Threshold: {best_thresh:.2f}")
    log.info("\n" + classification_report(
        test_labels, test_preds,
        labels=[0, 1],
        target_names=["False Positive", "Planet"],
        digits=4,
    ))
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 8. Test-set plots
    # ------------------------------------------------------------------
    plot_roc_curve(
        test_labels, test_probs, test_metrics["auc"],
        output_dir / "roc_curve.png",
    )
    plot_confusion_matrix(
        test_labels, test_preds,
        output_dir / "confusion_matrix.png",
    )

    log.info(f"All outputs written to {output_dir}")
