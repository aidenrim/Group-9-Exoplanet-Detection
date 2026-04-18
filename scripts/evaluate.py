"""
Evaluate a trained ExoplanetCNN on the test split of a named dataset.

Loads the best checkpoint for the specified model, runs inference on the
test split, and writes accuracy, precision, recall, F1, and AUC-ROC to a
JSON file in the model's results sub-folder.

The decision threshold is taken from the checkpoint (the F1-optimal value
found on the validation set during training).  If the checkpoint pre-dates
threshold saving, it is recomputed from the validation split.

Output:
    results/{model}/eval_{dataset}.json

Usage:
    python scripts/evaluate.py --model run_v1 --dataset full_dataset
    python scripts/evaluate.py --model run_v1 --dataset kepler_only --batch-size 64
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dataset import DATASETS_DIR, ExoplanetDataset, load_splits
from src.models.model import ExoplanetCNN
from src.models.train import evaluate, find_best_threshold

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ExoplanetCNN on a named dataset's test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="NAME",
        help="Model name (checkpoint loaded from results/{name}/best_model.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="NAME",
        help="Dataset name (splits loaded from data/datasets/{name}/).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size for inference (default: 32).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Locate checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = ROOT / "results" / args.model / "best_model.pt"
    if not checkpoint_path.exists():
        log.error(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run scripts/train.py --name {args.model} --dataset <dataset> first."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Locate dataset splits
    # ------------------------------------------------------------------
    dataset_dir = DATASETS_DIR / args.dataset
    try:
        train_df, val_df, test_df = load_splits(dataset_dir)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(f"Dataset  : {args.dataset}  ({len(test_df)} test candidates)")

    # ------------------------------------------------------------------
    # 3. Build DataLoaders (test + val for threshold recovery)
    # ------------------------------------------------------------------
    val_loader  = torch.utils.data.DataLoader(
        ExoplanetDataset(val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        ExoplanetDataset(test_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 4. Load model and threshold from checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = ExoplanetCNN()
    model.load_state_dict(ckpt["model_state"])

    # Threshold was saved into the checkpoint by train.py after the val-set
    # F1 search.  If it's absent (old checkpoint), recompute it now.
    if "threshold" in ckpt:
        threshold = float(ckpt["threshold"])
        log.info(f"Threshold: {threshold:.2f}  (loaded from checkpoint)")
    else:
        log.info("Threshold not in checkpoint — recomputing from val split …")
        criterion = nn.BCEWithLogitsLoss()
        val_metrics = evaluate(model, val_loader, criterion, torch.device("cpu"))
        threshold, val_f1 = find_best_threshold(val_metrics["labels"], val_metrics["probs"])
        log.info(f"Threshold: {threshold:.2f}  (val F1 = {val_f1:.4f})")

    epoch       = ckpt.get("epoch", "?")
    best_val_auc = ckpt.get("best_val_auc", float("nan"))
    log.info(f"Model    : {args.model}  (epoch {epoch}, val AUC {best_val_auc:.4f})")

    # ------------------------------------------------------------------
    # 5. Inference on test split
    # ------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_loader, criterion, torch.device("cpu"))

    labels = test_metrics["labels"]
    probs  = test_metrics["probs"]
    preds  = (probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # 6. Compute metrics
    # ------------------------------------------------------------------
    accuracy  = float(accuracy_score(labels, preds))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall    = float(recall_score(labels, preds, zero_division=0))
    f1        = float(f1_score(labels, preds, zero_division=0))
    auc       = float(test_metrics["auc"])

    results = {
        "model":     args.model,
        "dataset":   args.dataset,
        "epoch":     epoch,
        "threshold": round(threshold, 4),
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "auc":       round(auc,       4),
    }

    # ------------------------------------------------------------------
    # 7. Log and save
    # ------------------------------------------------------------------
    log.info("=" * 52)
    log.info("TEST SET RESULTS")
    log.info(f"  Accuracy  : {accuracy:.4f}")
    log.info(f"  Precision : {precision:.4f}")
    log.info(f"  Recall    : {recall:.4f}")
    log.info(f"  F1        : {f1:.4f}")
    log.info(f"  AUC-ROC   : {auc:.4f}")
    log.info("=" * 52)

    out_path = ROOT / "results" / args.model / "eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
