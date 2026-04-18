#!/usr/bin/env python3
"""
Train the ExoplanetCNN model.

Trains the dual-branch 1D CNN on preprocessed KOI lightcurve arrays.
Requires pre-built train/val/test splits in data/datasets/ — run
scripts/build_dataset.py first if they don't exist.

Training loop summary:
  • Loss      : BCEWithLogitsLoss with pos_weight
  • Optimiser : Adam with L2 weight decay
  • Scheduler : ReduceLROnPlateau (halves LR when val AUC stalls)
  • Stopping  : Early stopping on val AUC
  • Checkpoint: best model by val AUC -> results/checkpoints/best_model.pt

Outputs:
  results/checkpoints/best_model.pt
  results/plots/training_curves.png
  results/plots/roc_curve.png
  results/plots/confusion_matrix.png

Usage:
    # Standard training run
    python scripts/train.py --name run_v1 --dataset full_dataset

    # Quick smoke-test (5 epochs)
    python scripts/train.py --name smoke --dataset full_dataset --epochs 5 --batch-size 64

    # Adjust hyperparameters
    python scripts/train.py --name run_v2 --dataset full_dataset --lr 5e-4 --dropout 0.3

    # Resume from a saved checkpoint
    python scripts/train.py --name run_v1 --dataset full_dataset --resume
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DATASETS_DIR
from src.models.train import ROOT, train

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "train.log"),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ExoplanetCNN on preprocessed KOI lightcurves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required identifiers
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        metavar="NAME",
        help="Model name. Checkpoint and plots are saved to results/{name}/.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="NAME",
        help="Dataset name to load splits from (data/datasets/{name}/).",
    )

    # Training hyperparameters
    parser.add_argument("--epochs",       type=int,   default=100,  help="Max training epochs (default: 100)")
    parser.add_argument("--batch-size",   type=int,   default=32,   help="Batch size (default: 32)")
    parser.add_argument("--lr",           type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam L2 weight decay (default: 1e-4)")
    parser.add_argument("--dropout",      type=float, default=0.5,  help="Dropout probability in head (default: 0.5)")

    # Stopping / scheduling
    parser.add_argument("--patience",           type=int, default=15, help="Early stopping patience in epochs (default: 15)")
    parser.add_argument("--scheduler-patience", type=int, default=5,  help="LR reduction patience in epochs (default: 5)")

    # Misc
    parser.add_argument("--workers", type=int,        default=0,     help="DataLoader worker processes (default: 0)")
    parser.add_argument("--resume",  action="store_true",             help="Resume training from results/checkpoints/best_model.pt")

    args = parser.parse_args()

    dataset_dir = DATASETS_DIR / args.dataset
    if not (dataset_dir / "train.csv").exists():
        log.error(
            f"Dataset splits not found in {dataset_dir}\n"
            f"Run scripts/build_dataset.py --name {args.dataset} first."
        )
        sys.exit(1)

    train(
        model_name=args.name,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
