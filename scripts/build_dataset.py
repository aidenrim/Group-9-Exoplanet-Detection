#!/usr/bin/env python3
"""
Build and save train / val / test dataset splits.

Reads data/datasets/manifest.csv (written by scripts/preprocess.py), performs
a star-stratified split into train / val / test partitions, and saves the
resulting DataFrames as CSV files in data/datasets/.

The split is done BY STAR (kepid) to prevent data leakage from multi-planet
systems sharing the same underlying lightcurve.

Output files:
    data/datasets/train.csv
    data/datasets/val.csv
    data/datasets/test.csv

These files are loaded by scripts/train.py — run this script once before
training, or any time you want to rebuild the splits with different fractions
or a different random seed.

Usage:
    python scripts/build_dataset.py --name full_dataset
    python scripts/build_dataset.py --name small_test --val-frac 0.10 --test-frac 0.10
    python scripts/build_dataset.py --name full_dataset --random-state 123
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import DATASETS_DIR, MANIFEST_FILE, ROOT, make_splits, save_splits

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "build_dataset.log"),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and save train/val/test dataset splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        metavar="NAME",
        help="Dataset name. Splits are saved to data/datasets/{name}/.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of data to reserve for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of data to reserve for testing (default: 0.15).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    args = parser.parse_args()

    if not MANIFEST_FILE.exists():
        log.error(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run scripts/preprocess.py first to generate the manifest."
        )
        sys.exit(1)

    named_dir = DATASETS_DIR / args.name

    log.info(f"Building splits from {MANIFEST_FILE} ...")
    log.info(
        f"  Name — {args.name}  |  "
        f"train: {1 - args.val_frac - args.test_frac:.2f}  "
        f"val: {args.val_frac}  test: {args.test_frac}  "
        f"(random_state={args.random_state})"
    )

    train_df, val_df, test_df = make_splits(
        manifest_path=MANIFEST_FILE,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
    )

    save_splits(train_df, val_df, test_df, datasets_dir=named_dir)

    n_train_pos = int(train_df["label"].sum())
    n_train_neg = len(train_df) - n_train_pos
    log.info("=" * 60)
    log.info("Split complete.")
    log.info(f"  Train : {len(train_df):>5}  KOIs  ({n_train_pos} pos / {n_train_neg} neg)")
    log.info(f"  Val   : {len(val_df):>5}  KOIs")
    log.info(f"  Test  : {len(test_df):>5}  KOIs")
    log.info(f"  Saved to {named_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
