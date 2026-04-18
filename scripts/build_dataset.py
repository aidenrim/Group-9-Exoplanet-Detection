#!/usr/bin/env python3
"""
Build and save train / val / test dataset splits.

Reads data/datasets/manifest.csv (written by scripts/preprocess.py), performs
a star-stratified split into train / val / test partitions, and saves the
resulting DataFrames as CSV files in data/datasets/{name}/.

The split is done BY STAR (id column) to prevent data leakage from multi-planet
systems sharing the same underlying lightcurve.

Use --mission to restrict splits to one mission:
    --mission kepler   Kepler KOIs only
    --mission tess     TESS TOIs only
    --mission both     All candidates (default)

Output files:
    data/datasets/{name}/train.csv
    data/datasets/{name}/val.csv
    data/datasets/{name}/test.csv

These files are loaded by scripts/train.py — run this script once before
training, or any time you want to rebuild the splits with different fractions
or a different random seed.

Usage:
    python scripts/build_dataset.py --name full_dataset
    python scripts/build_dataset.py --name kepler_only --mission kepler
    python scripts/build_dataset.py --name tess_v1 --mission tess
    python scripts/build_dataset.py --name small_test --val-frac 0.10 --test-frac 0.10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
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
    parser.add_argument(
        "--mission",
        choices=["kepler", "tess", "both"],
        default="both",
        help="Filter manifest to one mission before splitting (default: both).",
    )
    parser.add_argument(
        "--candidates",
        choices=["include", "exclude"],
        default="include",
        help=(
            "How to handle CANDIDATE dispositions (label=-1 in manifest). "
            "'include' remaps them to label 1, treating them as confirmed planets. "
            "'exclude' drops them entirely (default: include)."
        ),
    )
    args = parser.parse_args()

    if not MANIFEST_FILE.exists():
        log.error(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run scripts/preprocess.py first to generate the manifest."
        )
        sys.exit(1)

    named_dir = DATASETS_DIR / args.name

    manifest = pd.read_csv(MANIFEST_FILE)

    if "mission" in manifest.columns and args.mission != "both":
        target = args.mission.upper()
        manifest = manifest[manifest["mission"] == target].reset_index(drop=True)
        log.info(f"  Filtered to {target}: {len(manifest)} rows")

    # Migrate old manifests where CANDIDATEs still carry label=1 (written before
    # the -1 change).  Detection uses the disposition column which is always present.
    if "disposition" in manifest.columns:
        old_candidates = (manifest["disposition"] == "CANDIDATE") & (manifest["label"] == 1)
        if old_candidates.any():
            manifest = manifest.copy()
            manifest.loc[old_candidates, "label"] = -1
            log.info(f"  Migrated {old_candidates.sum()} CANDIDATE rows from label=1 to label=-1")

    # Apply --candidates choice before splitting.
    n_candidates = int((manifest["label"] == -1).sum())
    if args.candidates == "exclude":
        manifest = manifest[manifest["label"] != -1].reset_index(drop=True)
        log.info(f"  --candidates exclude: removed {n_candidates} candidates, {len(manifest)} rows remain")
    else:  # include
        manifest = manifest.copy()
        manifest.loc[manifest["label"] == -1, "label"] = 1
        log.info(f"  --candidates include: remapped {n_candidates} candidates to label 1")

    log.info(f"Building splits from {MANIFEST_FILE} ...")
    log.info(
        f"  Name — {args.name}  |  mission: {args.mission}  |  candidates: {args.candidates}  |  "
        f"train: {1 - args.val_frac - args.test_frac:.2f}  "
        f"val: {args.val_frac}  test: {args.test_frac}  "
        f"(random_state={args.random_state})"
    )

    train_df, val_df, test_df = make_splits(
        manifest_df=manifest,
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
