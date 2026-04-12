#!/usr/bin/env python3
"""
Preprocess Kepler lightcurves into CNN-ready arrays.

Converts raw stitched lightcurves (from scripts/download.py) into fixed-length,
normalized arrays ready for CNN input.  Each KOI produces:

    Global view  (201 points) — full phase-folded lightcurve
    Local view   ( 61 points) — zoomed transit window

Pipeline per KOI:
    1. Load stitched FITS lightcurve
    2. Remove NaN cadences
    3. Clip upward flux spikes > 5σ (MAD-based)
    4. Detrend with 301-cadence Savitzky-Golay filter (~6.3 days)
    5. Phase-fold on (period, epoch) from catalog
    6. Bin into global (201) and local (61) views
    7. Subtract 1.0 → baseline = 0; fill empty bins with 0.0
    8. Save as .npz; append row to data/datasets/manifest.csv

Usage:
    python scripts/preprocess.py                   # process all KOIs
    python scripts/preprocess.py --max-stars 50    # smoke-test on 50 stars
    python scripts/preprocess.py --force           # reprocess already-done KOIs
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import CATALOG_FILE, ROOT, run_preprocessing

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "preprocess.log"),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess KOI lightcurves into CNN-ready arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N unique stars (for smoke-testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess KOIs whose .npz output already exists.",
    )
    args = parser.parse_args()

    if not CATALOG_FILE.exists():
        log.error(
            f"Catalog not found: {CATALOG_FILE}\n"
            "Run Phase 1 first:  python scripts/download.py"
        )
        sys.exit(1)

    catalog = pd.read_csv(CATALOG_FILE)
    log.info(f"Catalog loaded: {len(catalog)} KOIs.")

    run_preprocessing(catalog, max_stars=args.max_stars, force=args.force)


if __name__ == "__main__":
    main()
