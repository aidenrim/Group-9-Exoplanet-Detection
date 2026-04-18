#!/usr/bin/env python3
"""
Preprocess Kepler and/or TESS lightcurves into CNN-ready arrays.

Converts raw stitched lightcurves (from scripts/download.py) into fixed-length,
normalized arrays ready for CNN input.  Each candidate produces:

    Global view  (201 points) — full phase-folded lightcurve
    Local view   (61 points) — zoomed transit window

Pipeline per candidate:
    1. Load stitched FITS lightcurve
    2. Remove NaN cadences
    3. Clip upward flux spikes > 5σ (MAD-based)
    4. Detrend with Savitzky-Golay filter
    5. Phase-fold on (period, epoch) from catalog
    6. Bin into global (201) and local (61) views
    7. Subtract 1.0 -> baseline = 0; fill empty bins with 0.0
    8. Save as .npz; append row to data/datasets/manifest.csv

Usage:
    python scripts/preprocess.py                       # Kepler (default)
    python scripts/preprocess.py --mission tess        # TESS only
    python scripts/preprocess.py --mission both        # both missions
    python scripts/preprocess.py --max-stars 50        # Only process first 50 stars
    python scripts/preprocess.py --force               # reprocess already-done
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.preprocess import CATALOG_FILE, ROOT, run_preprocessing, MISSION_KEPLER, MISSION_TESS
from src.data.download import TESS_CATALOG_FILE

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
        description="Preprocess Kepler/TESS lightcurves into CNN-ready arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N unique stars per mission.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess candidates whose .npz output already exists.",
    )
    parser.add_argument(
        "--mission",
        choices=["kepler", "tess", "both"],
        default="kepler",
        help="Which mission to preprocess (default: kepler).",
    )
    args = parser.parse_args()

    do_kepler = args.mission in ("kepler", "both")
    do_tess   = args.mission in ("tess",   "both")

    if do_kepler:
        if not CATALOG_FILE.exists():
            log.error(
                f"Kepler catalog not found: {CATALOG_FILE}\n"
                "Run first:  python scripts/download.py"
            )
            sys.exit(1)
        catalog = pd.read_csv(CATALOG_FILE)
        log.info(f"Kepler catalog loaded: {len(catalog)} KOIs.")
        run_preprocessing(catalog, max_stars=args.max_stars, force=args.force,
                          mission=MISSION_KEPLER)

    if do_tess:
        if not TESS_CATALOG_FILE.exists():
            log.error(
                f"TESS catalog not found: {TESS_CATALOG_FILE}\n"
                "Run first:  python scripts/download.py --mission tess"
            )
            sys.exit(1)
        catalog_t = pd.read_csv(TESS_CATALOG_FILE)
        log.info(f"TESS catalog loaded: {len(catalog_t)} TOIs.")
        run_preprocessing(catalog_t, max_stars=args.max_stars, force=args.force,
                          mission=MISSION_TESS)


if __name__ == "__main__":
    main()
