#!/usr/bin/env python3
"""
Download Kepler KOI catalog and lightcurves.

Two-step pipeline:

  Step 1 — Catalog
      Downloads the Kepler KOI cumulative table from the NASA Exoplanet Archive
      via TAP API → data/catalogs/koi_cumulative.csv

  Step 2 — Lightcurves
      For each unique Kepler star in the catalog, downloads all long-cadence
      quarters from MAST, stitches them, and saves as data/raw/kic_XXXXXXXXX.fits

      Downloads are parallelized and fully resumable — already-downloaded
      files are skipped on subsequent runs.

Usage:
    # Download everything (~4 000 stars, takes several hours)
    python scripts/download.py

    # Quick smoke-test: download only the first 20 stars
    python scripts/download.py --max-stars 20

    # Tune parallelism (default 4 threads; MAST dislikes more than ~8)
    python scripts/download.py --workers 6

    # Force a fresh catalog fetch even if the file already exists
    python scripts/download.py --force-catalog
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download import ROOT, download_catalog, download_lightcurves

# ---------------------------------------------------------------------------
# Logging — writes to stdout AND a log file in the project root
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "download.log"),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Kepler KOI catalog and lightcurves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=None,
        metavar="N",
        help="Only download the first N unique stars (useful for smoke-testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel download threads (default: 4, max recommended: 8).",
    )
    parser.add_argument(
        "--force-catalog",
        action="store_true",
        help="Re-download the KOI catalog even if data/catalogs/koi_cumulative.csv exists.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Labeled catalog
    # ------------------------------------------------------------------
    df = download_catalog(force=args.force_catalog)

    # ------------------------------------------------------------------
    # Step 2: Lightcurves (one FITS file per unique star)
    # ------------------------------------------------------------------
    # Multiple KOIs can share the same star (multi-planet systems).
    # We download one lightcurve per star and cross-reference by KIC ID
    # during preprocessing — no need to download the same star twice.
    from src.data.download import RAW_DIR
    kepids: list[int] = df["kepid"].dropna().astype(int).unique().tolist()

    if args.max_stars is not None:
        kepids = kepids[: args.max_stars]
        log.info(f"--max-stars {args.max_stars}: limiting to {len(kepids)} stars.")

    summary = download_lightcurves(kepids, workers=args.workers)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    fits_on_disk = sum(1 for _ in RAW_DIR.glob("kic_*.fits"))
    log.info("=" * 60)
    log.info("Download complete.")
    log.info(f"  Catalog KOIs  : {len(df)}")
    log.info(f"  Unique stars  : {len(kepids)}")
    for status, count in summary.items():
        log.info(f"  {status:<14}: {count}")
    log.info(f"  FITS on disk  : {fits_on_disk}  ({RAW_DIR})")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
