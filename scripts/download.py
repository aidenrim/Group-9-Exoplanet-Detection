#!/usr/bin/env python3
"""
Download KOI/TOI catalogs and lightcurves for Kepler and/or TESS.

Two-step pipeline per mission:

  Step 1 — Catalog
      Kepler: Downloads the KOI cumulative table -> data/catalogs/koi_cumulative.csv
      TESS:   Downloads the TOI table            -> data/catalogs/toi_catalog.csv

  Step 2 — Lightcurves
      Kepler: kic_XXXXXXXXX.fits  (long-cadence, one per KIC star)
      TESS:   tic_XXXXXXXXXX.fits (2-min SPOC or 10-min FFI, one per TIC star)

      Both land in data/raw/ and are distinguished by filename prefix.
      Downloads are parallelized and fully resumable.

Usage:
    # Kepler only (default)
    python scripts/download.py

    # TESS only
    python scripts/download.py --mission tess

    # Both missions
    python scripts/download.py --mission both

    # Smoke-test: first 20 stars per mission
    python scripts/download.py --mission both --max-stars 20

    # Force a fresh catalog fetch
    python scripts/download.py --force-catalog
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download import (
    ROOT, RAW_DIR,
    download_catalog, download_lightcurves,
    download_tess_catalog, download_tess_lightcurves,
)

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
        description="Download KOI/TOI catalogs and lightcurves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=None,
        metavar="N",
        help="Only download the first N unique stars per mission (smoke-testing).",
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
        help="Re-download the catalog(s) even if the file(s) already exist.",
    )
    parser.add_argument(
        "--mission",
        choices=["kepler", "tess", "both"],
        default="kepler",
        help="Which mission to download (default: kepler).",
    )
    args = parser.parse_args()

    do_kepler = args.mission in ("kepler", "both")
    do_tess   = args.mission in ("tess",   "both")

    # ------------------------------------------------------------------
    # Kepler
    # ------------------------------------------------------------------
    if do_kepler:
        df_koi = download_catalog(force=args.force_catalog)
        kepids: list[int] = df_koi["kepid"].dropna().astype(int).unique().tolist()
        if args.max_stars is not None:
            kepids = kepids[:args.max_stars]
            log.info(f"--max-stars {args.max_stars}: limiting Kepler to {len(kepids)} stars.")
        summary_k = download_lightcurves(kepids, workers=args.workers)
        fits_k = sum(1 for _ in RAW_DIR.glob("kic_*.fits"))
        log.info("=" * 60)
        log.info("Kepler download complete.")
        log.info(f"  Catalog KOIs  : {len(df_koi)}")
        log.info(f"  Unique stars  : {len(kepids)}")
        for status, count in summary_k.items():
            log.info(f"  kepler {status:<14}: {count}")
        log.info(f"  FITS on disk  : {fits_k}  ({RAW_DIR})")
        log.info("=" * 60)

    # ------------------------------------------------------------------
    # TESS
    # ------------------------------------------------------------------
    if do_tess:
        df_toi = download_tess_catalog(force=args.force_catalog)
        tic_ids: list[int] = df_toi["tid"].dropna().astype(int).unique().tolist()
        if args.max_stars is not None:
            tic_ids = tic_ids[:args.max_stars]
            log.info(f"--max-stars {args.max_stars}: limiting TESS to {len(tic_ids)} stars.")
        summary_t = download_tess_lightcurves(tic_ids, workers=args.workers)
        fits_t = sum(1 for _ in RAW_DIR.glob("tic_*.fits"))
        log.info("=" * 60)
        log.info("TESS download complete.")
        log.info(f"  Catalog TOIs  : {len(df_toi)}")
        log.info(f"  Unique stars  : {len(tic_ids)}")
        for status, count in summary_t.items():
            log.info(f"  tess   {status:<14}: {count}")
        log.info(f"  FITS on disk  : {fits_t}  ({RAW_DIR})")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
