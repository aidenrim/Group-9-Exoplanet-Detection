"""
Data acquisition library
========================

Functions for downloading the Kepler KOI catalog and raw lightcurves.

Two-step pipeline:

  Step 1 — Catalog
      Downloads the Kepler KOI (Kepler Objects of Interest) cumulative table
      from the NASA Exoplanet Archive via their TAP (Table Access Protocol) API.
      Each row is one planet candidate (KOI); we keep the disposition label,
      orbital period, transit epoch, and transit duration — everything needed
      later to phase-fold the lightcurve.

  Step 2 — Lightcurves
      For each unique Kepler star (identified by KIC ID) in the catalog,
      downloads all available long-cadence (30-min) quarters from MAST using
      the `lightkurve` library, stitches them into a single continuous
      lightcurve, and saves it as a FITS file on disk.

      Downloads are parallelized across a thread pool and are fully resumable
      — already-downloaded files are skipped on subsequent runs.
"""

import logging
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import lightkurve as lk
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_DIR = ROOT / "data" / "catalogs"
RAW_DIR = ROOT / "data" / "raw"

CATALOG_FILE = CATALOG_DIR / "koi_cumulative.csv"

# ---------------------------------------------------------------------------
# NASA Exoplanet Archive — KOI cumulative table via TAP
#
# The TAP (Table Access Protocol) endpoint accepts an ADQL/SQL-like query and
# returns results as plain CSV.  We only fetch the six columns we actually use
# so the download stays small (~500 KB instead of ~10 MB for the full table).
#
# Table reference:
#   https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
# ---------------------------------------------------------------------------

_TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_TAP_QUERY = (
    "select kepid, kepoi_name, koi_disposition, "
    "koi_period, koi_time0bk, koi_duration "
    "from cumulative"
)
KOI_CATALOG_URL = f"{_TAP_BASE}?{urllib.parse.urlencode({'query': _TAP_QUERY, 'format': 'csv'})}"

# Dispositions we keep.  CONFIRMED and CANDIDATE are positive examples;
# FALSE POSITIVE (eclipsing binaries, background stars, etc.) are negatives.
VALID_DISPOSITIONS = {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Catalog download
# ---------------------------------------------------------------------------

def download_catalog(force: bool = False) -> pd.DataFrame:
    """
    Fetch the KOI cumulative table from NASA Exoplanet Archive and return a
    cleaned DataFrame.

    Columns returned:
        kepid           int     Kepler Input Catalog star ID
        kepoi_name      str     KOI identifier, e.g. "K00001.01"
        koi_disposition str     "CONFIRMED", "CANDIDATE", or "FALSE POSITIVE"
        koi_period      float   Orbital period in days
        koi_time0bk     float   Transit epoch in BKJD (BJD − 2 454 833.0)
        koi_duration    float   Transit duration in hours

    The file is cached at data/catalogs/koi_cumulative.csv and reused on
    subsequent runs unless --force-catalog is passed.
    """
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    if CATALOG_FILE.exists() and not force:
        log.info(f"Catalog found on disk — loading {CATALOG_FILE}")
        df = pd.read_csv(CATALOG_FILE)
        log.info(f"  {len(df)} KOIs loaded from cache.")
        return df

    log.info("Fetching KOI catalog from NASA Exoplanet Archive TAP ...")
    log.info(f"  URL: {KOI_CATALOG_URL}")

    response = requests.get(
        KOI_CATALOG_URL,
        timeout=120,
        headers={"User-Agent": "exoplanet-cnn/1.0 (student project)"},
    )
    response.raise_for_status()

    # The TAP endpoint may prepend comment lines starting with '#' — strip them.
    clean = "\n".join(
        line for line in response.text.splitlines() if not line.startswith("#")
    )
    df = pd.read_csv(StringIO(clean))
    log.info(f"  Raw rows: {len(df)}")

    # ---- Filter 1: keep only rows with a disposition we recognize ------------
    # Some rows have disposition = "NOT DISPOSITIONED" (pipeline didn't finish).
    df = df[df["koi_disposition"].isin(VALID_DISPOSITIONS)].copy()
    log.info(f"  After disposition filter: {len(df)} rows")

    # ---- Filter 2: drop rows missing any transit parameter we need -----------
    # Without period, epoch, and duration we cannot phase-fold the lightcurve.
    before = len(df)
    df = df.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
    dropped = before - len(df)
    if dropped:
        log.info(f"  Dropped {dropped} rows with missing period/epoch/duration")

    # ---- Filter 3: sanity-check parameter values ----------------------------
    df = df[(df["koi_period"] > 0) & (df["koi_duration"] > 0)].copy()

    # ---- Summary ------------------------------------------------------------
    disposition_counts = df["koi_disposition"].value_counts().to_dict()
    log.info(f"  Final catalog: {len(df)} KOIs — {disposition_counts}")

    df.to_csv(CATALOG_FILE, index=False)
    log.info(f"  Saved to {CATALOG_FILE}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Per-star lightcurve download
# ---------------------------------------------------------------------------

def _fits_path(kepid: int) -> Path:
    """Return the expected on-disk path for a star's stitched lightcurve."""
    # Zero-pad to 9 digits to match the KIC naming convention.
    return RAW_DIR / f"kic_{kepid:09d}.fits"


def _download_one_star(kepid: int, retries: int = 3) -> tuple[int, str]:
    """
    Download, stitch, and save all long-cadence Kepler quarters for one star.

    Each Kepler quarter is a ~90-day segment.  Most stars have ~17 quarters
    across the 4-year mission.  We:
      1. Search MAST for all available long-cadence products for this KIC ID.
      2. Download each quarter as a LightCurve object (PDCSAP flux preferred).
      3. Stitch all quarters into one continuous LightCurve — lightkurve
         normalizes each quarter to median = 1.0 before concatenating, which
         removes inter-quarter flux offsets caused by different pixel masks.
      4. Write the result to a FITS file.

    PDCSAP_FLUX (Pre-search Data Conditioning SAP Flux) is the pre-processed
    flux column from the Kepler pipeline.  It has long-term instrumental trends
    removed using co-trending basis vectors, making it much cleaner than raw
    SAP flux while preserving transit signals.

    Returns (kepid, status) where status ∈
        'downloaded'  — freshly fetched and saved
        'cached'      — file already existed, nothing to do
        'not_found'   — MAST returned no results for this KIC ID
        'error:<msg>' — an exception occurred (after all retries)
    """
    out_path = _fits_path(kepid)
    if out_path.exists():
        return kepid, "cached"

    for attempt in range(1, retries + 1):
        try:
            # Query MAST for all long-cadence Kepler data for this star.
            # author="Kepler" excludes community-contributed light curves
            # (e.g., K2) that share the same target catalogue.
            search = lk.search_lightcurve(
                f"KIC {kepid}",
                mission="Kepler",
                cadence="long",
                author="Kepler",
            )

            if len(search) == 0:
                return kepid, "not_found"

            # Download all quarters.  quality_bitmask="default" masks cadences
            # flagged for known instrumental problems (attitude tweaks, cosmic
            # rays, safe-mode events, etc.) without being overly aggressive.
            collection = search.download_all(
                quality_bitmask="default",
                flux_column="pdcsap_flux",
            )

            if collection is None or len(collection) == 0:
                return kepid, "not_found"

            # Stitch quarters.  Each quarter is normalized to its own median
            # before concatenation so inter-quarter offsets don't create
            # artificial steps in the combined lightcurve.
            lc = collection.stitch()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            lc.to_fits(str(out_path), overwrite=True)
            return kepid, "downloaded"

        except Exception as exc:
            # lightkurve caches individual quarter FITS files.  If a download
            # was interrupted the truncated file remains in the cache and every
            # subsequent retry reads the same corrupt copy — exponential backoff
            # alone does not help.  When the error message names a specific file
            # in the lightkurve cache, delete it so the next attempt fetches a
            # fresh copy from MAST.
            exc_str = str(exc)
            match = re.search(r"(/[^\s]+\.fits)", exc_str)
            if match:
                corrupt = Path(match.group(1))
                if corrupt.exists() and ".lightkurve" in str(corrupt):
                    corrupt.unlink()
                    log.debug(f"KIC {kepid}: deleted corrupt cache file {corrupt}")

            if attempt < retries:
                wait = 2 ** attempt  # exponential backoff: 2s, 4s
                log.debug(
                    f"KIC {kepid}: attempt {attempt} failed ({exc}), "
                    f"retrying in {wait}s ..."
                )
                time.sleep(wait)
            else:
                return kepid, f"error:{exc}"

    # Should be unreachable, but satisfies the type-checker.
    return kepid, "error:max retries exceeded"


def download_lightcurves(kepids: list[int], workers: int = 4) -> dict[str, int]:
    """
    Download lightcurves for every KIC ID in *kepids* using a thread pool.

    MAST is an HTTP service and most of the wait time is network I/O, so
    threading (not multiprocessing) is the right concurrency model here.
    Keep workers ≤ 8 to stay within MAST's informal rate limits.

    Returns a dict mapping status → count for final reporting.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        "downloaded": 0,
        "cached": 0,
        "not_found": 0,
        "error": 0,
    }
    total = len(kepids)
    log.info(
        f"Starting lightcurve downloads: {total} unique stars, "
        f"{workers} parallel workers."
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one_star, kid): kid for kid in kepids}

        for i, future in enumerate(as_completed(futures), start=1):
            kepid, status = future.result()

            if status.startswith("error"):
                counts["error"] += 1
                log.warning(f"  [{i:>4}/{total}] KIC {kepid:>9}: {status}")
            elif status == "not_found":
                counts["not_found"] += 1
                log.warning(f"  [{i:>4}/{total}] KIC {kepid:>9}: not found on MAST")
            else:
                counts[status] += 1
                # Log every 'downloaded' and every 100th entry to avoid
                # flooding the log while still showing progress.
                if status == "downloaded" or i % 100 == 0:
                    log.info(f"  [{i:>4}/{total}] KIC {kepid:>9}: {status}")

    return counts
