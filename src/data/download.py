"""
Data acquisition library
========================

Functions for downloading the Kepler and Tess Objects of Interest catalogs and raw lightcurves.

Two-step pipeline:

  Step 1 — Catalog
      Downloads the appropriage cumulative table
      from the NASA Exoplanet Archive via their TAP API.
      Each row is one object of interest; we keep the disposition label,
      orbital period, transit epoch, and transit duration — columns needed
      later to phase-fold the lightcurve.

  Step 2 — Lightcurves
      For each unique star (identified by ID) in the catalog,
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
import logging as _logging
_logging.getLogger("lightkurve").setLevel(_logging.ERROR)
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_DIR = ROOT / "data" / "catalogs"
RAW_DIR = ROOT / "data" / "raw"

KEPLER_CATALOG_FILE = CATALOG_DIR / "koi_cumulative.csv"

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
KEPLER_TAP_QUERY = (
    "select kepid, kepoi_name, koi_disposition, "
    "koi_period, koi_time0bk, koi_duration "
    "from cumulative"
)
KOI_CATALOG_URL = f"{_TAP_BASE}?{urllib.parse.urlencode({'query': KEPLER_TAP_QUERY, 'format': 'csv'})}"

VALID_DISPOSITIONS = {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Catalog download
# ---------------------------------------------------------------------------

def download_catalog(force: bool = False) -> pd.DataFrame:
    """
    Fetch the catalog table from NASA Exoplanet Archive and return a
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

    if KEPLER_CATALOG_FILE.exists() and not force:
        log.info(f"Catalog found on disk — loading {KEPLER_CATALOG_FILE}")
        df = pd.read_csv(KEPLER_CATALOG_FILE)
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

    df.to_csv(KEPLER_CATALOG_FILE, index=False)
    log.info(f"  Saved to {KEPLER_CATALOG_FILE}")
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

    Returns a dict mapping status -> count for final reporting.
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


# ---------------------------------------------------------------------------
# TESS — TOI catalog and lightcurve constants
# ---------------------------------------------------------------------------

TESS_CATALOG_FILE = CATALOG_DIR / "toi_catalog.csv"

TESS_TAP_QUERY = (
    "select tid, toi, tfopwg_disp, pl_orbper, pl_tranmid, pl_trandurh "
    "from toi"
)
TESS_CATALOG_URL = (
    f"{_TAP_BASE}?"
    + urllib.parse.urlencode({"query": TESS_TAP_QUERY, "format": "csv"})
)

# TFOPWG dispositions we keep (CP/KP = confirmed, PC = candidate, FP/FA = false positive).
TESS_VALID_DISPOSITIONS = {"CP", "KP", "PC", "FP", "FA"}


# ---------------------------------------------------------------------------
# TESS — Catalog download
# ---------------------------------------------------------------------------


def download_tess_catalog(force: bool = False) -> pd.DataFrame:
    """
    Fetch the TESS TOI table from NASA Exoplanet Archive and return a
    cleaned DataFrame.

    The catalog is saved with the original NASA column names:
        tid           int     TIC (TESS Input Catalog) star ID
        toi           float   TOI identifier, e.g. 103.01
        tfopwg_disp   str     TFOPWG disposition: CP, KP, PC, FP, or FA
        pl_orbper     float   Orbital period in days
        pl_tranmid    float   Transit midpoint in BJD (NOT BTJD)
        pl_trandurh   float   Transit duration in hours

    The BTJD conversion (subtract 2,457,000) and column renaming happen in
    preprocess.py's _normalize_catalog(), not here.

    The file is cached at data/catalogs/toi_catalog.csv.
    """
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    if TESS_CATALOG_FILE.exists() and not force:
        log.info(f"TESS catalog found on disk — loading {TESS_CATALOG_FILE}")
        df = pd.read_csv(TESS_CATALOG_FILE)
        log.info(f"  {len(df)} TOIs loaded from cache.")
        return df

    log.info("Fetching TESS TOI catalog from NASA Exoplanet Archive TAP ...")
    log.info(f"  URL: {TESS_CATALOG_URL}")

    response = requests.get(
        TESS_CATALOG_URL,
        timeout=120,
        headers={"User-Agent": "exoplanet-cnn/1.0 (student project)"},
    )
    response.raise_for_status()

    clean = "\n".join(
        line for line in response.text.splitlines() if not line.startswith("#")
    )
    df = pd.read_csv(StringIO(clean))
    log.info(f"  Raw rows: {len(df)}")

    # ---- Filter 1: keep only rows with a recognized disposition --------
    df = df[df["tfopwg_disp"].isin(TESS_VALID_DISPOSITIONS)].copy()
    log.info(f"  After disposition filter: {len(df)} rows")

    # ---- Filter 2: drop rows missing any transit parameter we need -----
    before = len(df)
    df = df.dropna(subset=["pl_orbper", "pl_tranmid", "pl_trandurh"])
    dropped = before - len(df)
    if dropped:
        log.info(f"  Dropped {dropped} rows with missing period/epoch/duration")

    # ---- Filter 3: sanity-check parameter values -----------------------
    df = df[(df["pl_orbper"] > 0) & (df["pl_trandurh"] > 0)].copy()

    disposition_counts = df["tfopwg_disp"].value_counts().to_dict()
    log.info(f"  Final catalog: {len(df)} TOIs — {disposition_counts}")

    df.to_csv(TESS_CATALOG_FILE, index=False)
    log.info(f"  Saved to {TESS_CATALOG_FILE}")
    return df


# ---------------------------------------------------------------------------
# TESS — Per-star lightcurve download
# ---------------------------------------------------------------------------


def _tess_fits_path(tic_id: int) -> Path:
    """Return the expected on-disk path for a TIC star's stitched TESS lightcurve."""
    return RAW_DIR / f"tic_{tic_id:010d}.fits"


def _download_one_tess_star(tic_id: int, retries: int = 3) -> tuple[int, str]:
    """
    Download, stitch, and save all available TESS lightcurves for one star.

    Tries 2-minute SPOC cadence first (highest quality); falls back to
    10-minute TESS-SPOC FFI lightcurves if no short-cadence data exists.

    Returns (tic_id, status) with the same status strings as _download_one_star.
    """
    out_path = _tess_fits_path(tic_id)
    if out_path.exists():
        return tic_id, "cached"

    for attempt in range(1, retries + 1):
        try:
            # Primary: 2-minute SPOC cadence.
            search = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS",
                cadence="short",
                author="SPOC",
            )

            # Fallback: 10-minute TESS-SPOC FFI lightcurves.
            if len(search) == 0:
                search = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS",
                    author="TESS-SPOC",
                )

            if len(search) == 0:
                return tic_id, "not_found"

            collection = search.download_all(
                quality_bitmask="default",
                flux_column="pdcsap_flux",
            )

            if collection is None or len(collection) == 0:
                return tic_id, "not_found"

            lc = collection.stitch()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            lc.to_fits(str(out_path), overwrite=True)
            return tic_id, "downloaded"

        except Exception as exc:
            exc_str = str(exc)
            match = re.search(r"(/[^\s]+\.fits)", exc_str)
            if match:
                corrupt = Path(match.group(1))
                if corrupt.exists() and ".lightkurve" in str(corrupt):
                    corrupt.unlink()
                    log.debug(f"TIC {tic_id}: deleted corrupt cache file {corrupt}")

            if attempt < retries:
                wait = 2 ** attempt
                log.debug(
                    f"TIC {tic_id}: attempt {attempt} failed ({exc}), "
                    f"retrying in {wait}s ..."
                )
                time.sleep(wait)
            else:
                return tic_id, f"error:{exc}"

    return tic_id, "error:max retries exceeded"


def download_tess_lightcurves(tic_ids: list[int], workers: int = 4) -> dict[str, int]:
    """
    Download TESS lightcurves for every TIC ID in *tic_ids* using a thread pool.

    Identical structure to download_lightcurves() but dispatches
    _download_one_tess_star.  Kepler and TESS FITS files coexist in data/raw/
    distinguished by their filename prefix (kic_* vs tic_*).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        "downloaded": 0,
        "cached": 0,
        "not_found": 0,
        "error": 0,
    }
    total = len(tic_ids)
    log.info(
        f"Starting TESS lightcurve downloads: {total} unique stars, "
        f"{workers} parallel workers."
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one_tess_star, tid): tid for tid in tic_ids}

        for i, future in enumerate(as_completed(futures), start=1):
            tic_id, status = future.result()

            if status.startswith("error"):
                counts["error"] += 1
                log.warning(f"  [{i:>4}/{total}] TIC {tic_id:>10}: {status}")
            elif status == "not_found":
                counts["not_found"] += 1
                log.warning(f"  [{i:>4}/{total}] TIC {tic_id:>10}: not found on MAST")
            else:
                counts[status] += 1
                if status == "downloaded" or i % 100 == 0:
                    log.info(f"  [{i:>4}/{total}] TIC {tic_id:>10}: {status}")

    return counts
