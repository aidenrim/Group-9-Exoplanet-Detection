"""
Preprocessing library
=====================

Converts raw stitched lightcurves (Phase 1) into fixed-length, normalized
arrays ready for CNN input.

Each KOI produces two 1-D arrays:

    Global view  (201 points)
        The full phase-folded lightcurve binned uniformly across [-0.5, 0.5].
        Gives the network overall context — out-of-transit noise floor, any
        secondary eclipse near phase ±0.5 (a red-flag for eclipsing binaries),
        and ellipsoidal brightness variations.

    Local view   ( 61 points)
        A zoomed window centred on the transit (phase 0), spanning ±2 transit
        durations in phase space.  Gives fine morphology — a flat-bottomed
        U-shape is planetary; a pointed V-shape suggests an eclipsing binary
        at grazing incidence.

Pipeline per KOI
----------------
    1.  Load stitched Kepler lightcurve from FITS (written by download.py)
    2.  Remove NaN cadences (gaps between quarters leave NaN padding)
    3.  Clip upward flux spikes > 5σ using MAD-based sigma estimate
        — preserves downward transit dips, removes cosmic-ray hits
    4.  Detrend with a 301-cadence Savitzky-Golay filter (≈6.3 days at
        30-min Kepler cadence) to divide out slow stellar variability
    5.  Phase-fold on the KOI catalog period and epoch, stacking all
        transits to boost signal-to-noise
    6.  Bin into global view  (201 bins, range [-0.5,  0.5] in phase)
    7.  Bin into local  view  ( 61 bins, range [-2T, +2T] in phase,
        where T = transit_duration_hours / (24 × period_days))
    8.  Subtract 1.0 so baseline ≈ 0 and transit depth is negative
    9.  Fill empty bins with 0.0 (out-of-transit baseline assumption)
    10. Save as compressed .npz; append a row to manifest.csv
"""

import logging
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_FILE = ROOT / "data" / "catalogs" / "koi_cumulative.csv"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_FILE = ROOT / "data" / "datasets" / "manifest.csv"

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Output array lengths — these must match the CNN input layer sizes in Phase 3.
GLOBAL_BINS = 201
LOCAL_BINS = 61

# The local view spans ± (MULTIPLIER × transit_duration / period) in phase.
LOCAL_HALF_WIDTH_MULTIPLIER = 2.0
# Hard bounds so no KOI gets an absurdly narrow or absurdly wide local view.
MIN_LOCAL_HALF_WIDTH = 0.01   # ≥ 1 % of the period on each side
MAX_LOCAL_HALF_WIDTH = 0.25   # ≤ 25 % of the period on each side

# Savitzky-Golay window for stellar-variability removal.
# Long-cadence Kepler = 30-min sampling → 48 cadences / day.
# 301 cadences ≈ 6.3 days — wide enough to track typical stellar rotation
# (> 10 days for most Kepler targets) without distorting the transit shape
# (longest KOI transit duration ≈ 15 hours ≈ 30 cadences, well inside one bin).
SG_WINDOW = 301

# Only clip upward outliers.  5σ is aggressive enough to catch cosmic rays
# while being conservative enough not to clip real astrophysical variability.
SIGMA_UPPER = 5.0

# Discard lightcurves with fewer than this many cadences after cleaning.
# 500 cadences ≈ 10 days — the absolute minimum for a meaningful phase fold.
MIN_CADENCES = 500

# Binary classification targets.
# CANDIDATE is treated as positive: the Kepler pipeline vetted these as
# astrophysically plausible; most will eventually be confirmed.
LABEL_MAP: dict[str, int] = {
    "CONFIRMED": 1,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 0,
}

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _fits_path(kepid: int) -> Path:
    """Path to the raw stitched lightcurve written by download.py."""
    return RAW_DIR / f"kic_{kepid:09d}.fits"


def _npz_path(kepoi_name: str) -> Path:
    """
    Path for the processed output of one KOI.

    Dots in KOI names (e.g. 'K00001.01') are replaced with underscores
    so the filename is safe on all filesystems.
    """
    return PROCESSED_DIR / f"{kepoi_name.replace('.', '_')}.npz"


# ---------------------------------------------------------------------------
# Step A: Clean and detrend (once per star)
# ---------------------------------------------------------------------------


def _clean_and_detrend(lc: lk.LightCurve) -> lk.LightCurve | None:
    """
    Prepare a raw stitched lightcurve for phase-folding.

    Returns a flattened LightCurve with baseline ≈ 1.0, or None if the
    lightcurve is too short or degenerate to process.

    Outlier clipping
    ----------------
    We clip cadences whose flux exceeds the median + 5 × MAD-sigma.
    MAD (Median Absolute Deviation) is far more robust than standard
    deviation here because:
      • transit dips are often many sigma below the median — using std would
        inflate the noise estimate and make the threshold too permissive.
      • MAD is computed on the sorted residuals; a handful of large spikes
        do not pull it up the way they do with std.

    Savitzky-Golay detrending
    -------------------------
    Stars exhibit brightness variations on timescales of days to weeks due to
    rotation (star spots crossing the disk) and convective granulation.  Left
    uncorrected, these slow trends create a sloping or curved baseline in the
    phase-folded lightcurve that the CNN could mistake for a transit signal.

    lightkurve's flatten() fits a Savitzky-Golay polynomial filter to the flux
    and divides it out.  The key constraint is that the filter window must be
    MUCH wider than the transit duration so the filter does not "see" the
    transit dip and erroneously absorb it into the trend.

    With SG_WINDOW = 301 cadences ≈ 6.3 days, the widest transit in the KOI
    catalog (≈15 hours) is ≈30 cadences — only 10 % of the window.  Studies
    of Kepler data have shown that a 3-day window (144 cadences) already
    attenuates transits by < 1 % for durations < 12 hours, so 6.3 days gives
    us comfortable headroom.
    """
    lc = lc.remove_nans()

    if len(lc) < MIN_CADENCES:
        return None

    # ---- Upward outlier clipping -------------------------------------------
    flux = lc.flux.value
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    # 1.4826 is the consistency factor that converts MAD to an equivalent
    # standard deviation under the assumption of a normal distribution.
    sigma_est = mad * 1.4826
    upper = median + SIGMA_UPPER * sigma_est
    lc = lc[flux < upper]

    if len(lc) < MIN_CADENCES:
        return None

    # ---- Savitzky-Golay detrending -----------------------------------------
    # flatten() requires an odd window length.  Reduce if the lightcurve is
    # shorter than our preferred window (rare but possible for damaged quarters).
    window = SG_WINDOW
    if window >= len(lc):
        # Round down to nearest odd number less than len(lc).
        window = ((len(lc) - 1) // 2) * 2 + 1

    if window < 5:
        # scipy requires window_length ≥ polyorder + 2.  Default polyorder = 2.
        return None

    # niters=3 sigma=3.0: iteratively mask remaining outliers during the
    # polynomial fit so they don't distort the estimated baseline.
    flat_lc = lc.flatten(window_length=window, niters=3, sigma=3.0)
    return flat_lc


# ---------------------------------------------------------------------------
# Step B: Phase-fold and bin (once per KOI per star)
# ---------------------------------------------------------------------------


def _fold_and_bin(
    flat_lc: lk.LightCurve,
    period: float,
    epoch: float,
    transit_duration_hours: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Phase-fold a detrended lightcurve and produce the global + local views.

    Args:
        flat_lc:                Detrended lightcurve; baseline ≈ 1.0.
        period:                 Orbital period in days.
        epoch:                  Transit epoch in BKJD (BJD − 2 454 833.0).
                                Kepler lightcurve times are also in BKJD,
                                so no unit conversion is needed.
        transit_duration_hours: Transit duration in hours, used to set the
                                local view window width.

    Returns:
        (global_view, local_view) as float32 arrays of shape (201,) and (61,),
        or None if there are too few phase-folded points to fill the bins.
    """
    # Phase-fold -----------------------------------------------------------
    try:
        folded = flat_lc.fold(period=period, epoch_time=epoch)
    except Exception as exc:
        raise ValueError(f"fold() failed: {exc}") from exc

    phase = folded.phase.value   # dimensionless fractions in [-0.5, 0.5]
    flux = folded.flux.value     # normalized, baseline ≈ 1.0

    # Remove any NaN / inf introduced during folding.
    valid = np.isfinite(phase) & np.isfinite(flux)
    phase = phase[valid]
    flux = flux[valid]

    if len(phase) < 50:
        # Too few points to produce meaningful bins.
        return None

    # Global view ----------------------------------------------------------
    global_view, _, _ = binned_statistic(
        phase, flux, statistic="median", bins=GLOBAL_BINS, range=(-0.5, 0.5)
    )

    # Local view -----------------------------------------------------------
    # Compute local half-width in phase units and clamp to sane limits.
    transit_dur_phase = (transit_duration_hours / 24.0) / period
    local_half = float(
        np.clip(
            LOCAL_HALF_WIDTH_MULTIPLIER * transit_dur_phase,
            MIN_LOCAL_HALF_WIDTH,
            MAX_LOCAL_HALF_WIDTH,
        )
    )

    local_view, _, _ = binned_statistic(
        phase, flux, statistic="median", bins=LOCAL_BINS,
        range=(-local_half, local_half),
    )

    # Normalize -----------------------------------------------------------
    global_view = (global_view - 1.0).astype(np.float32)
    local_view = (local_view - 1.0).astype(np.float32)

    # Empty bins come back as NaN from binned_statistic.
    # Replace with 0.0 = the out-of-transit baseline level.
    global_view = np.nan_to_num(global_view, nan=0.0)
    local_view = np.nan_to_num(local_view, nan=0.0)

    return global_view, local_view


# ---------------------------------------------------------------------------
# Per-star driver
# ---------------------------------------------------------------------------


def _process_star(
    kepid: int,
    group: pd.DataFrame,
    force: bool,
) -> tuple[list[dict], dict[str, int]]:
    """
    Load, clean, and process all KOIs for a single host star.

    Loading and detrending are done once; phase-folding is repeated for
    each KOI (different period/epoch per planet on the same star).

    Returns:
        results: list of dicts, one per successfully processed KOI.
                 Each dict has keys: kepoi_name, kepid, koi_disposition,
                 label, global_view, local_view.
        counts:  tally of outcomes for progress reporting.
    """
    counts = {"ok": 0, "no_file": 0, "short": 0, "error": 0}
    results: list[dict] = []

    # ------------------------------------------------------------------
    # Filter KOIs that are already processed (unless --force).
    # ------------------------------------------------------------------
    if not force:
        pending = group[
            ~group["kepoi_name"].apply(lambda n: _npz_path(n).exists())
        ]
        already_done = len(group) - len(pending)
        if already_done:
            counts["ok"] += already_done   # count cached as OK for reporting
        group = pending

    if group.empty:
        return results, counts

    # ------------------------------------------------------------------
    # Load and detrend the lightcurve (once for all KOIs on this star).
    # ------------------------------------------------------------------
    fits_path = _fits_path(kepid)
    if not fits_path.exists():
        counts["no_file"] += len(group)
        return results, counts

    try:
        lc = lk.read(str(fits_path))
        flat_lc = _clean_and_detrend(lc)
    except Exception as exc:
        log.warning(f"KIC {kepid}: load/detrend failed — {exc}")
        counts["error"] += len(group)
        return results, counts

    if flat_lc is None:
        log.debug(f"KIC {kepid}: too short after cleaning, skipping {len(group)} KOI(s).")
        counts["short"] += len(group)
        return results, counts

    # ------------------------------------------------------------------
    # Phase-fold and bin for each KOI on this star.
    # ------------------------------------------------------------------
    for _, row in group.iterrows():
        try:
            views = _fold_and_bin(
                flat_lc,
                period=float(row["koi_period"]),
                epoch=float(row["koi_time0bk"]),
                transit_duration_hours=float(row["koi_duration"]),
            )
        except Exception as exc:
            log.debug(f"  KOI {row['kepoi_name']}: fold/bin error — {exc}")
            counts["error"] += 1
            continue

        if views is None:
            log.debug(f"  KOI {row['kepoi_name']}: too few points after fold.")
            counts["short"] += 1
            continue

        global_view, local_view = views

        results.append(
            {
                "kepoi_name": row["kepoi_name"],
                "kepid": kepid,
                "koi_disposition": row["koi_disposition"],
                "label": LABEL_MAP[row["koi_disposition"]],
                "global_view": global_view,
                "local_view": local_view,
            }
        )
        counts["ok"] += 1

    return results, counts


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_preprocessing(
    catalog: pd.DataFrame,
    max_stars: int | None,
    force: bool,
) -> None:
    """
    Iterate over every star in *catalog*, process its KOIs, and write output.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Group by star so each lightcurve is loaded and detrended only once.
    star_groups = list(catalog.groupby("kepid"))
    if max_stars is not None:
        star_groups = star_groups[:max_stars]

    n_stars = len(star_groups)
    n_kois = sum(len(g) for _, g in star_groups)
    log.info(f"Processing {n_kois} KOIs across {n_stars} stars ...")

    manifest_rows: list[dict] = []
    overall = {"ok": 0, "no_file": 0, "short": 0, "error": 0}

    for i, (kepid, group) in enumerate(star_groups, start=1):
        if i == 1 or i % 100 == 0:
            pct = 100 * i / n_stars
            log.info(f"  [{i:>4}/{n_stars}] ({pct:4.1f}%)  KIC {int(kepid):>9d}")

        results, counts = _process_star(int(kepid), group, force=force)

        for key in overall:
            overall[key] += counts.get(key, 0)

        for r in results:
            if r.get("global_view") is None:
                # Cached KOIs are counted in ok but have no arrays to save.
                continue

            out_path = _npz_path(r["kepoi_name"])

            # Save global_view (201,) and local_view (61,) as a compressed
            # numpy archive.  Compression is ~5× smaller than uncompressed
            # with negligible load-time overhead for arrays this small.
            np.savez_compressed(
                out_path,
                global_view=r["global_view"],
                local_view=r["local_view"],
                label=np.int8(r["label"]),
            )

            manifest_rows.append(
                {
                    "kepoi_name": r["kepoi_name"],
                    "kepid": r["kepid"],
                    "koi_disposition": r["koi_disposition"],
                    "label": r["label"],
                    "path": str(out_path.relative_to(ROOT)),
                }
            )

    # ------------------------------------------------------------------
    # Write / append manifest
    # ------------------------------------------------------------------
    if manifest_rows:
        new_df = pd.DataFrame(manifest_rows)

        if MANIFEST_FILE.exists() and not force:
            existing = pd.read_csv(MANIFEST_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset="kepoi_name", keep="last")
            combined.to_csv(MANIFEST_FILE, index=False)
            manifest_df = combined
        else:
            new_df.to_csv(MANIFEST_FILE, index=False)
            manifest_df = new_df
    else:
        manifest_df = pd.read_csv(MANIFEST_FILE) if MANIFEST_FILE.exists() else pd.DataFrame()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    n_files = sum(1 for _ in PROCESSED_DIR.glob("*.npz"))
    log.info("=" * 60)
    log.info("Preprocessing complete.")
    log.info(f"  ok         : {overall['ok']}")
    log.info(f"  no_file    : {overall['no_file']}  (star not downloaded in Phase 1)")
    log.info(f"  short      : {overall['short']}  (too few cadences after cleaning)")
    log.info(f"  error      : {overall['error']}")
    log.info(f"  .npz files : {n_files}  →  {PROCESSED_DIR}")
    log.info(f"  manifest   : {MANIFEST_FILE}")

    if not manifest_df.empty and "label" in manifest_df.columns:
        dist = manifest_df["label"].value_counts().to_dict()
        total = len(manifest_df)
        log.info(
            f"  class dist : {dist.get(1, 0)} planet(s) / {dist.get(0, 0)} false-positive(s)"
            f"  ({100 * dist.get(1, 0) / total:.1f} % positive)"
        )
    log.info("=" * 60)
