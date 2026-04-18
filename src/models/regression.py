import logging

import lightkurve as lk
import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares

from src.data.preprocess import _clean_and_detrend, SG_WINDOW, SG_WINDOW_TESS

log = logging.getLogger(__name__)

# Transit durations to search over. Using three representative values covers
# the range from hot Jupiters (~1 h) to long-period giants (~8 h) without
# blowing up compute time.
_DURATIONS_DAYS = np.array([0.05, 0.1, 0.2])   # hours: ~1.2, 2.4, 4.8


def estimate_period_bls(
    lc: lk.LightCurve,
    period_min: float = 0.5,
    period_max: float = 50.0,
    n_periods: int = 10_000,
) -> dict | None:
    """
    Estimate the orbital period of a transit signal using Box Least Squares.

    Detrending is handled internally using the same Savitzky-Golay pipeline as
    the preprocessing step, so the caller should pass the raw (undetrended)
    lightcurve as returned by ``lk.read(fits_path)``.

    Args:
        lc:          Raw LightCurve as loaded by lightkurve (not yet detrended).
        period_min:  Lower bound of the period search grid in days (default 0.5).
        period_max:  Upper bound of the period search grid in days (default 50.0).
                     For TESS single-sector data consider capping at ~27 days.
        n_periods:   Number of trial periods in the grid (default 10 000).

    Returns:
        A dict with keys:
            best_period   float         Period at maximum BLS power, in days.
            best_duration float         Estimated transit duration, in hours.
            best_t0       float         Estimated transit epoch in mission-native
                                        time (BKJD for Kepler, BTJD for TESS).
            periods       np.ndarray    Shape (n_periods,) — period grid, days.
            power         np.ndarray    Shape (n_periods,) — BLS power values.

        Returns None if detrending fails (lightcurve too short or degenerate).
    """
    # Select the correct SG window based on mission cadence.
    mission = str(getattr(lc, "mission", "") or "")
    sg_window = SG_WINDOW_TESS if "TESS" in mission.upper() else SG_WINDOW

    flat_lc = _clean_and_detrend(lc, sg_window=sg_window)
    if flat_lc is None:
        log.warning("estimate_period_bls: detrending failed — lightcurve too short or degenerate.")
        return None

    # Use astropy's BLS directly so we evaluate at exactly our period grid,
    # rather than lightkurve's wrapper which ignores the explicit array and
    # recomputes its own grid from frequency_factor and the baseline.
    time = np.asarray(flat_lc.time.value, dtype=float)   # BKJD / BTJD, days
    flux = np.asarray(flat_lc.flux,       dtype=float)   # normalised, baseline ≈ 1

    periods   = np.linspace(period_min, period_max, n_periods)   # days
    model     = BoxLeastSquares(time * u.day, flux)
    result    = model.power(periods * u.day, _DURATIONS_DAYS * u.day)

    best_idx  = int(np.argmax(result.power))

    return {
        "best_period":   float(result.period[best_idx].to(u.day).value),
        "best_duration": float(result.duration[best_idx].to(u.hour).value),
        "best_t0":       float(result.transit_time[best_idx].value),
        "periods":       periods,
        "power":         np.asarray(result.power),
    }
