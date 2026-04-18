import numpy as np

def clean_lightcurve(time, flux, flux_err=None, sigma=5):
    """
    Removes NaNs and extreme outliers from a light curve.
    """

    # Remove NaNs
    mask = ~np.isnan(time) & ~np.isnan(flux)
    if flux_err is not None:
        mask &= ~np.isnan(flux_err)

    time = time[mask]
    flux = flux[mask]

    # Remove outliers using sigma clipping
    median = np.median(flux)
    std = np.std(flux)

    good = np.abs(flux - median) < sigma * std

    time = time[good]
    flux = flux[good]

    return time, flux


def normalize_flux(flux):
    """
    Normalize flux to zero mean and unit variance.
    """
    mean = np.mean(flux)
    std = np.std(flux)

    if std == 0:
        return flux

    return (flux - mean) / std


def segment_lightcurve(time, flux, window_size=1024, stride=512):
    """
    Break a light curve into overlapping segments.
    """

    segments = []

    N = len(flux)

    for start in range(0, N - window_size, stride):
        end = start + window_size

        segment = flux[start:end]

        if len(segment) == window_size:
            segments.append(segment)

    return np.array(segments)



def phase_fold(time, flux, period):
    """
    Phase-fold a light curve on a given period.

    Returns (phase, folded_flux) sorted by phase in [0, 1).
    """
    phase = (time % period) / period
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def preprocess_lightcurve(lc, window_size=1024, stride=512, fold_period=None):
    """
    Full preprocessing pipeline for a LightCurve object.
    """

    # Extract raw arrays
    time = lc.time.value
    flux = lc.flux.value

    # Step 1: Clean
    time, flux = clean_lightcurve(time, flux)

    # Step 2: Detrend (using lightkurve)
    lc_clean = lc.copy()
    lc_clean = lc_clean.remove_nans().flatten()

    flux = lc_clean.flux.value
    time = lc_clean.time.value

    # Step 3: Normalize
    flux = normalize_flux(flux)

    # Step 4 (optional): Phase-fold on known/candidate period
    if fold_period is not None:
        _, flux = phase_fold(time, flux, fold_period)

    # Step 5: Segment
    segments = segment_lightcurve(time, flux, window_size, stride)

    return segments