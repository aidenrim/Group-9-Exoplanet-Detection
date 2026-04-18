import numpy as np


def estimate_period_bls(lc):
    """
    Estimate the orbital period of an exoplanet using a Box Least Squares
    periodogram on a lightkurve LightCurve object.

    Returns the best-fit period in days, or None if the periodogram fails.
    """
    try:
        from lightkurve import periodogram
        pg = lc.to_periodogram(method="bls", minimum_period=0.5, maximum_period=30)
        best_period = pg.period_at_max_power.value
        return float(best_period)
    except Exception:
        return None
