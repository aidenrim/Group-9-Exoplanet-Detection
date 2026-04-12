"""
Regression models for exoplanet characterisation.

- RadiusRegressor: predicts planet radius (Earth radii) from transit depth and stellar radius.
- PeriodRegressor: predicts orbital period (days) from BLS periodogram of a light curve.

Both estimators follow the sklearn API (fit / predict).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


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
