from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_regressor(random_state: int = 42) -> Pipeline: 
    """
    Builds a regression pipeline for predicting exoplanet parameters
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_regressor(features: np.ndarray, targets: np.ndarray, random_state: int = 42) -> Pipeline:
    model = build_regressor(random_state=random_state)
    model.fit(features, targets)
    return model


def save_regressor(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_regressor(path: str) -> Pipeline:
    return joblib.load(path)

