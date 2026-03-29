from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exoplanet_detection.data.ingestion import fetch_light_curve
from exoplanet_detection.data.preprocessing import (
    detrend_flux,
    normalize_flux,
    phase_fold,
    preprocess_for_model,
)
from exoplanet_detection.models.classifier import TransitCNN
from exoplanet_detection.models.regression import load_regressor


@dataclass(slots=True)
class PredictionResult:
    target_id: str
    mission: str
    confidence: float
    is_exoplanet_candidate: bool
    predicted_radius_rearth: float | None
    predicted_period_days: float
    derived_features: dict[str, float]
    raw_time: np.ndarray
    raw_flux: np.ndarray
    folded_phase: np.ndarray
    folded_flux: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "mission": self.mission,
            "confidence": self.confidence,
            "is_exoplanet_candidate": self.is_exoplanet_candidate,
            "predicted_radius_rearth": self.predicted_radius_rearth,
            "predicted_period_days": self.predicted_period_days,
            "derived_features": self.derived_features,
        }


class ExoplanetPredictor:
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        artifacts = Path(artifacts_dir)
        checkpoint = torch.load(artifacts / "classifier.pt", map_location="cpu")
        bins = int(checkpoint.get("input_bins", 512))

        self.classifier = TransitCNN(input_bins=bins)
        self.classifier.load_state_dict(checkpoint["state_dict"])
        self.classifier.eval()
        self.bins = bins

        self.radius_regressor = None
        self.period_regressor = None
        radius_path = artifacts / "radius_regressor.joblib"
        period_path = artifacts / "period_regressor.joblib"
        if radius_path.exists():
            self.radius_regressor = load_regressor(str(radius_path))
        if period_path.exists():
            self.period_regressor = load_regressor(str(period_path))

    def predict_from_arrays(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        target_id: str = "custom",
        mission: str = "custom",
    ) -> PredictionResult:
        curve, features = preprocess_for_model(time=time, flux=flux, bins=self.bins)
        model_in = torch.tensor(curve, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logit = self.classifier(model_in).item()
        confidence = float(1.0 / (1.0 + np.exp(-logit)))
        is_candidate = confidence >= 0.5

        feat_arr = features.to_array().reshape(1, -1)
        radius = float(self.radius_regressor.predict(feat_arr)[0]) if self.radius_regressor is not None else None
        period = (
            float(self.period_regressor.predict(feat_arr)[0])
            if self.period_regressor is not None
            else float(features.estimated_period)
        )
        period = max(period, 1e-3)

        clean_flux = detrend_flux(normalize_flux(np.asarray(flux, dtype=np.float32)))
        phase, folded = phase_fold(np.asarray(time, dtype=np.float32), clean_flux, period_days=period)

        return PredictionResult(
            target_id=target_id,
            mission=mission,
            confidence=confidence,
            is_exoplanet_candidate=is_candidate,
            predicted_radius_rearth=radius,
            predicted_period_days=period,
            derived_features={
                "transit_depth": float(features.transit_depth),
                "flux_std": float(features.flux_std),
                "skew_proxy": float(features.skew_proxy),
                "estimated_period_from_signal": float(features.estimated_period),
                "duration_proxy": float(features.duration_proxy),
            },
            raw_time=np.asarray(time, dtype=np.float32),
            raw_flux=np.asarray(flux, dtype=np.float32),
            folded_phase=phase,
            folded_flux=folded,
        )

    def predict_from_target(self, target_id: str, mission: str = "TESS") -> PredictionResult:
        sample = fetch_light_curve(target_id=target_id, mission=mission)
        return self.predict_from_arrays(
            time=sample.time,
            flux=sample.flux,
            target_id=sample.target_id,
            mission=sample.mission,
        )

