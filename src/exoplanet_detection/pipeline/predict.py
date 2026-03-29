from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from exoplanet_detection.service import ExoplanetPredictor


def predict_from_target(
    target_id: str,
    mission: str = "TESS",
    artifacts_dir: str | Path = "artifacts",
) -> dict:
    predictor = ExoplanetPredictor(artifacts_dir=artifacts_dir)
    result = predictor.predict_from_target(target_id=target_id, mission=mission)
    return result.to_dict()


def predict_from_csv(
    csv_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
) -> dict:
    frame = pd.read_csv(csv_path)
    if not {"time", "flux"} <= set(frame.columns):
        raise ValueError("Input CSV must include 'time' and 'flux' columns.")

    predictor = ExoplanetPredictor(artifacts_dir=artifacts_dir)
    result = predictor.predict_from_arrays(
        time=frame["time"].to_numpy(),
        flux=frame["flux"].to_numpy(),
        target_id=Path(csv_path).stem,
        mission="uploaded",
    )
    return asdict(result)

