from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from exoplanet_detection.service import ExoplanetPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict exoplanet candidacy for a target or custom CSV.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mission", type=str, default="TESS")
    parser.add_argument("--target-id", type=str, default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV with columns 'time' and 'flux' (used when --target-id is omitted).",
    )
    args = parser.parse_args()

    predictor = ExoplanetPredictor(artifacts_dir=args.artifacts_dir)

    if args.target_id:
        result = predictor.predict_from_target(target_id=args.target_id, mission=args.mission)
    elif args.csv:
        frame = pd.read_csv(args.csv)
        if not {"time", "flux"} <= set(frame.columns):
            raise ValueError("CSV must include columns: time, flux")
        result = predictor.predict_from_arrays(
            time=frame["time"].to_numpy(),
            flux=frame["flux"].to_numpy(),
            target_id=args.csv.stem,
            mission="uploaded",
        )
    else:
        raise ValueError("Provide either --target-id or --csv.")

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

