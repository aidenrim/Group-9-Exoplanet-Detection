from __future__ import annotations

import argparse
import json
from pathlib import Path

from exoplanet_detection.config import TrainingConfig
from exoplanet_detection.pipeline.train import train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exoplanet detection models.")
    parser.add_argument("--dataset", type=Path, required=True, help="CSV with target_id, mission, time, flux, label.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help="Skip regression training and produce classification-only artifacts.",
    )
    args = parser.parse_args()

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        bins=args.bins,
        learning_rate=args.learning_rate,
    )
    metrics = train_from_csv(
        args.dataset,
        output_dir=args.output_dir,
        config=cfg,
        classification_only=args.classification_only,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
