from __future__ import annotations

import argparse
import json
from pathlib import Path

from exoplanet_detection.config import TrainingConfig
from exoplanet_detection.data.ingestion import load_tabular_dataset
from exoplanet_detection.pipeline.train import train_from_csv
from exoplanet_detection.service import ExoplanetPredictor
from curate_dataset import curate_dataset
from make_synthetic_dataset import build_dataset


def run_backend_workflow(
    working_dir: Path,
    n_samples: int = 600,
    seed: int = 42,
    epochs: int = 20,
    classification_only: bool = True,
) -> dict:
    data_dir = working_dir / "data"
    artifacts_dir = working_dir / "artifacts"
    raw_csv = data_dir / "synthetic_raw.csv"
    curated_csv = data_dir / "curated_light_curves.csv"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    build_dataset(n_samples=n_samples, output_csv=raw_csv, seed=seed)
    curation = curate_dataset(input_csv=raw_csv, output_csv=curated_csv, seed=seed)

    training = train_from_csv(
        dataset_csv=curated_csv,
        output_dir=artifacts_dir,
        config=TrainingConfig(epochs=epochs),
        classification_only=classification_only,
    )

    predictor = ExoplanetPredictor(artifacts_dir=artifacts_dir)
    samples = load_tabular_dataset(curated_csv)
    probe = next((s for s in samples if s.split == "test"), samples[0])
    result = predictor.predict_from_arrays(
        time=probe.time,
        flux=probe.flux,
        target_id=probe.target_id,
        mission=probe.mission,
    )

    report = {
        "curation": curation,
        "training": training,
        "example_prediction": result.to_dict(),
    }
    (artifacts_dir / "backend_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backend-only workflow (curation -> training -> prediction).")
    parser.add_argument("--working-dir", type=Path, default=Path("."))
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train full pipeline (classification + regression). Default is classification-only.",
    )
    args = parser.parse_args()

    report = run_backend_workflow(
        working_dir=args.working_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        epochs=args.epochs,
        classification_only=not args.full,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
