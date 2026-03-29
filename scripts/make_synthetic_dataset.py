from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from exoplanet_detection.data.ingestion import LightCurveSample, save_tabular_dataset


def _simulate_curve(
    rng: np.random.Generator,
    is_planet: bool,
    n_points: int = 1500,
    duration_days: float = 27.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    time = np.linspace(0.0, duration_days, n_points, dtype=np.float32)
    baseline = 1.0 + rng.normal(0, 0.0008, size=n_points).astype(np.float32)
    periodic = 0.0005 * np.sin(2 * np.pi * time / rng.uniform(5.0, 12.0)).astype(np.float32)
    flux = baseline + periodic

    if not is_planet:
        return time, flux, np.nan, np.nan

    period = float(rng.uniform(1.5, 9.0))
    depth = float(rng.uniform(0.001, 0.02))
    width_days = float(rng.uniform(0.08, 0.35))
    phase = np.mod(time, period)
    in_transit = np.minimum(phase, period - phase) < (width_days / 2.0)
    flux = flux.copy()
    flux[in_transit] -= depth

    radius_rearth = max(1.0, 110.0 * np.sqrt(depth))
    return time, flux, period, radius_rearth


def build_dataset(n_samples: int, output_csv: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    samples: list[LightCurveSample] = []

    for i in range(n_samples):
        is_planet = bool(rng.random() < 0.35)
        time, flux, period, radius = _simulate_curve(rng, is_planet)
        mission = "TESS" if i % 2 == 0 else "KEPLER"
        samples.append(
            LightCurveSample(
                target_id=f"SYNTH-{i:04d}",
                mission=mission,
                time=time,
                flux=flux,
                label=int(is_planet),
                period_days=None if np.isnan(period) else period,
                radius_rearth=None if np.isnan(radius) else radius,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_tabular_dataset(samples, output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic transit dataset.")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_light_curves.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.n_samples, args.output, args.seed)
    print(f"Synthetic dataset written to {args.output}")


if __name__ == "__main__":
    main()

