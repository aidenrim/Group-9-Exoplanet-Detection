from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from exoplanet_detection.data.ingestion import LightCurveSample, load_tabular_dataset, save_tabular_dataset


def _clean_sample(sample: LightCurveSample, min_points: int, max_points: int) -> LightCurveSample | None:
    time = np.asarray(sample.time, dtype=np.float32)
    flux = np.asarray(sample.flux, dtype=np.float32)

    n = min(len(time), len(flux))
    if n < min_points:
        return None

    time = time[:n]
    flux = flux[:n]
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[finite_mask]
    flux = flux[finite_mask]
    if len(time) < min_points:
        return None

    order = np.argsort(time)
    time = time[order]
    flux = flux[order]

    _, unique_idx = np.unique(time, return_index=True)
    keep_idx = np.sort(unique_idx)
    time = time[keep_idx]
    flux = flux[keep_idx]
    if len(time) < min_points:
        return None

    if len(time) > max_points:
        pick = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time = time[pick]
        flux = flux[pick]

    return LightCurveSample(
        target_id=sample.target_id,
        mission=sample.mission,
        time=time,
        flux=flux,
        label=sample.label,
        period_days=sample.period_days,
        radius_rearth=sample.radius_rearth,
        split=sample.split,
    )


def _assign_splits(
    samples: list[LightCurveSample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    n = len(samples)
    if n < 3:
        for s in samples:
            s.split = "train"
        return

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    idx = np.arange(n)
    labels = np.asarray([int(s.label or 0) for s in samples], dtype=np.int64)
    stratify = labels if len(np.unique(labels)) > 1 else None

    try:
        train_idx, hold_idx = train_test_split(
            idx,
            train_size=train_ratio,
            random_state=seed,
            stratify=stratify,
        )
        hold_labels = labels[hold_idx]
        hold_stratify = hold_labels if len(np.unique(hold_labels)) > 1 else None
        rel_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            hold_idx,
            train_size=rel_val_ratio,
            random_state=seed,
            stratify=hold_stratify,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

    split_lookup = {int(i): "train" for i in train_idx}
    split_lookup.update({int(i): "val" for i in val_idx})
    split_lookup.update({int(i): "test" for i in test_idx})
    for i, sample in enumerate(samples):
        sample.split = split_lookup.get(i, "train")


def curate_dataset(
    input_csv: Path,
    output_csv: Path,
    min_points: int = 512,
    max_points: int = 4000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    raw = load_tabular_dataset(input_csv)
    curated = [clean for s in raw if (clean := _clean_sample(s, min_points=min_points, max_points=max_points)) is not None]
    if not curated:
        raise ValueError("No usable samples remained after curation.")

    _assign_splits(curated, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_tabular_dataset(curated, output_csv)

    labels = np.asarray([int(s.label or 0) for s in curated], dtype=np.int64)
    splits = [s.split for s in curated]
    report = {
        "input_samples": len(raw),
        "curated_samples": len(curated),
        "dropped_samples": len(raw) - len(curated),
        "label_counts": {
            "negative_0": int((labels == 0).sum()),
            "positive_1": int((labels == 1).sum()),
        },
        "split_counts": {
            "train": int(sum(1 for s in splits if s == "train")),
            "val": int(sum(1 for s in splits if s == "val")),
            "test": int(sum(1 for s in splits if s == "test")),
        },
        "missions": sorted({s.mission for s in curated}),
        "min_points": min_points,
        "max_points": max_points,
    }

    report_path = output_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate and split exoplanet light-curve dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV in project format.")
    parser.add_argument("--output", type=Path, default=Path("data/curated_light_curves.csv"))
    parser.add_argument("--min-points", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = curate_dataset(
        input_csv=args.input,
        output_csv=args.output,
        min_points=args.min_points,
        max_points=args.max_points,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

