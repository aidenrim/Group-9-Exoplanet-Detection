from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from exoplanet_detection.config import Paths, TrainingConfig
from exoplanet_detection.data.datasets import FoldedCurveDataset
from exoplanet_detection.data.ingestion import load_tabular_dataset
from exoplanet_detection.data.preprocessing import preprocess_for_model
from exoplanet_detection.models.classifier import TransitCNN
from exoplanet_detection.models.regression import fit_regressor, save_regressor


def _prepare_arrays(dataset_path: str | Path, bins: int) -> dict[str, np.ndarray]:
    samples = load_tabular_dataset(dataset_path)
    curves = []
    features = []
    labels = []
    radius = []
    period = []
    splits = []

    for sample in samples:
        curve, feat = preprocess_for_model(sample.time, sample.flux, bins=bins)
        curves.append(curve)
        features.append(feat.to_array())
        labels.append(int(sample.label or 0))
        radius.append(np.nan if sample.radius_rearth is None else float(sample.radius_rearth))
        period.append(np.nan if sample.period_days is None else float(sample.period_days))
        splits.append(sample.split if sample.split is not None else "")

    return {
        "curves": np.asarray(curves, dtype=np.float32),
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "radius": np.asarray(radius, dtype=np.float32),
        "period": np.asarray(period, dtype=np.float32),
        "splits": np.asarray(splits, dtype=object),
    }


def _train_classifier(
    curves: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    cfg: TrainingConfig,
) -> tuple[TransitCNN, dict[str, float], np.ndarray, np.ndarray]:
    split_labels = {str(x).strip().lower() for x in splits if str(x).strip()}
    has_explicit_split = "train" in split_labels and ("test" in split_labels or "val" in split_labels)
    if has_explicit_split:
        train_idx = np.where(np.char.lower(splits.astype(str)) == "train")[0]
        eval_name = "test" if "test" in split_labels else "val"
        test_idx = np.where(np.char.lower(splits.astype(str)) == eval_name)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            has_explicit_split = False

    if not has_explicit_split:
        stratify = labels if len(np.unique(labels)) > 1 else None
        idx = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify,
        )

    train_dataset = FoldedCurveDataset(curves[train_idx], labels[train_idx])
    test_dataset = FoldedCurveDataset(curves[test_idx], labels[test_idx])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = TransitCNN(input_bins=cfg.bins)
    positive = labels[train_idx].sum()
    negative = len(train_idx) - positive
    pos_weight = torch.tensor([max(negative / max(positive, 1), 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for _ in range(cfg.epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    logits_list = []
    y_true = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(x_batch)
            logits_list.append(logits.numpy())
            y_true.append(y_batch.numpy())

    logits_all = np.concatenate(logits_list)
    y_true_all = np.concatenate(y_true).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits_all))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_all, preds)),
        "precision": float(precision_score(y_true_all, preds, zero_division=0)),
        "recall": float(recall_score(y_true_all, preds, zero_division=0)),
        "f1": float(f1_score(y_true_all, preds, zero_division=0)),
        "used_dataset_splits": bool(has_explicit_split),
    }

    return model, metrics, train_idx, test_idx


def _train_regressors(
    features: np.ndarray,
    labels: np.ndarray,
    radius: np.ndarray,
    period: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    random_state: int,
) -> dict[str, float | None]:
    train_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    metrics: dict[str, float | None] = {"radius_rmse": None, "period_rmse": None}

    radius_train_mask = train_mask & (labels == 1) & np.isfinite(radius)
    radius_test_mask = test_mask & (labels == 1) & np.isfinite(radius)
    if radius_train_mask.sum() >= 20 and radius_test_mask.sum() >= 5:
        radius_model = fit_regressor(features[radius_train_mask], radius[radius_train_mask], random_state)
        radius_preds = radius_model.predict(features[radius_test_mask])
        metrics["radius_rmse"] = float(
            math.sqrt(mean_squared_error(radius[radius_test_mask], radius_preds))
        )
        save_regressor(radius_model, str(output_dir / "radius_regressor.joblib"))

    period_train_mask = train_mask & (labels == 1) & np.isfinite(period)
    period_test_mask = test_mask & (labels == 1) & np.isfinite(period)
    if period_train_mask.sum() >= 20 and period_test_mask.sum() >= 5:
        period_model = fit_regressor(features[period_train_mask], period[period_train_mask], random_state)
        period_preds = period_model.predict(features[period_test_mask])
        metrics["period_rmse"] = float(
            math.sqrt(mean_squared_error(period[period_test_mask], period_preds))
        )
        save_regressor(period_model, str(output_dir / "period_regressor.joblib"))

    return metrics


def train_from_csv(
    dataset_csv: str | Path,
    output_dir: str | Path = "artifacts",
    config: TrainingConfig | None = None,
    classification_only: bool = False,
) -> dict[str, Any]:
    cfg = config or TrainingConfig()
    arrays = _prepare_arrays(dataset_csv, bins=cfg.bins)
    curves = arrays["curves"]
    features = arrays["features"]
    labels = arrays["labels"]
    radius = arrays["radius"]
    period = arrays["period"]
    splits = arrays["splits"]
    if len(labels) < 10:
        raise ValueError("At least 10 samples are required to train the baseline models.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    classifier, cls_metrics, train_idx, test_idx = _train_classifier(curves, labels, splits, cfg)
    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "input_bins": cfg.bins,
        },
        output_path / "classifier.pt",
    )

    reg_metrics: dict[str, float | bool | None]
    if classification_only:
        # Prevent stale regressors from older runs being used accidentally.
        for fname in ("radius_regressor.joblib", "period_regressor.joblib"):
            reg_path = output_path / fname
            if reg_path.exists():
                reg_path.unlink()
        reg_metrics = {
            "skipped": True,
            "radius_rmse": None,
            "period_rmse": None,
        }
    else:
        reg_metrics = _train_regressors(
            features=features,
            labels=labels,
            radius=radius,
            period=period,
            train_idx=train_idx,
            test_idx=test_idx,
            output_dir=output_path,
            random_state=cfg.random_state,
        )
        reg_metrics["skipped"] = False

    metadata = {
        "mode": "classification_only" if classification_only else "full",
        "training_config": asdict(cfg),
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "classification_metrics": cls_metrics,
        "regression_metrics": reg_metrics,
        "feature_names": [
            "transit_depth",
            "flux_std",
            "skew_proxy",
            "estimated_period",
            "duration_proxy",
        ],
    }

    with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def default_paths() -> Paths:
    return Paths().resolve()
