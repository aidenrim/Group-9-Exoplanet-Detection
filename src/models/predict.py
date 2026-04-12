"""
Prediction library
==================

Single-KOI inference for the ExoplanetCNN.

Functions
---------
load_model(checkpoint_path)              -> (ExoplanetCNN, metadata_dict)
predict_koi(model_name, kepoi_name, ...) -> result_dict
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import MANIFEST_FILE, ROOT
from src.models.model import ExoplanetCNN


def load_model(checkpoint_path: Path) -> tuple[ExoplanetCNN, dict]:
    """
    Load an ExoplanetCNN from a checkpoint file.

    Args:
        checkpoint_path: Absolute path to a best_model.pt checkpoint.

    Returns:
        (model, metadata) where model is in eval mode on CPU and metadata is:
            {
                "threshold": float   # F1-optimal threshold; 0.5 if not saved
                "val_auc":   float   # best validation AUC from training
                "epoch":     int|str # epoch at which checkpoint was saved
            }
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = ExoplanetCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metadata = {
        "threshold": ckpt.get("threshold", 0.5),
        "val_auc":   ckpt.get("best_val_auc", float("nan")),
        "epoch":     ckpt.get("epoch", "?"),
    }
    return model, metadata


def predict_koi(
    model_name: str,
    kepoi_name: str,
    threshold: float | None = None,
) -> dict:
    """
    Run inference for a single KOI.

    Args:
        model_name: Directory name under results/ (e.g. "run_v1").
        kepoi_name: KOI identifier from the manifest (e.g. "K00010.01").
        threshold:  Decision threshold in [0, 1].  If None, the value saved
                    in the checkpoint is used (F1-optimal on the val set), or
                    0.5 if the checkpoint pre-dates threshold saving.

    Returns a dict with keys:
        probability     float       raw sigmoid output in [0, 1]
        threshold       float       threshold actually applied
        prediction      int         1 = PLANET, 0 = FALSE POSITIVE
        confidence      float       prob if PLANET; (1 - prob) if FALSE POSITIVE
        epoch           int | str   checkpoint epoch
        val_auc         float       best val AUC from training
        kepoi_name      str
        kepid           int
        koi_disposition str         "CONFIRMED", "CANDIDATE", or "FALSE POSITIVE"
        known_label     int         0 or 1 from manifest
        global_view     np.ndarray  shape (201,) — phase-folded full view
        local_view      np.ndarray  shape  (61,) — zoomed transit window

    Raises:
        FileNotFoundError: if the checkpoint or .npz data file is missing.
        KeyError:          if kepoi_name is not in the manifest.
    """
    checkpoint_path = ROOT / "results" / model_name / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run scripts/train.py --name {model_name} --dataset <dataset> first."
        )

    model, meta = load_model(checkpoint_path)

    if threshold is None:
        threshold = meta["threshold"]

    # --- look up candidate in manifest ------------------------------------
    manifest = pd.read_csv(MANIFEST_FILE)
    matches = manifest[manifest["kepoi_name"] == kepoi_name]
    if matches.empty:
        raise KeyError(
            f"Candidate '{kepoi_name}' not found in manifest.\n"
            f"Example names: {manifest['kepoi_name'].sample(min(5, len(manifest))).tolist()}"
        )
    row = matches.iloc[0]

    # --- load preprocessed arrays ----------------------------------------
    npz_path = ROOT / row["path"]
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {npz_path}\n"
            "Run scripts/preprocess.py to regenerate it."
        )
    data = np.load(npz_path)
    global_view_np = data["global_view"]   # (201,)
    local_view_np  = data["local_view"]    # (61,)

    # --- inference (CPU for numerical stability) -------------------------
    global_t = torch.from_numpy(global_view_np).float().unsqueeze(0)  # (1, 201)
    local_t  = torch.from_numpy(local_view_np).float().unsqueeze(0)   # (1, 61)

    with torch.no_grad():
        prob = model.predict_proba(global_t, local_t).item()

    prediction = 1 if prob >= threshold else 0
    confidence = prob if prediction == 1 else 1.0 - prob

    return {
        "probability":     prob,
        "threshold":       threshold,
        "prediction":      prediction,
        "confidence":      confidence,
        "epoch":           meta["epoch"],
        "val_auc":         meta["val_auc"],
        "kepoi_name":      kepoi_name,
        "kepid":           int(row["kepid"]),
        "koi_disposition": str(row["koi_disposition"]),
        "known_label":     int(row["label"]),
        "global_view":     global_view_np,
        "local_view":      local_view_np,
    }
