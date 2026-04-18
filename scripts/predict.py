"""
Classify a single KOI candidate using a trained ExoplanetCNN.

Looks up the candidate by kepoi_name in data/datasets/manifest.csv,
loads its preprocessed .npz arrays, and runs the model forward pass
on CPU to produce a planet probability and classification.

The decision threshold used is the one saved into the checkpoint during
training (the F1-optimal threshold found on the validation set).  If the
checkpoint pre-dates this feature, 0.5 is used as a fallback.

Usage:
    python scripts/predict.py --model run_v1 --candidate K00010.01
    python scripts/predict.py --model run_v1 --candidate K00113.01
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.data.dataset import MANIFEST_FILE
from src.models.model import ExoplanetCNN

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a single KOI candidate using a trained ExoplanetCNN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="NAME",
        help="Model name to use (checkpoint loaded from results/{name}/best_model.pt).",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        metavar="KOI_NAME",
        help="KOI candidate name to classify, e.g. K00010.01.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Locate checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = ROOT / "results" / args.model / "best_model.pt"
    if not checkpoint_path.exists():
        log.error(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run scripts/train.py --name {args.model} --dataset <dataset> first."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Look up candidate in manifest
    # ------------------------------------------------------------------
    if not MANIFEST_FILE.exists():
        log.error(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run scripts/preprocess.py first."
        )
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_FILE)
    matches = manifest[manifest["name"] == args.candidate]
    if matches.empty:
        log.error(
            f"Candidate '{args.candidate}' not found in manifest.\n"
            f"Example names: {manifest['name'].sample(min(5, len(manifest))).tolist()}"
        )
        sys.exit(1)
    row = matches.iloc[0]

    # ------------------------------------------------------------------
    # 3. Load preprocessed arrays
    # ------------------------------------------------------------------
    npz_path = ROOT / row["path"]
    if not npz_path.exists():
        log.error(
            f"Preprocessed data file not found: {npz_path}\n"
            "Run scripts/preprocess.py to regenerate it."
        )
        sys.exit(1)

    data = np.load(npz_path)
    global_view = torch.from_numpy(data["global_view"]).float().unsqueeze(0)  # (1, 201)
    local_view  = torch.from_numpy(data["local_view"]).float().unsqueeze(0)   # (1, 61)

    # ------------------------------------------------------------------
    # 4. Load model and threshold from checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = ExoplanetCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # threshold is saved after training by src/models/train.py; fall back
    # to 0.5 for checkpoints created before this feature was added.
    threshold = ckpt.get("threshold", 0.5)
    best_val_auc = ckpt.get("best_val_auc", float("nan"))
    epoch        = ckpt.get("epoch", "?")

    # ------------------------------------------------------------------
    # 5. Inference — always on CPU for numerical stability
    # ------------------------------------------------------------------
    with torch.no_grad():
        prob = model.predict_proba(global_view, local_view).item()

    prediction = "PLANET" if prob >= threshold else "FALSE POSITIVE"
    # Confidence = how far the probability is from the opposite outcome.
    confidence = prob if prediction == "PLANET" else 1.0 - prob

    known_label = {
        1:  "CONFIRMED",
        0:  "FALSE POSITIVE",
        -1: "CANDIDATE",
    }.get(int(row["label"]), "UNKNOWN")

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print()
    print("=" * 52)
    print(f"  Candidate    : {args.candidate}")
    print(f"  Known label  : {known_label}  ({row['disposition']})")
    print(f"  Model        : {args.model}  "
          f"(epoch {epoch}, val AUC {best_val_auc:.4f})")
    print(f"  Probability  : {prob:.4f}")
    print(f"  Threshold    : {threshold:.2f}")
    print(f"  Prediction   : {prediction}  (confidence {confidence:.1%})")
    print("=" * 52)
    print()


if __name__ == "__main__":
    main()
