# src/predict.py

import torch
import numpy as np

from pathlib import Path
import sys
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.model import ResNet18_1D
from src.data.dataset_builder import get_segments


# -------------------------
# Load Model
# -------------------------
def load_model(model_path):
    model = ResNet18_1D()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# -------------------------
# Predict Function
# -------------------------
def predict_target(model_name, target, threshold=0.5):
    """
    Predict whether a target contains an exoplanet signal.

    Returns:
        dict with prediction, confidence, and segment info
    """
    models_dir = Path("models")

    # Load model
    model = load_model(models_dir / f"{model_name}.pt")

    # Get preprocessed segments
    mission = "TESS"
    if "Kepler" in target:
        mission = "Kepler"
    elif "TIC" in target:
        mission = "TESS"
    segments = get_segments(target, mission=mission)

    if len(segments) == 0:
        raise ValueError("No valid segments found for this target.")

    # Convert to tensor
    X = torch.tensor(segments, dtype=torch.float32)
    X = X.unsqueeze(1)  # (N, 1, L)

    # Run model
    with torch.no_grad():
        outputs = model(X)
        probs = torch.sigmoid(outputs).squeeze().numpy()

    # -------------------------
    # Aggregate predictions
    # -------------------------
    mean_prob = np.mean(probs)

    prediction = int(mean_prob > threshold)

    return {
        "prediction": prediction,
        "confidence": float(mean_prob),
        "num_segments": len(segments)
    }