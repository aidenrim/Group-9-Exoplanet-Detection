import torch
from pathlib import Path
import sys
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.model import ResNet18_1D


def load_model(path="../models/resnet18_1d.pt"):
    model = ResNet18_1D()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model