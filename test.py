import torch
from pathlib import Path
import numpy as np

"""
from dataloader import get_dataloaders
from model import ResNet18_1D
from evaluate import evaluate_model
import json

from utils import load_model


train_loader, val_loader, test_loader = get_dataloaders(
    "../data/processed/X.npy",
    "../data/processed/y.npy"
)

model = load_model()
metrics = evaluate_model(model, test_loader)

# Save metrics
with open("../results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Final Metrics:", metrics)
"""