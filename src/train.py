import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os

from tqdm import tqdm

from model import ResNet18_1D
from dataloader import get_dataloaders
from evaluate import evaluate_model


# -------------------------
# Config
# -------------------------
EPOCHS = 10
LR = 1e-3
MODEL_PATH = "../models/resnet18_1d.pt"


# -------------------------
# Training Function
# -------------------------
def train():

    # Create folders if needed
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        "../data/processed/X.npy",
        "../data/processed/y.npy"
    )

    # Model
    model = ResNet18_1D()

    # Loss + optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in tqdm(range(EPOCHS)):

        model.train()
        running_loss = 0.0

        for X, y in train_loader:
            y = y.unsqueeze(1)

            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                y = y.unsqueeze(1)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # -------------------------
    # Save Model
    # -------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # -------------------------
    # Plot Loss Curves
    # -------------------------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig("../results/loss_curve.png")
    plt.close()

    # -------------------------
    # Evaluate on Test Set
    # -------------------------
    metrics = evaluate_model(model, test_loader)

    # Save metrics
    with open("../results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Final Metrics:", metrics)


if __name__ == "__main__":
    train()