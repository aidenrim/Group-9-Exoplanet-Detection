import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from pathlib import Path
import sys
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.model import ResNet18_1D
from src.data.ExoplanetDataset import get_dataloaders
from src.models.evaluate import evaluate_model


# Config
EPOCHS = 30
LR = 1e-3


# Training Function
def train(dataset_name="default", model_name="default", epochs=EPOCHS):

    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path("data") / "processed"

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_dir / f"{dataset_name}_X.npy",
        dataset_dir / f"{dataset_name}_y.npy"
    )

    # Model
    model = ResNet18_1D()

    # Loss + optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    # Training Loop
    for epoch in tqdm(range(epochs)):

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

        # Validation
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

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), models_dir / f"{model_name}.pt")
    print(f"Model saved to {models_dir / f'{model_name}.pt'}")

    # Plot Loss Curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig(results_dir / f"{model_name}_loss_curve.png")
    plt.close()

    # Evaluate on Test Set
    metrics = evaluate_model(model, test_loader)

    # Save metrics
    with open(results_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Final Metrics:", metrics)


if __name__ == "__main__":
    train()