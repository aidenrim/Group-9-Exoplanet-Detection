import argparse
from pathlib import Path
import sys
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to save"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)"
    )

    args = parser.parse_args()

    train(args.dataset_name, args.model_name, epochs=args.epochs)