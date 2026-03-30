import argparse
from pathlib import Path
import sys
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.predict import predict_target



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use"
    )

    parser.add_argument(
        "--target_name",
        type=str,
        required=True,
        help="Name of the target to predict"
    )

    args = parser.parse_args()

    result = predict_target(
        model_name=args.model_name,
        target=args.target_name
    )

    print(result)