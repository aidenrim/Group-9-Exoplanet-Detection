import argparse
from pathlib import Path
import sys
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_builder import build_dataset

planet_targets = [
    "Kepler-10",
    "Kepler-22"
]

non_planet_targets = [
    "Kepler-20",
    "Kepler-21"
]

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
output_path = project_root / "data" / "processed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument(
        "--planet_targets",
        nargs="+",
        required=True,
        help="List of target strings"
    )

    parser.add_argument(
        "--non_planet_targets",
        nargs="+",
        required=True,
        help="List of column strings"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset"
    )

    args = parser.parse_args()
    filepath = output_path / args.dataset_name
    build_dataset(args.planet_targets, args.non_planet_targets, filepath)