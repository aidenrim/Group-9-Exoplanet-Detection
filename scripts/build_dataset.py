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

# Stars confirmed to have NO exoplanets (do not use Kepler-20/21 — they have confirmed planets)
non_planet_targets = [
    "KIC 3733346",
    "KIC 4914423"
]

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
    build_dataset(args.planet_targets, args.non_planet_targets, args.dataset_name)