import numpy as np
from pathlib import Path
from lightkurve import search_lightcurve
import sys
# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.preprocessing import preprocess_lightcurve


def build_dataset(planet_targets, non_planet_targets, save_name="default"):
    X = []
    y = []

    # positive (planet = 1)
    for target in planet_targets:
        segments = get_segments(target)


        X.append(segments)
        y.append(np.ones(len(segments)))

    # negative (no planet = 0)
    for target in non_planet_targets:
        segments = get_segments(target)

        X.append(segments)
        y.append(np.zeros(len(segments)))

    # Combine everything
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    print("Final dataset shape:", X.shape, y.shape)

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save
    np.save(output_dir / f"{save_name}_X.npy", X)
    np.save(output_dir / f"{save_name}_y.npy", y)

    return X, y


def get_segments(target, mission="Kepler"):
    if "Kepler" in target:
        mission = "Kepler"
    elif "TIC" in target:
        mission = "TESS"
    search_result = search_lightcurve(target, mission=mission)
    lc_collection = search_result.download_all()
    lc = lc_collection.stitch()

    segments = preprocess_lightcurve(lc)

    return segments