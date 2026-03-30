import numpy as np
from data_loader import get_segments


def build_dataset(planet_targets, non_planet_targets, save_path="data/processed/"):
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

    # Save
    np.save(save_path + "X.npy", X)
    np.save(save_path + "y.npy", y)

    return X, y