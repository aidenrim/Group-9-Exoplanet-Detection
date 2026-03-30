from dataset_builder import build_dataset

planet_targets = [
    "Kepler-10",
    "Kepler-22"
]

non_planet_targets = [
    "Kepler-20",
    "Kepler-21"
]

build_dataset(planet_targets, non_planet_targets)