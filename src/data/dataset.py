"""
Dataset library
===============

PyTorch Dataset class, train/val/test split utilities, and split
persistence for the preprocessed KOI lightcurve arrays.

The split is done BY STAR (kepid), not by individual KOI.  Two planet
candidates on the same star were extracted from the same underlying
lightcurve, so their global-view arrays share the same noise floor, the
same stellar-variability residuals, and the same inter-quarter normalisation
artefacts.  Splitting by KOI would allow the model to see those patterns in
both training and test — a data leak that would inflate test-set performance
metrics without reflecting real generalisation.

Splitting by star and stratifying by the star's dominant label ensures:
  • No star appears in more than one split.
  • Each split has approximately the same positive/negative ratio.
  • Multi-planet systems are kept intact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = ROOT / "data" / "datasets"
MANIFEST_FILE = DATASETS_DIR / "manifest.csv"
TRAIN_FILE    = DATASETS_DIR / "train.csv"
VAL_FILE      = DATASETS_DIR / "val.csv"
TEST_FILE     = DATASETS_DIR / "test.csv"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class KOIDataset(Dataset):
    """
    Loads (global_view, local_view, label) triplets from preprocessed .npz files.

    Each item corresponds to one KOI (Kepler Object of Interest).  The arrays
    were written by preprocess.py and have fixed shapes:

        global_view : float32 (201,)   full phase-folded, baseline-centred
        local_view  : float32  (61,)   zoomed transit window
        label       : float32  scalar  read from the manifest (1 = planet,
                                       0 = false positive; -1 values should
                                       not appear here — remap or drop them
                                       in build_dataset.py before constructing
                                       this dataset)

    The label is float32 rather than int so it matches the shape expected by
    BCEWithLogitsLoss without an explicit cast in the training loop.
    """

    def __init__(self, manifest: pd.DataFrame) -> None:
        """
        Args:
            manifest: DataFrame slice with at minimum columns
                      [kepoi_name, kepid, label, path].
                      Pass in the train, val, or test sub-DataFrame from
                      make_splits() or load_splits(); do not pass the full
                      manifest here.
        """
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]

        # The 'path' column stores a path relative to the project root so that
        # the manifest stays portable across machines.
        data = np.load(ROOT / row["path"])

        global_view = torch.from_numpy(data["global_view"]).float()   # (201,)
        local_view  = torch.from_numpy(data["local_view"]).float()    # (61,)
        label       = torch.tensor(float(row["label"]), dtype=torch.float32)

        return global_view, local_view, label

    def get_labels(self) -> np.ndarray:
        """Return all labels as a 1-D numpy array.

        Used by make_weighted_sampler() to build per-sample weights without
        iterating through __getitem__ for every sample.
        """
        return self.manifest["label"].to_numpy(dtype=np.int64)

    def class_counts(self) -> tuple[int, int]:
        """Return (n_positives, n_negatives) for this split."""
        labels = self.get_labels()
        n_pos = int(labels.sum())
        return n_pos, len(labels) - n_pos


# ---------------------------------------------------------------------------
# Weighted sampler (optional alternative to pos_weight in the loss)
# ---------------------------------------------------------------------------


def make_weighted_sampler(dataset: KOIDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws balanced batches.

    Each sample is assigned a weight inversely proportional to its class
    frequency, so on average each batch will contain equal numbers of planets
    and false positives regardless of the true class ratio.

    This is an *alternative* to passing pos_weight to BCEWithLogitsLoss.
    Both approaches address class imbalance; use one or the other, not both.
    """
    labels = dataset.get_labels()
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Dataset contains only one class — cannot build a weighted sampler.")

    weights = np.where(labels == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float32)

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------


def make_splits(
    manifest_path: Path = MANIFEST_FILE,
    manifest_df: pd.DataFrame | None = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the manifest into train / val / test DataFrames, partitioned by star.

    Strategy
    --------
    1.  Compute a star-level label: 1 if the star has at least one planet or
        candidate, 0 if all its candidates are false positives.
    2.  Stratified-split those star IDs 70 / 15 / 15 (default).
    3.  Assign every candidate to whichever split its host star ended up in.

    Args:
        manifest_path:  Path to manifest.csv written by preprocess.py.
                        Ignored when manifest_df is provided.
        manifest_df:    Optional pre-filtered DataFrame to use instead of
                        reading from disk (used by build_dataset.py --mission).
        val_frac:       Fraction of all data to reserve for validation.
        test_frac:      Fraction of all data to reserve for testing.
        random_state:   Seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) — DataFrames ready to pass into KOIDataset.
    """
    manifest = manifest_df if manifest_df is not None else pd.read_csv(manifest_path)

    # --- Star-level labels ------------------------------------------------
    star_labels = manifest.groupby("id")["label"].max()
    stars   = star_labels.index.to_numpy()
    slabels = star_labels.to_numpy()

    # --- Split 1: carve out the test set ----------------------------------
    train_val_stars, test_stars = train_test_split(
        stars,
        test_size=test_frac,
        stratify=slabels,
        random_state=random_state,
    )
    train_val_labels = star_labels.loc[train_val_stars].to_numpy()

    # --- Split 2: carve out the validation set from the remainder ---------
    val_frac_adj = val_frac / (1.0 - test_frac)
    train_stars, val_stars = train_test_split(
        train_val_stars,
        test_size=val_frac_adj,
        stratify=train_val_labels,
        random_state=random_state,
    )

    # --- Assign candidates to splits -------------------------------------
    train_df = manifest[manifest["id"].isin(train_stars)].reset_index(drop=True)
    val_df   = manifest[manifest["id"].isin(val_stars)  ].reset_index(drop=True)
    test_df  = manifest[manifest["id"].isin(test_stars) ].reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Split persistence
# ---------------------------------------------------------------------------


def save_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    datasets_dir: Path = DATASETS_DIR,
) -> None:
    """
    Save train / val / test split DataFrames to CSV files in *datasets_dir*.

    Written by scripts/build_dataset.py; loaded by scripts/train.py via
    load_splits().  Persisting the splits guarantees that every training run
    uses identical partitions without re-running the split logic.
    """
    datasets_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(datasets_dir / "train.csv", index=False)
    val_df.to_csv(datasets_dir / "val.csv",     index=False)
    test_df.to_csv(datasets_dir / "test.csv",   index=False)


def load_splits(
    datasets_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-saved train / val / test split DataFrames from *datasets_dir*.

    Args:
        datasets_dir: Path to the named dataset directory, e.g.
                      data/datasets/full_dataset/.  Must contain
                      train.csv, val.csv, and test.csv.

    Raises FileNotFoundError if any of the split CSVs do not exist — run
    scripts/build_dataset.py --name <name> first.
    """
    for fname in ["train.csv", "val.csv", "test.csv"]:
        path = datasets_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{fname} not found at {path}\n"
                "Run scripts/build_dataset.py --name <name> first to generate the splits."
            )
    train_df = pd.read_csv(datasets_dir / "train.csv")
    val_df   = pd.read_csv(datasets_dir / "val.csv")
    test_df  = pd.read_csv(datasets_dir / "test.csv")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def make_loaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap split DataFrames into DataLoaders ready for the training loop.

    Args:
        use_weighted_sampler:
            If True, draw balanced batches via WeightedRandomSampler.
            If False (default), rely on pos_weight in BCEWithLogitsLoss instead.
            Do not set both True at the same time.
        num_workers:
            Number of worker processes for data loading.  Set to 0 on
            systems where multiprocessing causes issues (e.g. some macOS configs).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = KOIDataset(train_df)
    val_ds   = KOIDataset(val_df)
    test_ds  = KOIDataset(test_df)

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
