from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Paths:
    root: Path = Path(".")
    data_dir: Path = Path("data")
    models_dir: Path = Path("artifacts")
    cache_dir: Path = Path("data/cache")

    def resolve(self) -> "Paths":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(slots=True)
class TrainingConfig:
    bins: int = 512
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    test_size: float = 0.2
    random_state: int = 42

