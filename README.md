Group 9 Exoplanet Detection

Members: Ayush Adhikari, Nikhil Khot, Arha Yalamanchili, Aiden Rim

# Exoplanet Transit Classifier

A 1D convolutional neural network that classifies Kepler and TESS lightcurves as indicating the presence of an exoplanet or not.

---

## Overview

The pipeline takes raw photometric lightcurves from NASA's Kepler and TESS missions and produces a binary classification. Each lightcurve is phase-folded on its catalog period and binned into two fixed-length views fed to a dual-branch CNN:

- **Global view** (201 bins) — full orbital phase;
- **Local view** (61 bins) — local transit window;

---

## Project Structure

```
├── app.py                      # Streamlit app
├── scripts/
│   ├── download.py             # Download catalogs and FITS lightcurves
│   ├── preprocess.py           # Build .npz arrays and manifest.csv
│   ├── build_dataset.py        # Create train/val/test splits
│   ├── train.py                # Train the CNN
│   ├── evaluate.py             # Evaluate a model on a test split
│   ├── compare_models.py       # Plot grouped bar chart across models
│   └── predict.py              # Classify a single candidate from the CLI
├── src/
│   ├── data/
│   │   ├── download.py         # Download logic (Kepler + TESS)
│   │   ├── preprocess.py       # Detrending, phase-folding, binning
│   │   └── dataset.py          # Dataset, split utilities, DataLoaders
│   └── models/
│       ├── model.py            # ExoplanetCNN architecture
│       ├── train.py            # Training loop, evaluation, plotting
│       ├── predict.py          # Single-candidate inference library
│       └── regression.py       # BLS period estimation
├── data/
│   ├── catalogs/               # koi_cumulative.csv, toi_catalog.csv
│   ├── raw/                    # Stitched FITS files (kic_*.fits, tic_*.fits)
│   ├── processed/              # Phase-folded .npz arrays
│   └── datasets/               # manifest.csv and named split directories
└── results/                    # One sub-folder per training run
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Full Pipeline

### 1. Download

Downloads the KOI/TOI catalog and stitched FITS lightcurves from the MAST archive.

```bash
# Kepler only (default)
python scripts/download.py

# TESS only
python scripts/download.py --mission tess

# Both missions
python scripts/download.py --mission both

# Limit to first 20 stars
python scripts/download.py --mission both --max-stars 20
```

Note: Downloading the full catalog will take several hours. If needed, you can stop and continue downloading later.
The script will recognize which lightcurves are already downloaded and pick up with the ones that haven't been downloaded.
Note 2: Sometimes downloads are interrupted and fail, leaving a corrupted file in the cache. Rerunning the script will remove corrupted files from the cache and attempt to redownload. You may need to run the script multiple times to download the full catalog.

### 2. Preprocess

Detrends each lightcurve, phase-folds on the catalog period, bins into global and local views, and writes `.npz` files and `data/datasets/manifest.csv`.

```bash
python scripts/preprocess.py                    # Kepler (default)
python scripts/preprocess.py --mission tess
python scripts/preprocess.py --mission both
python scripts/preprocess.py --max-stars 20     # Process only the first 20 stars
python scripts/preprocess.py --force            # reprocess existing files
```

### 3. Build Dataset

Splits the manifest into train / val / test partitions. Defaults to a 70 / 15 / 15 split, but can be specified.

```bash
python scripts/build_dataset.py --name full_dataset

# Kepler or TESS only
python scripts/build_dataset.py --name kepler_only --mission kepler
python scripts/build_dataset.py --name tess_v1     --mission tess

# Include CANDIDATE dispositionss as positive examples (CONFIRMED planets). Default is exclude
python scripts/build_dataset.py --name confirmed_only --candidates include

# Custom split fractions
python scripts/build_dataset.py --name custom --val-frac 0.10 --test-frac 0.10
```

Note: In the manifest, CONFIRMED planets have `label=1`, FALSE POSITIVEs have `label=0`, and CANDIDATEs have `label=-1`. The `--candidates` flag controls how candidates are handled at split time:
- `include` — remapped to `label=1`, treated as confirmed planets
- `exclude` (default) — dropped from the dataset entirely

Candidates (especially from the Kepler mission) have already undergone some manual vetting, so it may be of interest to treat them as positive examples while training.

### 4. Train

Trains a model on a specified dataset. The name of the dataset should be the name of the appropriate subfolder in `data/datasets` witht the train/val/test splits.

```bash
python scripts/train.py --name model_v1 --dataset full_dataset

# Specify training options
python scripts/train.py --name model_v1 --dataset full_dataset --epochs 100 --lr 1e-4 --batch-size 32 --dropout 0.5 --patience 15
```

Outputs written to `results/model_v1/`:
- `best_model.pt` — best checkpoint (by val AUC)
- `training_curves.png`
- `roc_curve.png`
- `confusion_matrix.png`

### 5. Evaluate

Runs the best checkpoint on the test split and saves metrics to `results/{model}/eval.json`. The model must already be trained to run this script

```bash
python scripts/evaluate.py --model model_v1 --dataset full_dataset
```

`eval.json` contains: accuracy, precision, recall, F1, AUC-ROC, threshold, epoch.

### 6. Compare Models

Plots a grouped bar chart (one bar per model, grouped by metric) across precision, accuracy, F1, and AUC-ROC.

```bash
python scripts/compare_models.py --models model_v1 model_v2 model_v3
```

### 7. Predict a Single Candidate (CLI)

```bash
python scripts/predict.py --model model_v1 --candidate K00010.01
python scripts/predict.py --model model_v1 --candidate TOI-103.01
```

---

## Interactive App

```bash
streamlit run app.py
```

Enter a KOI name (e.g. `K00010.01`) or TOI name (e.g. `TOI-103.01`) to:
- See the planet probability and prediction at the chosen threshold
- View phase-folded global and local lightcurve plots
- Run BLS period estimation and compare the predicted period to the catalog value

---

## Manifest Schema

The file `data/datasets/manifest.csv` contains mission-agnostic data:

| Column | Description |
|---|---|
| `id` | Star ID (Kepler KIC or TESS TIC) |
| `name` | Candidate name (`K00010.01` or `TOI-103.01`) |
| `disposition` | `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE` |
| `period` | Orbital period in days |
| `time0bk` | Transit epoch (BKJD for Kepler, BTJD for TESS) |
| `duration` | Transit duration in hours |
| `label` | `1` = confirmed, `-1` = candidate, `0` = false positive |
| `mission` | `KEPLER` or `TESS` |
| `path` | Relative path to the `.npz` file |

---
