import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.predict import load_model, predict_koi

ROOT         = Path(__file__).resolve().parent
CATALOG_FILE = ROOT / "data" / "catalogs" / "koi_cumulative.csv"
RESULTS_DIR  = ROOT / "results"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Exoplanet CNN Classifier", layout="centered")
st.title("Exoplanet CNN Classifier")
st.write(
    "Enter a KOI name to classify it as a planet candidate or false positive "
    "using the trained dual-branch lightcurve CNN."
)

# ---------------------------------------------------------------------------
# Sidebar — model selection and threshold
# ---------------------------------------------------------------------------

available_models = sorted([
    p.name for p in RESULTS_DIR.iterdir()
    if p.is_dir() and (p / "best_model.pt").exists()
]) if RESULTS_DIR.exists() else []

if not available_models:
    st.error("No trained models found in results/. Run scripts/train.py first.")
    st.stop()

model_name = st.sidebar.selectbox("Model", available_models, index=0)

# Load checkpoint metadata to populate the threshold slider default.
_, meta = load_model(RESULTS_DIR / model_name / "best_model.pt")

threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, float(meta["threshold"]), 0.01
)
st.sidebar.caption(
    f"Epoch {meta['epoch']} · val AUC {meta['val_auc']:.4f}"
)

# ---------------------------------------------------------------------------
# Main — KOI input and classification
# ---------------------------------------------------------------------------

kepoi_name = st.text_input("KOI Name", placeholder="e.g. K00010.01, K00113.01")

if st.button("Run Classification") and kepoi_name.strip():
    with st.spinner("Running classification..."):
        try:
            result = predict_koi(model_name, kepoi_name.strip(), threshold=threshold)
        except KeyError as exc:
            st.error(str(exc))
            st.stop()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    # --- result banner ----------------------------------------------------
    label = "Planet Candidate" if result["prediction"] == 1 else "False Positive"
    color = "green"            if result["prediction"] == 1 else "red"
    st.markdown(f"### Result: :{color}[{label}]")

    # --- metrics row ------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence",  f"{result['confidence']:.1%}")
    col2.metric("Probability", f"{result['probability']:.4f}")
    col3.metric("Threshold",   f"{result['threshold']:.2f}")

    # --- known disposition ------------------------------------------------
    known_map = {1: "CONFIRMED / CANDIDATE", 0: "FALSE POSITIVE"}
    known_str = known_map.get(result["known_label"], "UNKNOWN")
    st.caption(
        f"Known disposition: **{result['koi_disposition']}** ({known_str}) "
        f"· Kepler ID: {result['kepid']}"
    )

    # --- orbital period from catalog -------------------------------------
    try:
        catalog = pd.read_csv(CATALOG_FILE)
        cat_matches = catalog[catalog["kepoi_name"] == kepoi_name.strip()]
        if not cat_matches.empty:
            cat_row = cat_matches.iloc[0]
            period   = cat_row["koi_period"]
            duration = cat_row["koi_duration"]
            p_col, d_col = st.columns(2)
            if not math.isnan(period):
                p_col.metric("Orbital period",   f"{period:.4f} days")
            if not math.isnan(duration):
                d_col.metric("Transit duration", f"{duration:.2f} hours")
    except Exception:
        pass

    # --- lightcurve plots -------------------------------------------------
    st.subheader("Phase-folded lightcurve views")

    # Global view — full orbital phase
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    phase_bins = np.linspace(-0.5, 0.5, len(result["global_view"]))
    ax1.plot(phase_bins, result["global_view"], lw=1.0, color="steelblue")
    ax1.axhline(0, color="gray", linestyle="--", lw=0.5, alpha=0.5)
    ax1.set_xlabel("Orbital phase")
    ax1.set_ylabel("Relative flux (baseline = 0)")
    ax1.set_title(f"Global view — {kepoi_name.strip()}  (201 phase bins)")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    # Local view — zoomed transit window
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    local_bins = np.linspace(-1, 1, len(result["local_view"]))
    ax2.plot(local_bins, result["local_view"], lw=1.0, color="darkorange")
    ax2.axhline(0, color="gray", linestyle="--", lw=0.5, alpha=0.5)
    ax2.set_xlabel("Phase (transit-centered, ±2 transit durations = ±1)")
    ax2.set_ylabel("Relative flux")
    ax2.set_title(f"Local view — {kepoi_name.strip()}  (61 phase bins, transit window)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)
