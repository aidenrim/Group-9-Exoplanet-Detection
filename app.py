import math
import sys
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.predict import load_model, predict_single_oi
from src.models.regression import estimate_period_bls

ROOT        = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Exoplanet CNN Classifier", layout="centered")
st.title("Exoplanet CNN Classifier")
st.write(
    "Enter a KOI or TOI name to classify it as a planet candidate or false positive "
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
# Main — Object of interest input and classification
# ---------------------------------------------------------------------------

name = st.text_input("KOI / TOI Name", placeholder="e.g. K00010.01  or  TOI-103.01")

if st.button("Run Classification") and name.strip():
    with st.spinner("Running classification..."):
        try:
            st.session_state["result"]     = predict_single_oi(model_name, name.strip(), threshold=threshold)
            st.session_state["name"] = name.strip()
        except KeyError as exc:
            st.error(str(exc))
            st.stop()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

if "result" in st.session_state:
    result     = st.session_state["result"]
    name = st.session_state["name"]

    # --- result banner ----------------------------------------------------
    label = "Planet Predicted" if result["prediction"] == 1 else "Not a Planet Predicted"
    color = "green"            if result["prediction"] == 1 else "red"
    st.markdown(f"### Result: :{color}[{label}]")

    # --- metrics row ------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Confidence in prediction",  f"{result['confidence']:.1%}")
    col2.metric("Probability of planet", f"{result['probability']:.4f}")
    col3.metric("Threshold",   f"{result['threshold']:.2f}")

    # --- known disposition ------------------------------------------------
    is_tess = name.upper().startswith("TOI")
    star_label = "TIC ID" if is_tess else "Kepler ID"
    disp_str = (
        "Planet" if result["disposition"] == "CONFIRMED"
        else "Not a Planet" if result["disposition"] == "FALSE POSITIVE"
        else "Unconfirmed (Candidate)"
    )
    st.caption(f"Known disposition: **{disp_str}** · {star_label}: {result['id']}")

    # --- lightcurve plots -------------------------------------------------
    st.subheader("Phase-folded lightcurve views")

    # Global view — full orbital phase
    global_line = st.checkbox("Show global view as line", value=False)
    global_kwargs = dict(lw=1.0) if global_line else dict(marker='.', linestyle='none')
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    phase_bins = np.linspace(-0.5, 0.5, len(result["global_view"]))
    ax1.plot(phase_bins, result["global_view"], color="steelblue", **global_kwargs)
    ax1.axhline(0, color="gray", linestyle="--", lw=0.5, alpha=0.5)
    ax1.set_xlabel("Orbital phase")
    ax1.set_ylabel("Relative flux (baseline = 0)")
    ax1.set_title(f"Global view — {name}  (201 phase bins)")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

    # Local view — zoomed transit window
    local_line = st.checkbox("Show local view as line", value=False)
    local_kwargs = dict(lw=1.0) if local_line else dict(marker='.', linestyle='none')
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    local_bins = np.linspace(-1, 1, len(result["local_view"]))
    ax2.plot(local_bins, result["local_view"], color="darkorange", **local_kwargs)
    ax2.axhline(0, color="gray", linestyle="--", lw=0.5, alpha=0.5)
    ax2.set_xlabel("Phase (transit-centered, ±2 transit durations = ±1)")
    ax2.set_ylabel("Relative flux")
    ax2.set_title(f"Local view — {name}  (61 phase bins, transit window)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

    # --- BLS period estimation -------------------------------------------
    star_id   = result["id"]
    fits_path = (
        ROOT / "data" / "raw" / f"tic_{star_id:010d}.fits"
        if is_tess else
        ROOT / "data" / "raw" / f"kic_{star_id:09d}.fits"
    )

    if fits_path.exists():
        st.subheader("BLS period estimation")
        with st.spinner("Running Box Least Squares…"):
            lc_raw = lk.read(str(fits_path))
            bls = estimate_period_bls(lc_raw)

        if bls is None:
            st.warning("BLS failed: lightcurve too short or degenerate after detrending.")
        else:
            catalog_period   = result.get("period", float("nan"))
            catalog_duration = result.get("duration", float("nan"))
            period_col1, period_col2, period_col3 = st.columns(3)
            period_col1.metric("Estimated period", f"{bls['best_period']:.4f} days")
            period_col2.metric("Catalog period",   f"{catalog_period:.4f} days")
            period_pct_err = abs(bls["best_period"] - catalog_period) / catalog_period * 100
            period_col3.metric("Error vs catalog", f"{period_pct_err:.2f} %")
            
            duration_col1, duration_col2, duration_col3 = st.columns(3)
            duration_col1.metric("Estimated duration", f"{bls['best_duration']:.2f} hours")
            duration_col2.metric("Catalog duration",   f"{catalog_duration:.4f} days")
            duration_pct_err = abs(bls["best_duration"] - catalog_duration) / catalog_duration * 100
            duration_col3.metric("Error vs catalog", f"{duration_pct_err:.2f} %")

            # Periodogram plot
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.plot(bls["periods"], bls["power"], lw=0.7, color="steelblue")
            ax3.axvline(bls["best_period"], color="red",  lw=1.2, linestyle="--",
                        label=f"Best period {bls['best_period']:.4f} d")
            if not math.isnan(catalog_period):
                ax3.axvline(catalog_period, color="green", lw=1.2, linestyle=":",
                            label=f"Catalog period {catalog_period:.4f} d")
            ax3.set_xscale("log")
            ax3.set_xlabel("Period (days)")
            ax3.set_ylabel("BLS power")
            ax3.set_title(f"BLS periodogram — {name}")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)

