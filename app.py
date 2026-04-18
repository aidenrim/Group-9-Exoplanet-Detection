import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models.predict import predict_target, load_model
from src.models.regression import estimate_period_bls
from src.data.dataset_builder import get_segments
from lightkurve import search_lightcurve

# Used Claude code to write the streamlit frontend.
# Allows user to input a Kepler or TESS star ID, runs the prediction pipeline, and displays results including:
# - Overall classification (exoplanet candidate or not)
# - Confidence score
# - Estimated orbital period (if candidate)
# - Plots of the light curve and per-segment probabilities



st.set_page_config(page_title="Exoplanet Detection", layout="centered")

st.title("Exoplanet Detection")
st.write("Enter a Kepler or TESS star ID to classify it as an exoplanet candidate.")

# Sidebar: model selection
models_dir = Path("models")
available_models = [p.stem for p in models_dir.glob("*.pt")] if models_dir.exists() else []

if not available_models:
    st.error("No trained models found in the `models/` directory. Train a model first.")
    st.stop()

model_name = st.sidebar.selectbox("Model", available_models, index=0)
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

# Input
target = st.text_input(
    "Star ID",
    placeholder="e.g. Kepler-442, TIC 260647166"
)

if st.button("Run Detection") and target.strip():
    with st.spinner("Downloading and processing light curve..."):
        try:
            result = predict_target(model_name, target.strip(), threshold=threshold)
        except Exception as e:
            st.error(f"Failed to process target: {e}")
            st.stop()

    # Result banner
    label = "Exoplanet Candidate" if result["prediction"] == 1 else "Not an Exoplanet"
    color = "green" if result["prediction"] == 1 else "red"
    st.markdown(f"### Result: :{color}[{label}]")

    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{result['confidence']:.1%}")
    col2.metric("Segments analysed", result["num_segments"])

    # Orbital period estimate (BLS)
    if result["prediction"] == 1:
        with st.spinner("Estimating orbital period via BLS periodogram..."):
            try:
                mission = "Kepler" if "Kepler" in target else "TESS"
                sr = search_lightcurve(target.strip(), mission=mission)
                lc_raw = sr.download_all().stitch().remove_nans().flatten()
                period = estimate_period_bls(lc_raw)
                if period is not None:
                    st.metric("Estimated orbital period", f"{period:.2f} days")
            except Exception:
                pass

    # Light curve plot
    with st.spinner("Plotting light curve..."):
        try:
            mission = "Kepler" if "Kepler" in target else "TESS"
            search_result = search_lightcurve(target.strip(), mission=mission)
            lc_collection = search_result.download_all()
            lc = lc_collection.stitch().remove_nans().flatten()

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(lc.time.value, lc.flux.value, lw=0.5, color="steelblue")
            ax.set_xlabel("Time (BTJD days)")
            ax.set_ylabel("Normalised Flux")
            ax.set_title(f"Light curve — {target.strip()}")
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.info("Could not plot light curve (data may already be cached from prediction).")

    # Segment-level probabilities
    with st.spinner("Computing per-segment probabilities..."):
        try:
            import torch
            segments = get_segments(target.strip())
            model = load_model(Path("models") / f"{model_name}.pt")
            X = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                probs = torch.sigmoid(model(X)).squeeze().numpy()

            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.bar(range(len(probs)), probs, color="steelblue", alpha=0.7)
            ax2.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
            ax2.set_xlabel("Segment index")
            ax2.set_ylabel("Exoplanet probability")
            ax2.set_title("Per-segment confidence scores")
            ax2.legend()
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception:
            pass
