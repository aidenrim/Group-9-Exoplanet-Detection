import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Observations
from astropy.io import fits
from exoplanet_research import fetch_and_plot

st.set_page_config(page_title="Kepler Light Curve Explorer", layout="wide")

st.title("🔭 Kepler Light Curve Explorer")
st.markdown("""
    Enter the name of a planetary body (e.g., **Kepler-10**, **KIC 11904151**) to visualize its light curve 
    and its sonified soundwave representation. Here is a list of bodies: 
    https://en.wikipedia.org/wiki/List_of_exoplanets_discovered_by_the_Kepler_space_telescope#Lists
            
    The radius is set to 0.001 degrees.
    """)

target = st.sidebar.text_input("Target Name", value="Kepler-10")

if st.sidebar.button("Fetch Data"):
    placeholder = st.empty()
    with st.spinner(f"Accessing MAST Archive for {target}..."):
        try:
            result = fetch_and_plot(target)
            if result is None:
                st.error(f"No data was found for {target}")
            else: 
                time, flux = fetch_and_plot(target)
                mask = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[mask], flux[mask]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Science Light Curve")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(time, flux, 'k.', markersize=0.5)
                    ax1.set_xlabel("Time in BKJD")
                    ax1.set_ylabel("Flux (e-/s)")
                    st.pyplot(fig1)
                with col2:
                    st.subheader("Soundwave Projection")
                    signal = (flux - np.mean(flux)) / np.max(np.abs(flux - np.mean(flux)))
                    fig2, ax2 = plt.subplots()
                    ax2.plot(time, signal, color='crimson', lw=0.5)
                    ax2.fill_between(time,signal,color="crimson",alpha=0.3)
                    ax2.set_xlabel("Time in BKJD")
                    ax2.set_ylabel("Amplitude")
                    st.pyplot(fig2)                
                st.success (f"Data for {target}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("Click 'Fetch Data' in the sidebar to begin.")