import argparse
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Observations
from astropy.io import fits
import sys

def fetch_and_plot(target_name):
    try:
        obs_table = Observations.query_object(target_name, radius="0.001 deg")
    except Exception as e:
        print(f"Error resolving target name: {e}")
        return
    
    kepler_obs = obs_table[obs_table['obs_collection'] == 'Kepler']
    
    if len(kepler_obs) == 0:
        print(f"No data found for '{target_name}'.")
        return

    data_products = Observations.get_product_list(kepler_obs[0])
    lc_files = Observations.filter_products(data_products, 
                                            productSubGroupDescription='LLC', 
                                            extension='fits')
    if len(lc_files) == 0:
        print("Target has no light.")
        return
    manifest = Observations.download_products(lc_files[0], mrp_only=False)
    file_path = manifest['Local Path'][0]
    
    with fits.open(file_path) as hdul:
        time = np.array(hdul[1].data['TIME'], dtype=np.float64)
        flux = np.array(hdul[1].data['PDCSAP_FLUX'], dtype=np.float64)

    mask = np.isfinite(time) & np.isfinite(flux)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time[mask], flux[mask], 'k.', markersize=1)
    plt.title(f"Kepler Light Curve: {target_name}")
    plt.xlabel("Time (Barycentric Kepler Julian Day)")
    plt.ylabel("Flux (e-/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("Displaying plot. Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Kepler Light Curves from the terminal.")
    parser.add_argument("target", type=str, help="Name of the planetary system or star (e.g., 'Kepler-10' or 'KIC 11904151')")
    
    args = parser.parse_args()
    
    try:
        fetch_and_plot(args.target)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)