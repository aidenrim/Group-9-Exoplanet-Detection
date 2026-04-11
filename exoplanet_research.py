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
        print(f"{target_name} is not a recognized planetary body")
        return
    
    kepler_obs = obs_table[obs_table['obs_collection'] == 'Kepler']
    
    if len(kepler_obs) == 0:
        print(f"No data found for '{target_name}'.")
        raise ValueError("Target not found in Kepler data")

    data_products = Observations.get_product_list(kepler_obs[0])
    lc_files = Observations.filter_products(data_products, 
                                            productSubGroupDescription='LLC', 
                                            extension='fits')
    if len(lc_files) == 0:
        print("Target has no light.")
        raise ValueError("Target has no light")
    manifest = Observations.download_products(lc_files[0], mrp_only=False)
    file_path = manifest['Local Path'][0]
    
    with fits.open(file_path) as hdul:
        time = np.array(hdul[1].data['TIME'], dtype=np.float64)
        flux = np.array(hdul[1].data['PDCSAP_FLUX'], dtype=np.float64)
    
    return time, flux

if __name__ == "__main__":
    target = sys.argv[1]
    t, f = fetch_and_plot(target)
    plt.plot(t, f)
    plt.show()