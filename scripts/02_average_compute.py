import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Setup & Configuration
# ==============================================================================
OUTPUT_DIR = "docs/assets/average_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEAD_DAYS = 8
WINDOWS = [1, 3, 5, 10, 15]
VARIABLES = [('maxt', 'Max T'), ('mint', 'Min T')]

all_files = sorted(glob.glob("data_cache/diff_*.nc"), reverse=True)

if not all_files:
    raise FileNotFoundError("No NetCDF files found in data_cache/. Run 01_daily_compute.py first.")

latest_date_str = os.path.basename(all_files[0]).replace('diff_', '').replace('.nc', '')
print(f"Most recent dataset date: {latest_date_str}")

with xr.open_dataset(all_files[0]) as ds_ref:
    lats = ds_ref.latitude.values
    lons = ds_ref.longitude.values

# ==============================================================================
# Plotting Helper
# ==============================================================================
def plot_bias_map(avg_bias_grid, lead_day, window_days, n_samples, end_date_str, var_label, output_path):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='gray')
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree()) 
    
    im = ax.pcolormesh(
        lons, lats, avg_bias_grid, 
        transform=ccrs.PlateCarree(), 
        cmap='bwr', 
        vmin=-12, 
        vmax=12
    )
    
    lead_hours = lead_day * 24
    ax.set_title(
        f"NBM Day {lead_day} (+{lead_hours}h) {var_label} Average Bias (°F)\n"
        f"{window_days}-Day Window Ending {end_date_str} (N={n_samples})",
        fontsize=13, fontweight='bold', pad=10
    )
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(f"Average {var_label} Bias (°F) [NBM - URMA]", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# Processing
# ==============================================================================
for window in WINDOWS:
    selected_files = all_files[:window]
    actual_count = len(selected_files)
    
    if actual_count == 0:
        continue

    print(f"\n--- Calculating {window}-Day Average (Using {actual_count} file(s)) ---")

    for var_key, var_label in VARIABLES:
        for lead in range(1, MAX_LEAD_DAYS + 1):
            nc_var_name = f"lead_{lead}_{var_key}"
            grid_stack = []

            for filepath in selected_files:
                try:
                    with xr.open_dataset(filepath) as ds:
                        if nc_var_name in ds.data_vars:
                            grid_stack.append(ds[nc_var_name].values)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue

            if not grid_stack:
                print(f"  [!] No valid data for {var_label} Lead Day {lead}")
                continue

            stack_array = np.array(grid_stack)
            avg_bias = np.nanmean(stack_array, axis=0)

            out_filename = f"day{lead}_{window}day_{var_key}.png"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            plot_bias_map(
                avg_bias_grid=avg_bias,
                lead_day=lead,
                window_days=window,
                n_samples=len(grid_stack),
                end_date_str=latest_date_str,
                var_label=var_label,
                output_path=out_path
            )
            print(f"  ==> Saved: {out_filename}")

print("\nAll average bias plots successfully generated!")
