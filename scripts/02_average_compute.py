import os
import glob
import pandas as pd
import numpy as np
import xarray as xr

# Use non-interactive backend for headless GitHub Actions runner
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
WINDOWS = [5, 10, 15]

# Find and sort all cached NetCDF files by date (latest first)
all_files = sorted(glob.glob("data_cache/diff_*.nc"), reverse=True)

if not all_files:
    raise FileNotFoundError("No NetCDF files found in data_cache/. Run 01_daily_compute.py first.")

print(f"Found {len(all_files)} cached file(s) for bias averaging.")

# Extract the date range available
latest_date_str = os.path.basename(all_files[0]).replace('diff_', '').replace('.nc', '')
print(f"Most recent dataset date: {latest_date_str}")

# Load grid coordinates from the latest file
with xr.open_dataset(all_files[0]) as ds_ref:
    lats = ds_ref.latitude.values
    lons = ds_ref.longitude.values

# ==============================================================================
# Helper Function: Plotting Bias Map
# ==============================================================================
def plot_bias_map(avg_bias_grid, lead_day, window_days, n_samples, end_date_str, output_path):
    fig = plt.figure(figsize=(12, 6))
    
    # Standard PlateCarree map projection centered on CONUS
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add geographical features
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='gray')
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree()) 
    
    # Plot divergence mesh (Centered at 0°F)
    im = ax.pcolormesh(
        lons, lats, avg_bias_grid, 
        transform=ccrs.PlateCarree(), 
        cmap='bwr', 
        vmin=-12, 
        vmax=12
    )
    
    # Titles and formatting
    lead_hours = lead_day * 24
    ax.set_title(
        f"NBM Day {lead_day} (+{lead_hours}h) Max T Average Bias (°F)\n"
        f"{window_days}-Day Rolling Average Ending {end_date_str} (N={n_samples})",
        fontsize=13, fontweight='bold', pad=10
    )
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label("Average Temperature Bias (°F) [NBM Forecast - URMA Observed]", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# Processing Averages
# ==============================================================================
for window in WINDOWS:
    # Get the files within this rolling window size
    selected_files = all_files[:window]
    actual_count = len(selected_files)
    
    if actual_count == 0:
        continue

    print(f"\n--- Calculating {window}-Day Average (Using {actual_count} file(s)) ---")

    for lead in range(1, MAX_LEAD_DAYS + 1):
        lead_key = f"lead_{lead}"
        grid_stack = []

        # Stack grids for this specific lead time across all files in the window
        for filepath in selected_files:
            try:
                with xr.open_dataset(filepath) as ds:
                    if lead_key in ds.data_vars:
                        grid_stack.append(ds[lead_key].values)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

        if not grid_stack:
            print(f"  [!] No valid data for Lead Day {lead}")
            continue

        # Convert stack to 3D numpy array and calculate mean ignoring NaNs
        stack_array = np.array(grid_stack)
        avg_bias = np.nanmean(stack_array, axis=0)

        # File output path (e.g., docs/assets/average_plots/day1_5day.png)
        out_filename = f"day{lead}_{window}day.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        # Generate plot
        plot_bias_map(
            avg_bias_grid=avg_bias,
            lead_day=lead,
            window_days=window,
            n_samples=len(grid_stack),
            end_date_str=latest_date_str,
            output_path=out_path
        )
        print(f"  ==> Saved: {out_filename}")

print("\nAll average bias plots successfully generated!")
