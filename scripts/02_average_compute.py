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
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ==============================================================================
# Setup & Configuration
# ==============================================================================
OUTPUT_DIR = "docs/assets/average_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEAD_DAYS = 8
WINDOWS = [1, 3, 5, 10, 15]
VARIABLES = [('maxt', 'Max Temp'), ('mint', 'Min Temp'), ('dpt', '21z Dewpoint')]

all_files = sorted(glob.glob("data_cache/diff_*.nc"), reverse=True)

if not all_files:
    raise FileNotFoundError("No NetCDF files found in data_cache/. Run 01_daily_compute.py first.")

# Extract the URMA run date from the latest file
latest_file_date_str = os.path.basename(all_files[0]).replace('diff_', '').replace('.nc', '')
# Calculate the previous day's date for MaxT and Dewpoint
prev_day_date = pd.to_datetime(latest_file_date_str) - pd.Timedelta(days=1)
prev_day_date_str = prev_day_date.strftime('%Y-%m-%d')

print(f"Most recent dataset date: {latest_file_date_str}")

# FIX: Load only the max required files into memory ONCE
max_window = max(WINDOWS)
recent_files = all_files[:max_window]
datasets = []

for fp in recent_files:
    try:
        with xr.open_dataset(fp) as ds:
            datasets.append(ds.load())
    except Exception as e:
        print(f"Warning: Failed to read {fp}: {e}")

if not datasets:
    raise RuntimeError("No valid NetCDF datasets could be loaded.")

lats = datasets[0].latitude.values
lons = datasets[0].longitude.values

# ==============================================================================
# Initialization for in-place plotting
# ==============================================================================
# FIX: Initialize Cartopy figure and map features ONCE to save memory and time
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6, edgecolor='black')
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='gray')
ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree()) 

dummy_grid = np.zeros_like(lats, dtype=np.float32)
im = ax.pcolormesh(
    lons, lats, dummy_grid, 
    transform=ccrs.PlateCarree(), 
    cmap='bwr', 
    vmin=-12, 
    vmax=12,
    shading='auto'
)

# FIX: extend='both' correctly flags >12 and <-12 outliers for forecasters
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, extend='both')
title_obj = ax.set_title("", fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()

# ==============================================================================
# Processing
# ==============================================================================
for var_key, var_label in VARIABLES:
    
    if var_key == 'mint':
        display_date_str = latest_file_date_str
    else:
        display_date_str = prev_day_date_str
        
    cbar.set_label(f"Average {var_label} Bias (°F) [NBM - URMA]", fontsize=10)
        
    for lead in range(1, MAX_LEAD_DAYS + 1):
        nc_var_name = f"lead_{lead}_{var_key}"
        
        for window in WINDOWS:
            window_ds = datasets[:window]
            
            # FIX: Filter out completely empty/failed NaN grids from the average
            valid_grids = [
                ds[nc_var_name].values 
                for ds in window_ds 
                if nc_var_name in ds.data_vars and not np.all(np.isnan(ds[nc_var_name].values))
            ]

            if not valid_grids:
                print(f"  [!] Skipping {nc_var_name} ({window}d window): 0 valid samples.")
                continue

            stack_array = np.stack(valid_grids, axis=0)
            avg_bias = np.nanmean(stack_array, axis=0)
            
            # FIX: Update plot in-place rather than rebuilding it
            im.set_array(avg_bias.ravel())
            title_obj.set_text(
                f"NBM Day {lead} {var_label} Average Bias (°F)\n"
                f"{window}-Day Window Valid Through {display_date_str} (N={len(valid_grids)})"
            )

            out_filename = f"day{lead}_{window}day_{var_key}.png"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            
print("\nAll average bias plots successfully generated!")
plt.close(fig)

# ==============================================================================
# Export Metadata for Dashboard Update Time
# ==============================================================================
metadata = {
    "last_updated_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
    "latest_dataset_date": latest_file_date_str
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print("Exported metadata.json for dashboard.")
