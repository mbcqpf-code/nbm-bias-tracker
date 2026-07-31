import xarray as xr
import pandas as pd
import numpy as np
import requests
import os
import glob
from herbie import Herbie
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
# Dynamically target "yesterday" since today's URMA Max T isn't available yet
target_end = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
END_DATE = target_end.strftime('%Y-%m-%d')

DAYS_TO_KEEP = 15
MAX_LEAD_DAYS = 8
NBM_CYCLE = "06:00"

# Set up the folder structure
os.makedirs("data_cache", exist_ok=True)
os.makedirs("docs/assets/daily_plots", exist_ok=True)

print(f"--- Starting Daily Collector (Target End Date: {END_DATE}) ---")

# ==============================================================================
# Custom Downloader for NBM QMD
# ==============================================================================
def get_nbm_qmd_tmax(init_dt, lead_hrs):
    date_str = init_dt.strftime('%Y%m%d')
    cycle_str = f"{init_dt.hour:02d}"
    
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{cycle_str}/qmd"
    grib_name = f"blend.t{cycle_str}z.qmd.f{lead_hrs:03d}.co.grib2"
    
    r_idx = requests.get(f"{base_url}/{grib_name}.idx")
    if r_idx.status_code != 200:
        raise Exception("Index file not found")
        
    start_byte, end_byte = None, None
    lines = r_idx.text.splitlines()
    for i, line in enumerate(lines):
        if ":TMP:2 m above ground:" in line and "max fcst" in line and "50% level" in line:
            start_byte = int(line.split(':')[1])
            if i + 1 < len(lines):
                end_byte = int(lines[i+1].split(':')[1]) - 1
            break
            
    if start_byte is None:
        raise Exception("TMP max fcst (50%) not found.")
        
    headers = {"Range": f"bytes={start_byte}-{end_byte}" if end_byte else f"bytes={start_byte}-"}
    r_grib = requests.get(f"{base_url}/{grib_name}", headers=headers)
    
    temp_file = f"nbm_temp_{lead_hrs}h.grib2"
    with open(temp_file, "wb") as f:
        f.write(r_grib.content)
        
    return xr.open_dataset(temp_file, engine='cfgrib'), temp_file

# ==============================================================================
# Main Processing Loop (Smart Backfill)
# ==============================================================================
end_dt = pd.to_datetime(END_DATE)

for d in range(DAYS_TO_KEEP):
    target_dt = end_dt - pd.Timedelta(days=d)
    date_str = target_dt.strftime('%Y-%m-%d')
    nc_filename = f"data_cache/diff_{date_str}.nc"
    
    # 1. Idempotency Check: Does this file already exist?
    if os.path.exists(nc_filename):
        print(f"[{date_str}] -> Already exists. Skipping.")
        continue
        
    print(f"\n[{date_str}] -> Missing. Starting processing...")
    
    # 2. Download URMA (08z cycle captures the past 24h Max T)
    urma_dt = f"{date_str} 08:00"
    try:
        H_urma = Herbie(urma_dt, model="urma", product="anl")
        ds_urma = H_urma.xarray(":TMAX:2 m above ground:", remove_grib=False)
        urma_var = list(ds_urma.data_vars)[0]
        urma_tmax_f = (ds_urma[urma_var] - 273.15) * 9/5 + 32
    except Exception as e:
        print(f"  [!] URMA failed for {urma_dt}: {e}. Skipping day.")
        continue
        
    # Dictionary to hold the 8 layers of difference arrays
    daily_diffs = {}
    
    # 3. Download NBM Leads 1-8
    for lead in range(1, MAX_LEAD_DAYS + 1):
        lead_hours = lead * 24
        init_dt = target_dt - pd.Timedelta(days=lead)
        init_dt_str = pd.to_datetime(f"{init_dt.strftime('%Y-%m-%d')} {NBM_CYCLE}")
        
        try:
            ds_nbm, temp_grib = get_nbm_qmd_tmax(init_dt_str, lead_hours)
            nbm_var = list(ds_nbm.data_vars)[0]
            nbm_tmax_f = (ds_nbm[nbm_var] - 273.15) * 9/5 + 32
            
            diff_array = nbm_tmax_f.values - urma_tmax_f.values
            daily_diffs[f'lead_{lead}'] = (['y', 'x'], diff_array)
            
            ds_nbm.close()
            os.remove(temp_grib)
            for idx_file in glob.glob(f"{temp_grib}*.idx"):
                os.remove(idx_file)
                
            print(f"  -> Day {lead} forecast (+{lead_hours}h) processed.")
            
        except Exception as e:
            print(f"  [!] Day {lead} missing: {e}")
            daily_diffs[f'lead_{lead}'] = (['y', 'x'], np.full(urma_tmax_f.shape, np.nan))

    # 4. Save to NetCDF (With Compression)
    ds_out = xr.Dataset(
        data_vars=daily_diffs,
        coords={
            'latitude': (['y', 'x'], ds_urma.latitude.values),
            'longitude': (['y', 'x'], ds_urma.longitude.values)
        }
    )
    
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_out.data_vars}
    
    ds_out.to_netcdf(nc_filename, encoding=encoding)
    ds_out.close()
    print(f"  ==> Saved {nc_filename} (Compressed)")

# ==============================================================================
# The Janitor (Retention Policy)
# ==============================================================================
print("\n--- Running Janitor ---")
cutoff_date = end_dt - pd.Timedelta(days=DAYS_TO_KEEP)

for nc_file in glob.glob("data_cache/diff_*.nc"):
    # Extract just the filename (e.g., "diff_2026-07-28.nc") ignoring the folder path
    basename = os.path.basename(nc_file)
    # Strip away the prefix and suffix to isolate the date
    file_date_str = basename.replace('diff_', '').replace('.nc', '')
    
    try:
        file_date = pd.to_datetime(file_date_str)
        if file_date <= cutoff_date:
            os.remove(nc_file)
            print(f"Deleted old archive: {nc_file}")
    except Exception as e:
        print(f"Skipping {nc_file}: {e}")
        
print("Daily Collector finished!")
