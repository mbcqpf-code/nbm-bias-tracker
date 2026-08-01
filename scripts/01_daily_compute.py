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
target_end = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
END_DATE = target_end.strftime('%Y-%m-%d')

DAYS_TO_KEEP = 15
MAX_LEAD_DAYS = 8
NBM_CYCLE = "06:00"

os.makedirs("data_cache", exist_ok=True)
os.makedirs("docs/assets/average_plots", exist_ok=True)

print(f"--- Starting Daily Collector (Target End Date: {END_DATE}) ---")

# ==============================================================================
# Custom Downloader for NBM QMD (Max, Min, DPT)
# ==============================================================================
def get_nbm_qmd(init_dt, lead_hrs, var_type):
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
        line_lower = line.lower()
        if "50% level" in line_lower and "2 m above ground" in line_lower:
            if var_type == 'max' and "max" in line_lower:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
            elif var_type == 'min' and "min" in line_lower:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
            elif var_type == 'dpt' and ":dpt:" in line_lower:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
            
    if start_byte is None:
        raise Exception(f"{var_type.upper()} (50%) not found in GRIB index.")
        
    headers = {"Range": f"bytes={start_byte}-{end_byte}" if end_byte else f"bytes={start_byte}-"}
    r_grib = requests.get(f"{base_url}/{grib_name}", headers=headers)
    
    temp_file = f"nbm_temp_{var_type}_{lead_hrs}h.grib2"
    with open(temp_file, "wb") as f:
        f.write(r_grib.content)
        
    return xr.open_dataset(temp_file, engine='cfgrib'), temp_file

# ==============================================================================
# Main Processing Loop
# ==============================================================================
end_dt = pd.to_datetime(END_DATE)

for d in range(DAYS_TO_KEEP):
    target_dt = end_dt - pd.Timedelta(days=d)
    date_str = target_dt.strftime('%Y-%m-%d')
    valid_date_str = (target_dt - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    nc_filename = f"data_cache/diff_{date_str}.nc"
    
    if os.path.exists(nc_filename):
        print(f"[{date_str}] -> Already exists. Skipping.")
        continue
        
    print(f"\n[{date_str}] -> Missing. Starting processing...")
    
    urma_maxt_dt = f"{date_str} 08:00"
    urma_mint_dt = f"{date_str} 20:00"
    urma_dpt_dt = f"{valid_date_str} 21:00"
    
    try:
        # THE FIX: We apply .squeeze().values[::2, ::2] to downsample to 5km resolution!
        H_urma_max = Herbie(urma_maxt_dt, model="urma", product="anl")
        ds_urma_max = H_urma_max.xarray(":TMAX:2 m above ground:", remove_grib=False)
        urma_tmax_f = (ds_urma_max[list(ds_urma_max.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
        
        H_urma_min = Herbie(urma_mint_dt, model="urma", product="anl")
        ds_urma_min = H_urma_min.xarray(":TMIN:2 m above ground:", remove_grib=False)
        urma_tmin_f = (ds_urma_min[list(ds_urma_min.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32

        H_urma_dpt = Herbie(urma_dpt_dt, model="urma", product="anl")
        ds_urma_dpt = H_urma_dpt.xarray(":DPT:2 m above ground:", remove_grib=False)
        urma_tdpt_f = (ds_urma_dpt[list(ds_urma_dpt.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
        
        # Subsample the coordinate grids too
        lat_grid = ds_urma_max.latitude.squeeze().values[::2, ::2].astype(np.float32)
        lon_grid = ds_urma_max.longitude.squeeze().values[::2, ::2].astype(np.float32)
        
    except Exception as e:
        print(f"  [!] URMA failed for {date_str}: {e}. Skipping day.")
        continue
        
    daily_diffs = {}
    
    # 3. Download NBM Leads 1-8
    for lead in range(1, MAX_LEAD_DAYS + 1):
        lead_hours_max = lead * 24
        lead_hours_min = (lead * 24) + 12
        lead_hours_dpt = (lead * 24) - 9
        
        init_dt = target_dt - pd.Timedelta(days=lead)
        init_dt_str = pd.to_datetime(f"{init_dt.strftime('%Y-%m-%d')} {NBM_CYCLE}")
        
        # Max T
        try:
            ds_nbm_max, temp_grib_max = get_nbm_qmd(init_dt_str, lead_hours_max, 'max')
            nbm_tmax_f = (ds_nbm_max[list(ds_nbm_max.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
            daily_diffs[f'lead_{lead}_maxt'] = (['y', 'x'], (nbm_tmax_f - urma_tmax_f).astype(np.float32))
            ds_nbm_max.close()
            for f in glob.glob(f"{temp_grib_max}*"): os.remove(f)
        except Exception: daily_diffs[f'lead_{lead}_maxt'] = (['y', 'x'], np.full(urma_tmax_f.shape, np.nan, dtype=np.float32))

        # Min T
        try:
            ds_nbm_min, temp_grib_min = get_nbm_qmd(init_dt_str, lead_hours_min, 'min')
            nbm_tmin_f = (ds_nbm_min[list(ds_nbm_min.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
            daily_diffs[f'lead_{lead}_mint'] = (['y', 'x'], (nbm_tmin_f - urma_tmin_f).astype(np.float32))
            ds_nbm_min.close()
            for f in glob.glob(f"{temp_grib_min}*"): os.remove(f)
        except Exception: daily_diffs[f'lead_{lead}_mint'] = (['y', 'x'], np.full(urma_tmin_f.shape, np.nan, dtype=np.float32))

        # Dewpoint
        try:
            ds_nbm_dpt, temp_grib_dpt = get_nbm_qmd(init_dt_str, lead_hours_dpt, 'dpt')
            nbm_tdpt_f = (ds_nbm_dpt[list(ds_nbm_dpt.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
            daily_diffs[f'lead_{lead}_dpt'] = (['y', 'x'], (nbm_tdpt_f - urma_tdpt_f).astype(np.float32))
            ds_nbm_dpt.close()
            for f in glob.glob(f"{temp_grib_dpt}*"): os.remove(f)
        except Exception as e: 
            print(f"  [!] Day {lead} DPT missing: {e}")
            daily_diffs[f'lead_{lead}_dpt'] = (['y', 'x'], np.full(urma_tdpt_f.shape, np.nan, dtype=np.float32))

        print(f"  -> Day {lead} forecast (Max: +{lead_hours_max}h, Min: +{lead_hours_min}h, Dpt: +{lead_hours_dpt}h) processed.")

    # 4. Save to NetCDF
    ds_out = xr.Dataset(
        data_vars=daily_diffs,
        coords={
            'latitude': (['y', 'x'], lat_grid),
            'longitude': (['y', 'x'], lon_grid)
        }
    )
    
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_out.data_vars}
    
    ds_out.to_netcdf(nc_filename, encoding=encoding)
    ds_out.close()
    print(f"  ==> Saved {nc_filename}")

# ==============================================================================
# The Janitor
# ==============================================================================
print("\n--- Running Janitor ---")
cutoff_date = end_dt - pd.Timedelta(days=DAYS_TO_KEEP)

for nc_file in glob.glob("data_cache/diff_*.nc"):
    basename = os.path.basename(nc_file)
    file_date_str = basename.replace('diff_', '').replace('.nc', '')
    try:
        if pd.to_datetime(file_date_str) <= cutoff_date:
            os.remove(nc_file)
            print(f"Deleted old archive: {nc_file}")
    except Exception: pass
        
print("Daily Collector finished!")
