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
# Custom Downloader for NBM QMD (Max & Min)
# ==============================================================================
def get_nbm_qmd(init_dt, lead_hrs, var_type='max'):
    date_str = init_dt.strftime('%Y%m%d')
    cycle_str = f"{init_dt.hour:02d}"
    
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{cycle_str}/qmd"
    grib_name = f"blend.t{cycle_str}z.qmd.f{lead_hrs:03d}.co.grib2"
    
    r_idx = requests.get(f"{base_url}/{grib_name}.idx")
    if r_idx.status_code != 200:
        raise Exception("Index file not found")
        
    start_byte, end_byte = None, None
    lines = r_idx.text.splitlines()
    
    # Toggle search string based on Max/Min
    fcst_str = "max fcst" if var_type == 'max' else "min fcst"
    
    for i, line in enumerate(lines):
        if f":TMP:2 m above ground:" in line and fcst_str in line and "50% level" in line:
            start_byte = int(line.split(':')[1])
            if i + 1 < len(lines):
                end_byte = int(lines[i+1].split(':')[1]) - 1
            break
            
    if start_byte is None:
        raise Exception(f"TMP {fcst_str} (50%) not found.")
        
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
    nc_filename = f"data_cache/diff_{date_str}.nc"
    
    if os.path.exists(nc_filename):
        print(f"[{date_str}] -> Already exists. Skipping.")
        continue
        
    print(f"\n[{date_str}] -> Missing. Starting processing...")
    
    # 2. Download URMA Max T (08z) and Min T (20z)
    urma_maxt_dt = f"{date_str} 08:00"
    urma_mint_dt = f"{date_str} 20:00"
    
    try:
        # Max T
        H_urma_max = Herbie(urma_maxt_dt, model="urma", product="anl")
        ds_urma_max = H_urma_max.xarray(":TMAX:2 m above ground:", remove_grib=False)
        urma_max_var = list(ds_urma_max.data_vars)[0]
        urma_tmax_f = (ds_urma_max[urma_max_var] - 273.15) * 9/5 + 32
        
        # Min T
        H_urma_min = Herbie(urma_mint_dt, model="urma", product="anl")
        ds_urma_min = H_urma_min.xarray(":TMIN:2 m above ground:", remove_grib=False)
        urma_min_var = list(ds_urma_min.data_vars)[0]
        urma_tmin_f = (ds_urma_min[urma_min_var] - 273.15) * 9/5 + 32
        
    except Exception as e:
        print(f"  [!] URMA failed for {date_str}: {e}. Skipping day.")
        continue
        
    daily_diffs = {}
    
    # 3. Download NBM Leads 1-8
    for lead in range(1, MAX_LEAD_DAYS + 1):
        lead_hours = lead * 24
        init_dt = target_dt - pd.Timedelta(days=lead)
        init_dt_str = pd.to_datetime(f"{init_dt.strftime('%Y-%m-%d')} {NBM_CYCLE}")
        
        # Process Max T
        try:
            ds_nbm_max, temp_grib_max = get_nbm_qmd(init_dt_str, lead_hours, 'max')
            nbm_var_max = list(ds_nbm_max.data_vars)[0]
            nbm_tmax_f = (ds_nbm_max[nbm_var_max] - 273.15) * 9/5 + 32
            
            diff_max = (nbm_tmax_f.values - urma_tmax_f.values).astype(np.float32)
            daily_diffs[f'lead_{lead}_maxt'] = (['y', 'x'], diff_max)
            
            ds_nbm_max.close()
            os.remove(temp_grib_max)
            for idx_file in glob.glob(f"{temp_grib_max}*.idx"): os.remove(idx_file)
        except Exception as e:
            print(f"  [!] Day {lead} Max T missing: {e}")
            daily_diffs[f'lead_{lead}_maxt'] = (['y', 'x'], np.full(urma_tmax_f.shape, np.nan, dtype=np.float32))

        # Process Min T
        try:
            ds_nbm_min, temp_grib_min = get_nbm_qmd(init_dt_str, lead_hours, 'min')
            nbm_var_min = list(ds_nbm_min.data_vars)[0]
            nbm_tmin_f = (ds_nbm_min[nbm_var_min] - 273.15) * 9/5 + 32
            
            diff_min = (nbm_tmin_f.values - urma_tmin_f.values).astype(np.float32)
            daily_diffs[f'lead_{lead}_mint'] = (['y', 'x'], diff_min)
            
            ds_nbm_min.close()
            os.remove(temp_grib_min)
            for idx_file in glob.glob(f"{temp_grib_min}*.idx"): os.remove(idx_file)
        except Exception as e:
            print(f"  [!] Day {lead} Min T missing: {e}")
            daily_diffs[f'lead_{lead}_mint'] = (['y', 'x'], np.full(urma_tmin_f.shape, np.nan, dtype=np.float32))

        print(f"  -> Day {lead} forecast (+{lead_hours}h) Max/Min processed.")

    # 4. Save to NetCDF
    ds_out = xr.Dataset(
        data_vars=daily_diffs,
        coords={
            'latitude': (['y', 'x'], ds_urma_max.latitude.values.astype(np.float32)),
            'longitude': (['y', 'x'], ds_urma_max.longitude.values.astype(np.float32))
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
        file_date = pd.to_datetime(file_date_str)
        if file_date <= cutoff_date:
            os.remove(nc_file)
            print(f"Deleted old archive: {nc_file}")
    except Exception as e:
        pass
        
print("Daily Collector finished!")
