import xarray as xr
import pandas as pd
import numpy as np
import requests
import os
import glob
from herbie import Herbie
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Shared HTTP Session for connection pooling
http_session = requests.Session()

def get_nbm_qmd(init_dt, lead_hrs, var_type, session):
    date_str = init_dt.strftime('%Y%m%d')
    cycle_str = f"{init_dt.hour:02d}"
    
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{cycle_str}/qmd"
    grib_name = f"blend.t{cycle_str}z.qmd.f{lead_hrs:03d}.co.grib2"
    
    r_idx = session.get(f"{base_url}/{grib_name}.idx", timeout=15)
    r_idx.raise_for_status()
        
    start_byte, end_byte = None, None
    lines = r_idx.text.splitlines()
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "50% level" in line_lower and "2 m above ground" in line_lower:
            if (var_type == 'max' and "max" in line_lower) or \
               (var_type == 'min' and "min" in line_lower) or \
               (var_type == 'dpt' and ":dpt:" in line_lower):
                start_byte = int(line.split(':')[1])
                
                # FIX: Scan forward for the next line with a strictly greater byte offset
                for next_line in lines[i + 1:]:
                    next_byte = int(next_line.split(':')[1])
                    if next_byte > start_byte:
                        end_byte = next_byte - 1
                        break
                break
            
    if start_byte is None:
        raise Exception(f"{var_type.upper()} (50%) not found in GRIB index.")
        
    headers = {"Range": f"bytes={start_byte}-{end_byte}" if end_byte else f"bytes={start_byte}-"}
    r_grib = session.get(f"{base_url}/{grib_name}", headers=headers, timeout=30)
    r_grib.raise_for_status()
    
    temp_file = f"nbm_temp_{var_type}_{lead_hrs}h_{init_dt.strftime('%Y%m%d')}.grib2"
    with open(temp_file, "wb") as f:
        f.write(r_grib.content)
        
    return temp_file

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
        # FIX: Using remove_grib=True and closing xarray datasets to prevent resource leaks
        H_urma_max = Herbie(urma_maxt_dt, model="urma", product="anl")
        with H_urma_max.xarray(":TMAX:2 m above ground:", remove_grib=True) as ds_urma_max:
            urma_tmax_f = (ds_urma_max[list(ds_urma_max.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
            lat_grid = ds_urma_max.latitude.squeeze().values[::2, ::2].astype(np.float32)
            lon_grid = ds_urma_max.longitude.squeeze().values[::2, ::2].astype(np.float32)
        
        H_urma_min = Herbie(urma_mint_dt, model="urma", product="anl")
        with H_urma_min.xarray(":TMIN:2 m above ground:", remove_grib=True) as ds_urma_min:
            urma_tmin_f = (ds_urma_min[list(ds_urma_min.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32

        H_urma_dpt = Herbie(urma_dpt_dt, model="urma", product="anl")
        with H_urma_dpt.xarray(":DPT:2 m above ground:", remove_grib=True) as ds_urma_dpt:
            urma_tdpt_f = (ds_urma_dpt[list(ds_urma_dpt.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
            
    except Exception as e:
        print(f"  [!] URMA failed for {date_str}: {e}. Skipping day.")
        continue
        
    daily_diffs = {}
    
    # FIX: Parallelized Downloads 
    def process_lead(lead):
        lead_hours_max = lead * 24
        lead_hours_min = (lead * 24) + 12
        lead_hours_dpt = (lead * 24) - 9
        
        init_dt = target_dt - pd.Timedelta(days=lead)
        init_dt_str = pd.to_datetime(f"{init_dt.strftime('%Y-%m-%d')} {NBM_CYCLE}")
        
        results = {}
        
        # Max T
        try:
            temp_grib_max = get_nbm_qmd(init_dt_str, lead_hours_max, 'max', http_session)
            with xr.open_dataset(temp_grib_max, engine='cfgrib') as ds_nbm_max:
                nbm_tmax_f = (ds_nbm_max[list(ds_nbm_max.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
                assert nbm_tmax_f.shape == urma_tmax_f.shape, "NBM and URMA grid shape mismatch"
                results[f'lead_{lead}_maxt'] = (['y', 'x'], (nbm_tmax_f - urma_tmax_f).astype(np.float32))
            for f in glob.glob(f"{temp_grib_max}*"): os.remove(f)
        except Exception as e: 
            print(f"    [!] Day {lead} MaxT failed: {e}")
            results[f'lead_{lead}_maxt'] = (['y', 'x'], np.full(urma_tmax_f.shape, np.nan, dtype=np.float32))

        # Min T
        try:
            temp_grib_min = get_nbm_qmd(init_dt_str, lead_hours_min, 'min', http_session)
            with xr.open_dataset(temp_grib_min, engine='cfgrib') as ds_nbm_min:
                nbm_tmin_f = (ds_nbm_min[list(ds_nbm_min.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
                assert nbm_tmin_f.shape == urma_tmin_f.shape, "NBM and URMA grid shape mismatch"
                results[f'lead_{lead}_mint'] = (['y', 'x'], (nbm_tmin_f - urma_tmin_f).astype(np.float32))
            for f in glob.glob(f"{temp_grib_min}*"): os.remove(f)
        except Exception as e: 
            print(f"    [!] Day {lead} MinT failed: {e}")
            results[f'lead_{lead}_mint'] = (['y', 'x'], np.full(urma_tmin_f.shape, np.nan, dtype=np.float32))

        # Dewpoint
        try:
            temp_grib_dpt = get_nbm_qmd(init_dt_str, lead_hours_dpt, 'dpt', http_session)
            with xr.open_dataset(temp_grib_dpt, engine='cfgrib') as ds_nbm_dpt:
                nbm_tdpt_f = (ds_nbm_dpt[list(ds_nbm_dpt.data_vars)[0]].squeeze().values[::2, ::2] - 273.15) * 9/5 + 32
                assert nbm_tdpt_f.shape == urma_tdpt_f.shape, "NBM and URMA grid shape mismatch"
                results[f'lead_{lead}_dpt'] = (['y', 'x'], (nbm_tdpt_f - urma_tdpt_f).astype(np.float32))
            for f in glob.glob(f"{temp_grib_dpt}*"): os.remove(f)
        except Exception as e: 
            print(f"    [!] Day {lead} DPT failed: {e}")
            results[f'lead_{lead}_dpt'] = (['y', 'x'], np.full(urma_tdpt_f.shape, np.nan, dtype=np.float32))

        print(f"  -> Day {lead} forecast processed.")
        return results

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_lead, l) for l in range(1, MAX_LEAD_DAYS + 1)]
        for future in as_completed(futures):
            daily_diffs.update(future.result())

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
