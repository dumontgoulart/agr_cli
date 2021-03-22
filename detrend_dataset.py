import xarray as xr 
import numpy as np 
from  scipy import signal 

def detrend_dataset(da, months_to_select =[6, 7, 8,9,10]):
    
    mean_cli = da.mean(axis=0)
    da_cli_us_det_temp1 =  xr.DataArray(signal.detrend(da, axis=0), dims=da.dims, coords=da.coords, attrs=da.attrs) + mean_cli
    da_cli_us_det_temp2 = da_cli_us_det_temp1.where(da_cli_us_det_temp1 > -300, np.nan ).sel(time = da.indexes['time'].month.isin(months_to_select)) 
    da_cli_us_det_temp_mean = da_cli_us_det_temp2.groupby('time').mean(...)
    df_temp=da_cli_us_det_temp_mean.to_series()
    return df_temp
    
