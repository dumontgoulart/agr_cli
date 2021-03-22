# -*- coding: utf-8 -*-
"""
Create and merge and adjust netcdf data
Created on Wed Jun 10 12:23:34 2020

@author: morenodu
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr 

ds_iizumi = xr.open_dataset('outfile_cat1.nc4')
ds_iizumi['time'] = pd.to_datetime(list(range(1981, 2017)), format='%Y').year
ds_iizumi['lon'] = (ds_iizumi.coords['lon'] + 180) % 360 - 180
ds_iizumi = ds_iizumi.sortby(ds_iizumi.lon)
ds_iizumi = ds_iizumi.rename({'lon': 'longitude','lat': 'latitude','var' : 'yield'})
ds_iizumi = ds_iizumi.sel(latitude=slice(None, None, -1)) 
ds_iizumi['yield'].attrs = {'units': 'ton/ha', 'long_name': 'Yield in tons per hectare'}
ds_iizumi.to_netcdf('soybean_iizumi_1981_2016.nc')
test = xr.open_dataset("soybean_iizumi_1981_2016.nc")


#trial to remove nan values along time
for i in range(len(da_iizumi['latitude'].values)):
    for j in range(len(da_iizumi['longitude'].values)):
        if np.isnan(da_iizumi[:,i,j]).all() == False and np.isnan(da_iizumi[:,i,j]).all() == True:
            da_iizumi[:,i,j]= np.nan