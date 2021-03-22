# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:07:26 2020

@author: morenodu
"""
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
from  scipy import signal 
from mask_shape_border import mask_shape_border
from lag_linregress_3D import lag_linregress_3D

ds_tagp = xr.open_dataset("productivity/soy_yield_TAGP.nc")
ds_twso= xr.open_dataset("productivity/soy_yield_TWSO.nc")

#try for a mask on the US soy states

ds_tagp_us = mask_shape_border(ds_tagp,soy_us_states ) #clipping for us

ds_tagp_us.TAGP.groupby('time').mean(...).plot()

plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
ds_twso.TWSO.sel(time='2012').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='RdBu',levels=10)
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.stock_img()
ax.set_extent([-110,-70,25, 49], ccrs.PlateCarree())
plt.show()


plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
ds_twso.TWSO.sel(time='2018').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='RdBu',levels=10)
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.stock_img()
ax.set_extent([-70,-35,-39, -2], ccrs.PlateCarree())
plt.show()


for i in range(len(soy_us_states)):
    soy_state = soy_us_states.iloc[i]
    da_test=DS_cli
    da_test=da_test.rename({'lon': 'longitude','lat': 'latitude'})
    da_cli_state_det_tmax2 = mask_shape_border(da_test,soy_us_states ) #clipping for us
    da_cli_state_det_tmax_mean = da_cli_state_det_tmax2.groupby('time').mean(...)
    df_tmax=da_cli_state_det_tmax_mean.to_series()
    
    da_cli_state_det_prec2 = mask_shape_border(da_cli_us_det_prec2,soy_state )
    da_cli_us_det_prec_mean = da_cli_state_det_prec2.groupby('time').mean(...)
    df_prec=da_cli_state_det_prec_mean.to_series()

    da_cli_state_det_wet2 = mask_shape_border(da_cli_us_det_wet2,soy_state )
    da_cli_state_det_wet_mean = da_cli_state_det_wet2.groupby('time').mean(...)
    df_wet=da_cli_state_det_wet_mean.to_series()