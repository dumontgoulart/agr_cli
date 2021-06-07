# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:15:16 2021

@author: morenodu
"""

# Mask US states
DS_y_hr = mask_shape_border(DS_y_base_us,soy_us_states) #US_shape
DS_y_hr = DS_y_hr.dropna(dim = 'lon', how='all')
DS_y_hr = DS_y_hr.dropna(dim = 'lat', how='all')
DS_y_hr=DS_y_hr.sel(time=slice(start_date, end_date))

# Mask to guarantee minimum 0.5 ton/ha yield
DS_y_hr = DS_y_hr.where(DS_y_hr['yield'].mean('time') > 0.5 )

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y_hr['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

# Mask for MIRCA 2000 each tile >0.9 rainfed
ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")

DS_y_hr = DS_y_hr.where(ds_mask['soybean_rainfed'] > 0.9 )
DS_y_hr = DS_y_hr.dropna(dim = 'lon', how='all')
DS_y_hr = DS_y_hr.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y_hr=DS_y_hr.drop('spatial_ref')
    
#---------------
# EGU
DS_y_2012 = DS_y.sel(time=2012)
DS_y_dif_2012 =   DS_y_2012 - DS_y['yield'].mean('time')
DS_y_dif_2012['yield'].attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}


plt.figure(figsize=(11,6), dpi=300) #plot clusters
ax=plt.axes(projection=ccrs.LambertConformal())
DS_y_dif_2012['yield'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r,levels=6)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_title('')
ax.set_extent([-115,-67,23,50], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.6)
plt.tight_layout()
plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
plt.show()

#---------------  

DS_tmx = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.tmx.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_t_mn=xr.open_dataset("cru_ts4.04.1901.2019.tmn.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_dtr=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.dtr.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pre.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec = DS_prec.rename_vars({'pre': 'precip'})

DS_cli = xr.merge([DS_tmx.tmx, DS_prec.precip, DS_dtr.dtr]) 
DS_cli_us = DS_cli.where(DS_y_hr['yield'].mean('time') > -0.1 )

if len(DS_cli_us.coords) >3 :
    DS_cli_us=DS_cli_us.drop('spatial_ref')


# First convert datasets to dataframes divided by month of season and show timeseries
df_clim_avg_features_us, df_epic_det_us = conversion_clim_yield(
    DS_y, DS_cli_us, months_to_be_used=[7,8], detrend = True)


# alternative formation to aggregate values and improve performance
# if len(df_clim_avg_features_us.columns) == 6:
df_clim_agg_features_us =  pd.concat([df_clim_avg_features_us.iloc[:,0:2].mean(axis=1),df_clim_avg_features_us.iloc[:,4:6].mean(axis=1),df_clim_avg_features_us.iloc[:,2:4].mean(axis=1) ], axis=1)
df_clim_agg_features_us.columns=['tmx_7_8','dtr_7_8', 'precip_7_8']


# Main algorithm for training the ML model
brf_model_us, fig_dep_us, rf_scores_us = failure_probability(df_clim_agg_features_us, df_epic_det_us, show_partial_plots= True, model_choice = 'conservative')

bar_perf_train_us = bar_perf_train(df_perf_train_us, rf_scores_us)
