# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:14:42 2020

@author: morenodu
"""


#%% yield model - WOFOST
start_date, end_date = '1982-12-31','2015-12-31'

DS_y2 = xr.open_dataset("soybean_iizumi_1981_2016.nc",decode_times=True)
DS_y2 = DS_y2.rename({'latitude': 'lat', 'longitude': 'lon'})
DS_y2 = mask_shape_border(DS_y2,soy_us_states) #US_shape
DS_y2 = DS_y2.dropna(dim = 'lon', how='all')
DS_y2 = DS_y2.dropna(dim = 'lat', how='all')
DS_y=DS_y2
DS_y=DS_y.sel(time=slice(start_date, end_date))

#second mask
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")
DS_y = DS_y.where(ds_mask['soybean_rainfed'] >= 0.9 )

DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')

DS_y.to_dataframe().groupby(['time']).mean().plot()

dataarray_iso = DS_y['yield'].where(DS_y['yield'] > 0, -5)
mean_cli = dataarray_iso.mean(axis=0)
dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=DS_y['yield'].attrs, name = DS_y['yield'].name ) + mean_cli
dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > 0, np.nan ) 
df_y_det = dataarray_iso_2.to_dataframe().dropna(how='all')
dataarray_iso_2.to_dataframe().groupby(['time']).mean().plot()

DS_anom = dataarray_iso_2.sel(time=2012) - dataarray_iso_2.loc[dataarray_iso_2['time'] != 2012 ].mean('time')
  
plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_anom.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

DS_anom.to_netcdf("yield_anom_us.nc")

#%% average values for time
df_epic=DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_epic.plot()
range_test = list(range(20, int(df_epic.index[-1]-df_epic.index[0]), 20))
# option 1
df_epic_det = pd.DataFrame( signal.detrend(df_epic['yield'], axis=0, bp=range_test), index=df_epic.index, columns = df_epic.columns, ) + df_epic.mean()
df_epic_det.plot()


####### boxplot for wp3
plt.figure(figsize = (6,6), dpi=144)
mean_graph = np.mean(df_epic_det.loc[df_epic_det.index != 2012 ])
deficit = (df_epic_det - mean_graph)
deficit.to_csv('yield_anomaly_us.csv')
plt.bar(x=deficit.index, height = deficit.values.ravel() )
plt.bar(x=df_epic_det.loc[df_epic_det.index == 2012 ].index, height = deficit.loc[df_epic_det.index == 2012 ].values.ravel(), color = 'salmon' )
plt.ylabel('Yield Anomaly (ton/ha)')
plt.title(f'Yield anomaly per year')   
plt.tight_layout()
plt.show()
######









#%% yield model - WOFOST
start_date, end_date = '1982-12-31','2015-12-31'

# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_br_names = ['Rio Grande do Sul','Paraná'] #'Rio Grande do Sul','Paraná' #'Mato Grosso','Goiás' 
soy_br_states = bra[bra['NAME_1'].isin(state_br_names)]

DS_y2 = xr.open_dataset("soybean_iizumi_1981_2016.nc",decode_times=True)
DS_y2 = DS_y2.rename({'latitude': 'lat', 'longitude': 'lon'})
DS_y2 = mask_shape_border(DS_y2,soy_br_states) #US_shape
DS_y2 = DS_y2.dropna(dim = 'lon', how='all')
DS_y2 = DS_y2.dropna(dim = 'lat', how='all')
DS_y=DS_y2
DS_y=DS_y.sel(time=slice(start_date, end_date))

#second mask
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")
DS_y = DS_y.where(ds_mask['soybean_rainfed'] >= 0.9 )

DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')

DS_y.to_dataframe().groupby(['time']).mean().plot()

dataarray_iso = DS_y['yield'].where(DS_y['yield'] > 0, -5)
mean_cli = dataarray_iso.mean(axis=0)
dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=DS_y['yield'].attrs, name = DS_y['yield'].name ) + mean_cli
dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > 0, np.nan ) 
df_y_det = dataarray_iso_2.to_dataframe().dropna(how='all')
dataarray_iso_2.to_dataframe().groupby(['time']).mean().plot()

DS_anom = dataarray_iso_2.sel(time=2012) - dataarray_iso_2.loc[dataarray_iso_2['time'] != 2012 ].mean('time')
  
plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_anom.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-70,-25,0,-35], ccrs.PlateCarree())
plt.show()

DS_anom.to_netcdf("yield_anom_br_s.nc")

#%% average values for time
df_epic=DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_epic.plot()
range_test = list(range(20, int(df_epic.index[-1]-df_epic.index[0]), 20))
# option 1
df_epic_det = pd.DataFrame( signal.detrend(df_epic['yield'], axis=0, bp=range_test), index=df_epic.index, columns = df_epic.columns, ) + df_epic.mean()
df_epic_det.plot()


####### boxplot for wp3
plt.figure(figsize = (6,6), dpi=144)
mean_graph = np.mean(df_epic_det.loc[df_epic_det.index != 2012 ])
deficit = (df_epic_det - mean_graph)
deficit.to_csv('yield_anomaly_br_s.csv')
plt.bar(x=deficit.index, height = deficit.values.ravel() )
plt.bar(x=df_epic_det.loc[df_epic_det.index == 2012 ].index, height = deficit.loc[df_epic_det.index == 2012 ].values.ravel(), color = 'salmon' )
plt.ylabel('Yield Anomaly (ton/ha)')
plt.title(f'Yield anomaly per year')   
plt.tight_layout()
plt.show()
######





#%% yield model - WOFOST
start_date, end_date = '1982-12-31','2015-12-31'

# Crop space to either ARG or soy states
arg = gpd.read_file('gadm36_ARG_1.shp', crs="epsg:4326") 
ar1_shapes = list(shpreader.Reader('gadm36_ARG_1.shp').geometries())
state_ar_names = ['Buenos Aires','Santa Fe', 'Córdoba'] #'Rio Grande do Sul','Paraná' #'Mato Grosso','Goiás' 
soy_ar_states = arg[arg['NAME_1'].isin(state_ar_names)]


DS_y2 = xr.open_dataset("soybean_iizumi_1981_2016.nc",decode_times=True)
DS_y2 = DS_y2.rename({'latitude': 'lat', 'longitude': 'lon'})
DS_y2 = mask_shape_border(DS_y2,soy_ar_states) #US_shape
DS_y2 = DS_y2.dropna(dim = 'lon', how='all')
DS_y2 = DS_y2.dropna(dim = 'lat', how='all')
DS_y=DS_y2
DS_y=DS_y.sel(time=slice(start_date, end_date))

#second mask
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")
DS_y = DS_y.where(ds_mask['soybean_rainfed'] >= 0.9 )

DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')

DS_y.to_dataframe().groupby(['time']).mean().plot()

dataarray_iso = DS_y['yield'].where(DS_y['yield'] > 0, -5)
mean_cli = dataarray_iso.mean(axis=0)
dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=DS_y['yield'].attrs, name = DS_y['yield'].name ) + mean_cli
dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > 0, np.nan ) 
df_y_det = dataarray_iso_2.to_dataframe().dropna(how='all')
dataarray_iso_2.to_dataframe().groupby(['time']).mean().plot()

DS_anom = dataarray_iso_2.sel(time=2012) - dataarray_iso_2.loc[dataarray_iso_2['time'] != 2012 ].mean('time')
  
plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_anom.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-70,-25,0,-35], ccrs.PlateCarree())
plt.show()

DS_anom.to_netcdf("yield_anom_ar.nc")

#%% average values for time
df_epic=DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_epic.plot()
range_test = list(range(20, int(df_epic.index[-1]-df_epic.index[0]), 20))
# option 1
df_epic_det = pd.DataFrame( signal.detrend(df_epic['yield'], axis=0, bp=range_test), index=df_epic.index, columns = df_epic.columns, ) + df_epic.mean()
df_epic_det.plot()


####### boxplot for wp3
plt.figure(figsize = (6,6), dpi=144)
mean_graph = np.mean(df_epic_det.loc[df_epic_det.index != 2012 ])
deficit = (df_epic_det - mean_graph)
deficit.to_csv('yield_anomaly_ar.csv')
plt.bar(x=deficit.index, height = deficit.values.ravel() )
plt.bar(x=df_epic_det.loc[df_epic_det.index == 2012 ].index, height = deficit.loc[df_epic_det.index == 2012 ].values.ravel(), color = 'salmon' )
plt.ylabel('Yield Anomaly (ton/ha)')
plt.title(f'Yield anomaly per year')   
plt.tight_layout()
plt.show()
######






#%% yield model - WOFOST
start_date, end_date = '1982-12-31','2015-12-31'

# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_br_names = ['Mato Grosso','Goiás'] #'Rio Grande do Sul','Paraná' #'Mato Grosso','Goiás' 
soy_br_states = bra[bra['NAME_1'].isin(state_br_names)]



DS_y2 = xr.open_dataset("soybean_iizumi_1981_2016.nc",decode_times=True)
DS_y2 = DS_y2.rename({'latitude': 'lat', 'longitude': 'lon'})
DS_y2 = mask_shape_border(DS_y2,soy_br_states) #US_shape
DS_y2 = DS_y2.dropna(dim = 'lon', how='all')
DS_y2 = DS_y2.dropna(dim = 'lat', how='all')
DS_y=DS_y2
DS_y=DS_y.sel(time=slice(start_date, end_date))

#second mask
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")
DS_y = DS_y.where(ds_mask['soybean_rainfed'] >= 0.9 )

DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')

DS_y.to_dataframe().groupby(['time']).mean().plot()

dataarray_iso = DS_y['yield'].where(DS_y['yield'] > 0, -5)
mean_cli = dataarray_iso.mean(axis=0)
dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=DS_y['yield'].attrs, name = DS_y['yield'].name ) + mean_cli
dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > 0, np.nan ) 
df_y_det = dataarray_iso_2.to_dataframe().dropna(how='all')
dataarray_iso_2.to_dataframe().groupby(['time']).mean().plot()

DS_anom = dataarray_iso_2.sel(time=2012) - dataarray_iso_2.loc[dataarray_iso_2['time'] != 2012 ].mean('time')
  
plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_anom.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-70,-25,0,-35], ccrs.PlateCarree())
plt.show()

DS_anom.to_netcdf("yield_anom_br_c.nc")

#%% average values for time
df_epic=DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_epic.plot()
range_test = list(range(20, int(df_epic.index[-1]-df_epic.index[0]), 20))
# option 1
df_epic_det = pd.DataFrame( signal.detrend(df_epic['yield'], axis=0, bp=range_test), index=df_epic.index, columns = df_epic.columns, ) + df_epic.mean()
df_epic_det.plot()


####### boxplot for wp3
plt.figure(figsize = (6,6), dpi=144)
mean_graph = np.mean(df_epic_det.loc[df_epic_det.index != 2012 ])
deficit = (df_epic_det - mean_graph)
deficit.to_csv('yield_anomaly_br_c.csv')
plt.bar(x=deficit.index, height = deficit.values.ravel() )
plt.bar(x=df_epic_det.loc[df_epic_det.index == 2012 ].index, height = deficit.loc[df_epic_det.index == 2012 ].values.ravel(), color = 'salmon' )
plt.ylabel('Yield Anomaly (ton/ha)')
plt.title(f'Yield anomaly per year')   
plt.tight_layout()
plt.show()
######