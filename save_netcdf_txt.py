import pandas as pd
import xarray as xr
# Load
data = pd.read_csv('ACY_gswp3-w5e5_obsclim_default_soy_noirr.txt', sep=";")
# Reduction to raster format & conversion to standard labels
data_red = data[["x","y","YR","YLDG"]]
data_red = data_red.rename(columns={'x':'lon', 'y':'lat', "YR":"time", "YLDG":"yield"})
# Check that some values are missing 169.25,168.75,167.75
data.loc[(data['x'] ==-167.75 ) ]
#Drop duplicates that in this case are not changing the yield values
data_red_drop = data_red.drop_duplicates(subset=['time','lat','lon'])
# Transform dataset into multiindex
df_multiindex = data_red_drop.set_index(["lon","lat","time"])
# Conversion to dataset / xarray, configure coordinates
DS_gswp3 = df_multiindex.to_xarray()
DS_gswp3 =DS_gswp3.transpose('time','lat','lon')
DS_gswp3["yield"].attrs = {'units': 't ha-1 yr-1', 'long_name': 'Yields', 'standard_name': 'yield'}
# Save file in nc format
DS_gswp3.to_netcdf('ACY_gswp3-w5e5_obsclim_default_soy_noirr.nc')

#%% Load ########################### MIRCA 2000 masks #####################################################
data_rain = pd.read_csv('annual_area_harvested_rfc_crop08_ha_30mn.asc', header=None, skiprows=6, delimiter=' ')# skipinitialspace=True, index_col=0,
data_rain=data_rain.iloc[:,0:-1]
data_irri = pd.read_csv('annual_area_harvested_irc_crop08_ha_30mn.asc', header=None, skiprows=6, delimiter=' ')# skipinitialspace=True, index_col=0,
data_irri=data_irri.iloc[:,0:-1]

#Total harvested region as sum of rainfed and irrigated hectares per pixel value
total_harvested_area = data_rain + data_irri

#Fraction of rainfed as rainfed area divided by the total
fraction_rf_to_ir = data_rain /( total_harvested_area)
fraction_rf_to_ir_90 = fraction_rf_to_ir.where(fraction_rf_to_ir > 0.9)
fraction_rf_to_ir_90 = fraction_rf_to_ir_90.iloc[::-1]
fraction_rf_to_ir_90.columns=DS_pet.lon.values
fraction_rf_to_ir_90.index = DS_pet.lat.values
data_test = fraction_rf_to_ir_90

# mask_harv = xr.Dataset.from_dataframe(data)
DS_base = xr.open_dataset("cru_ts4.04.1901.2019.tmx.dat.nc",decode_times=True)
DS_base_mean = DS_base.tmx.mean('time')
DS_base_mean.values = data_test
DS_base_mean.name = 'soybean_rainfed'
DS_base_mean.to_netcdf('mirca_2000_mask_soybean_rainfed.nc')

plt.figure(figsize=(14,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_base_mean.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

plt.figure(figsize=(14,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()