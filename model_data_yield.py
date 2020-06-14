import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from mask_shape_border import mask_shape_border

DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(lon=slice(-61,-44),lat=slice(-5,-33))
DS_y['time'] = pd.to_datetime(list(range(1979, 2013)), format='%Y').year
DS_y=DS_y.sel(time=slice('1980','2010'))
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
clipped= mask_shape_border(DS_y,'gadm36_BRA_0.shp' )
DS_y=clipped

# DS_y2 = xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(lon=slice(-100.25,-80.25),lat=slice(50.25,30.25), time=slice(1,31))
# DS_y2['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
# clipped= mask_shape_border(DS_y2,'gadm36_USA_0.shp' )
# DS_y2=clipped

# Iizumi data yield
ds_iizumi = xr.open_mfdataset('soybean/*.nc4', concat_dim="time", combine='nested')
ds_iizumi = ds_iizumi.assign_coords({"time" : ds_iizumi.time})
ds_iizumi['lon'] = (ds_iizumi.coords['lon'] + 180) % 360 - 180
ds_iizumi = ds_iizumi.sortby(ds_iizumi.lon)
ds_iizumi['time'] = pd.to_datetime(list(range(1981, 2017)), format='%Y').year
ds_iizumi = ds_iizumi.rename({'lon': 'longitude','lat': 'latitude','var' : 'yield'})
ds_iizumi['yield'].attrs = {'units': 'ton/ha', 'long_name': 'Yield in tons per hectare'}
# ds_iizumi.to_netcdf('soybean_iizumi.nc')
# test = xr.open_dataset("soybean_iizumi.nc")
da_iizumi = ds_iizumi['yield']
clipped= mask_shape_border(ds_iizumi,'gadm36_BRA_0.shp' )
ds_iizumi=clipped

#%% Average and standard deviation of soybean yield
yield_mean=DS_y['yield'].mean(dim = 'time', keep_attrs=True)
yield_std=DS_y['yield'].std(dim = 'time', keep_attrs=True)

plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
fig=yield_mean.plot(x='lon', y='lat', transform=ccrs.PlateCarree(),robust=True,cbar_kwargs={'label': yield_mean.attrs['units']}, cmap='Reds')
ax.set_title('Mean soy yield along time')
ax.set_xticks(ax.get_xticks()[::2]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-61,-44,-33,-8], ccrs.PlateCarree())  
plt.show()

plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
fig=yield_std.plot(x='lon', y='lat', transform=ccrs.PlateCarree(),robust=True,cbar_kwargs={'label': yield_mean.attrs['units']}, cmap='Reds')
ax.set_title('Standard deviation of soybean yield along time')
ax.set_xticks(ax.get_xticks()[::2]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-61,-44,-33,-8], ccrs.PlateCarree())    
plt.show()

#Establish temporal behaviour of the yield function (model based wofosst) wrt to the mean, +-SD.
df_t=ds_iizumi['yield'].to_series().groupby(['time']).mean()
df_t_mean=df_t.mean()
df_t_std=df_t.std()
df_t_low=df_t_mean - df_t_std
df_t_high=df_t_mean + df_t_std
plt.figure(figsize=(20,10))
plt.axhline(y=df_t_mean, color='r', linestyle='-')
plt.axhline(y=df_t_low, color='r', linestyle='--')
plt.axhline(y=df_t_high, color='r', linestyle='--')
plt.plot(df_t)
# plt.show()

# df_t2=DS_y2['yield'].to_series().groupby(['time']).mean()
# df_t2_mean=df_t2.mean()
# df_t2_std=df_t2.std()
# df_t2_low=df_t2_mean - df_t2_std
# df_t2_high=df_t2_mean + df_t2_std
# # plt.figure(figsize=(20,10))
# plt.axhline(y=df_t2_mean, color='r', linestyle='-')
# plt.axhline(y=df_t2_low, color='r', linestyle='--')
# plt.axhline(y=df_t2_high, color='r', linestyle='--')
# plt.plot(df_t2)
# plt.show()


#%%
from sklearn.linear_model import LinearRegression

series = pd.read_csv('fao_yield_soybean.csv', sep = ';', header=0, index_col=0)
series = series.loc[(series.index >= 1980) & (series.index <= 2010)]
# fit linear model
X = [i for i in range(0, len(series))]
X = np.reshape(X, (len(X), 1))
y = series.values
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
# plot trend
plt.plot(y)
plt.plot(trend)
plt.show()
# detrend
detrended = [trend.mean() + y[i]-trend[i] for i in range(0, len(series))]
df_detrended = pd.DataFrame(detrended, index=series.index, columns=['yield'] )  # 1st row as the column names
df_data_det = df_detrended.loc[(df_detrended.index >= 1981) & (df_detrended.index <= 2010)]
df_data_det.index = pd.to_datetime(list(range(1981, 2011)), format='%Y').year
plt.plot(df_data_det)
plt.show()

#detrend yield
series = df_t
# fit linear model
X = [i for i in range(0, len(series))]
X = np.reshape(X, (len(X), 1))
y = series.values
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
# plot trend
plt.plot(y)
plt.plot(trend)
plt.show()
# detrend
detrended = [trend.mean() + y[i]-trend[i] for i in range(0, len(series))]
df_detrended = pd.DataFrame(detrended, index=series.index, columns=['yield'] )  # 1st row as the column names
df_iizumi = df_detrended.loc[(df_detrended.index >= 1980) & (df_detrended.index <= 2010)]
df_iizumi.index = pd.to_datetime(list(range(1981, 2011)), format='%Y').year
plt.plot(df_iizumi)
plt.show()

# compare model and data yield
plt.figure(figsize=(10,6))
p1=plt.plot(df_iizumi, label='Model yield')
p2=plt.axhline(y=df_t_low, color='r', linestyle='--', label='Upper SD model yield')
p3=plt.axhline(y=df_t_mean, color='r', linestyle='-', label='Mean model yield')
p4=plt.axhline(y=df_t_high, color='r', linestyle='--', label='Lower SD model yield')
p5=plt.plot(df_data_det, label='Data yield')
plt.title('Comparison model and data yield',fontsize='x-large')
plt.ylabel('Yield')
plt.xlabel('Year')
plt.legend(loc='best', fontsize='x-large')
plt.show()

# calculate Pearson's correlation
from scipy.stats import pearsonr
val1=[float(i) for i in df_data_det.values]
val2=[float(i) for i in df_iizumi.values]
corr, _ = pearsonr(val2, val1)
print('Pearsons correlation: %.3f' % corr)


