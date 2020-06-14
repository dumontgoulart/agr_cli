# WP3 GA presentation - el nino sst correlation 
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as plticker
from  scipy import stats, signal #Required for detrending data and computing regression
from lag_linregress_3D import lag_linregress_3D
import dask
from mask_shape_border import mask_shape_border

adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())

#%% data import 

#SST data global + SST&temp
DS_sgl = xr.open_dataset("sst_global1.nc")
da_sgl = DS_sgl.sst
da_sgl['time'] = da_sgl.indexes['time'].normalize()
da_sgl_enso = da_sgl.sel(time=slice('1981','2017'),longitude=slice(-170,-120), latitude = slice(5,-5) ) #enso size

DS_tempg = xr.open_dataset("sst_global.nc")
da_tempg = DS_tempg.temperature_anomaly

# enso data
DS_sst = xr.open_dataset("sst.nc").sel(time=slice('1982','2016'),longitude=slice(-170,-120), latitude = slice(5,-5) )
DS_sst['growing_month'] = DS_sst["time.month"]
DS_sst['growing_year'] = DS_sst["time.year"]
for i in range(len(DS_sst["time"].values)):
    if DS_sst["time.month"].values[i] < 5:
        DS_sst['growing_year'][i] = DS_sst["time.year"][i]
    else:
        DS_sst["growing_year"][i] = DS_sst["time.year"][i]+1    
DS_sst['growing_month'].values = np.array([9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]* int((len(DS_sst['time.year'])/12)))
DS_sst['time'] = DS_sst['growing_year']
da_sst = DS_sst.sst

#model yield Br

DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(lon=slice(-63.25,-43.25),lat=slice(-5.25,-35.75), time=slice(1,31))
DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y = DS_y.rename({'lon': 'longitude','lat': 'latitude'})
da_y= DS_y['yield']

#model yield USA

DS_y2 = xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(lon=slice(-100.25,-80.25),lat=slice(50.25,30.25), time=slice(1,31))
DS_y2['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y2= DS_y2.rename({'lon': 'longitude','lat': 'latitude'})
da_y2 = DS_y2['yield']

#import iizumi dataset

ds_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc").sel(time = slice(1982,2016))
clippedus= mask_shape_border(ds_iizumi,'gadm36_USA_0.shp' ) #clipping for us
clippedbr= mask_shape_border(ds_iizumi,'gadm36_BRA_0.shp' ) #clipping for br
ds_iizumi_us=clippedus
ds_iizumi_br=clippedbr

da_iizumi_us = ds_iizumi_us['yield']
da_iizumi_br = ds_iizumi_br['yield']

# ds_iizumi_us = da_iizumi.where(da_iizumi.isel(time = -1) > 0 ) #trying to remove nan
#%% #plot of iizumi data for soybean yield data-based

da_sst_mean = da_sst.to_dataframe().groupby(['time']).mean()
plt.plot(da_sst_mean) ###plot timeseries

df_sgl_enso_mean = da_sgl_enso.to_dataframe().groupby(['time']).mean()
da_sgl_enso_mean = df_sgl_enso_mean.to_xarray()

la_nina = da_sgl_enso_mean.where(da_sgl_enso_mean < -0.5,drop=True)

df_iizumi_us_mean = da_iizumi_us.to_dataframe().groupby(['time']).mean()
df_iizumi_br_mean = da_iizumi_br.to_dataframe().groupby(['time']).mean()
# plt.plot(df_iizumi_mean) ### plot time series

# ax = plt.axes(projection=ccrs.PlateCarree()) #####plot projection at time T
# da_iizumi.isel(time = 1).plot(levels=20,cmap='YlGn', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree());
# ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-63,-42,-35,-8], ccrs.PlateCarree())
#%% comparing linear data

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

#standardise it
# df_sst_scaled = pd.DataFrame(scaler.fit_transform(df_sst), columns = df_sst.columns, index=df_sst.index)

df_y_mean= da_y.to_dataframe().groupby(['time']).mean()
df_y_mean_sc = pd.DataFrame(scaler.fit_transform(df_y_mean), columns = df_y_mean.columns, index=df_y_mean.index)

#detrend yield
test=[]
for j in [df_iizumi_us_mean, df_iizumi_br_mean]:   
    series = j
    # fit linear model
    X = [i for i in range(0, len(series))]
    X = np.reshape(X, (len(X), 1))
    y = series.values
    model = LinearRegression()
    model.fit(X, y)
    # calculate trend
    trend = model.predict(X)
    # plot trend
    # plt.plot(y)
    # plt.plot(trend)
    # plt.show()
    # detrend
    detrended = [trend.mean() + y[i]-trend[i] for i in range(0, len(series))]
    df_detrended = pd.DataFrame(detrended, index=series.index, columns=['yield'] )  # 1st row as the column names
    j = df_detrended.loc[(df_detrended.index >= 1981) & (df_detrended.index <= 2016)]
    l = pd.DataFrame(scaler.fit_transform(j), columns = j.columns, index=j.index)
    plt.plot(l)
    plt.show()
    test.append(l)

df_iizumi_us_mean_scaled = test[0]
df_iizumi_br_mean_scaled = test[1]


plt.figure(figsize=(10,6))
p1=plt.plot(df_iizumi_scaled, label='Model yield')
p2=plt.plot(da_sst_mean, color='r', label='ENSO (SST)',linestyle='--')
p3=plt.plot( df_y_mean_sc, color='black', label='ENSO (SST)')


# # correlation
# mean_sst = da_sst_mean.mean(axis=0)
# da_sst_time = xr.DataArray(signal.detrend(da_sst_time, axis=0), dims=da_sst_time.dims, coords=da_sst_time.coords, attrs = da_sst_time.attrs) + mean_sst

mean_ii = np.nanmean(da_iizumi, axis=0)
da_iizumi_d = xr.DataArray(signal.detrend(da_iizumi, axis=0), dims=da_iizumi.dims, coords=da_iizumi.coords, attrs = da_iizumi.attrs) + mean_ii
df_iizumi_d_mean = da_iizumi_d.to_series().groupby(['time']).mean()
plt.plot(df_iizumi_d_mean)


#%% Correlation ENSO and yields

cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=da_sgl_enso,y=da_y)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(),robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu',levels=10)
ax.set_title('Comparison ENSO signal and soybean yield',fontsize='x-large')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-63,-43,-35,-8], ccrs.PlateCarree())
plt.show()


cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=da_sgl_enso,y=da_y2)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.LambertConformal(central_longitude=-90.0))
adm1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(),robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu',levels=10)
ax.set_title('Comparison ENSO signal and soybean yield',fontsize='x-large')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-100,-80,28, 49], ccrs.PlateCarree())
plt.show()


#%% sst plots globally - multiple plots
import seaborn as sns

df_iizumi_us_mean_scaled.index = pd.to_datetime(list(range(1982, 2017)), format='%Y')
df_iizumi_us_mean_scaled.index =df_iizumi_us_mean_scaled.index + pd.offsets.MonthEnd(11) 
df_iizumi_br_mean_scaled.index = pd.to_datetime(list(range(1982, 2017)), format='%Y')
df_iizumi_br_mean_scaled.index =df_iizumi_br_mean_scaled.index + pd.offsets.MonthEnd(5) 

# Line plot comparing enso with yields
sns.set_style("ticks")
sns.set_context("talk")
plt.figure(figsize=(15,10)) 
plt.title("ENSO and yield anomalies with respect to the 1960-1990 period")
plt.ylabel("Temperature and yield Anomalies")
plt.xlabel("")
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=-0.5, color='r', linestyle='--')
plt.axhline(y=-1, color='r', linestyle='--')
sns.lineplot(data=da_sgl_enso_mean.to_dataframe(), palette = "PuBuGn_d", linewidth=2.5)
sns.lineplot(data=df_iizumi_us_mean_scaled, palette="husl", linewidth=2.5)
sns.lineplot(data=df_iizumi_br_mean_scaled, palette=['orange'],linewidth=2.5)

# sst around the world
plt.figure(figsize=(20,10))
ax = plt.axes(projection=ccrs.PlateCarree())
da_sgl.sel(time='2011-12-16').plot(levels=10,cmap='RdBu_r', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree(),cbar_kwargs={'label': 'Temperature anomaly in Celsius degree'})
ax.set_title("SST anomaliy in December 2011 with respect to the 1960-1990 period")
ax.set_global(); ax.coastlines();
#temperature in the US
plt.figure(figsize=(20,10)) 
ax = plt.axes(projection=ccrs.PlateCarree())
da_tempg.sel(time='2012-03-16').plot(levels=10,cmap='RdBu_r', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree(),cbar_kwargs={'label': 'Temperature anomaly in Celsius degree'})
ax.set_title("Temperature anomaliy in March 2012 with respect to the 1960-1990 period")
ax.set_global(); ax.coastlines();
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.set_extent([-150,-50,0,75], ccrs.PlateCarree())
# temperature in south america
plt.figure(figsize=(20,10)) 
ax = plt.axes(projection=ccrs.PlateCarree())
da_tempg.sel(time='2011-12-16').sel(latitude=slice(-55,15),longitude=slice(-130,-23)).plot(levels=10,cmap='RdBu_r', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree(),cbar_kwargs={'label': 'Temperature anomaly in Celsius degree'})
ax.set_title("Temperature anomaliy in December 2011 with respect to the 1960-1990 period")
ax.coastlines();
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.set_extent([-110,-23,-55,15], ccrs.PlateCarree())


