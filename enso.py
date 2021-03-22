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

# Shapes for cropping 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
us1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
#%% data import 

#SST data global 
DS_sgl = xr.open_dataset("sst_global1.nc")
da_sgl = DS_sgl.sst
da_sgl['time'] = da_sgl.indexes['time'].normalize()

#ENSO
da_sgl_enso = da_sgl.sel(time=slice('1981','2017'),longitude=slice(-170,-120), latitude = slice(5,-5) ) #enso size

#Indian ocean dipole months since 1958-01-15
dmi =  xr.open_dataset("dmi.nc").DMI.sel(WEDCEN2=slice('1990','2015'))
dmi_3m = dmi.rolling(WEDCEN2=16, keep_attrs = True).mean()

#NAO index
da_nao = xr.open_dataset("nao_1990_2015.nc",decode_times=False ).index
da_nao['time'] = da_sgl['time'].sel(time=slice('1990','2015'))
da_nao_3 = da_nao.rolling(time=3, keep_attrs = True).mean()

# global temp
DS_tempg = xr.open_dataset("sst_global.nc")
da_tempg = DS_tempg.temperature_anomaly

#local climate
DS_cli=xr.open_dataset("era5_sst_global_2.nc")
DS_cli = DS_cli.sel(time=slice('1981-01-01','2012-12-31'),longitude=slice(-150,0), latitude = slice(60,-60))
#US-shape
da_cli_us = mask_shape_border(DS_cli,'gadm36_USA_0.shp' ).t2m
da_cli_us_mean = (da_cli_us.to_dataframe().groupby(['time']).mean()).to_xarray().t2m
da_cli_us_mean = da_cli_us_mean  -273.15
#detrend
mean_temp = da_cli_us_mean.mean()
da_cli_us_det = xr.DataArray(signal.detrend(da_cli_us_mean, axis=0), dims=da_cli_us_mean.dims, coords=da_cli_us_mean.coords, attrs = da_cli_us_mean.attrs) + mean_temp
# Climate for each month 
monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
for i in [7]:
    da_cli_us_month = da_cli_us_det.sel(time=DS_cli['time.month'] == i )
    da_cli_us_month['time'] = da_cli_us_month.indexes['time'].year
    
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

da_iizumi_us = ds_iizumi_us['yield'].sel(longitude=slice(-100.25,-80.25),latitude=slice(50.25,30.25))
da_iizumi_br = ds_iizumi_br['yield'].sel(longitude=slice(-63.25,-40.25),latitude=slice(-5.25,-35.25))

# detrend yield
da_iizumi_us_det = da_iizumi_us.where(da_iizumi_us > 0, 0 )
mean3 = da_iizumi_us_det.mean(axis=0)
da_iizumi_us_det_1 = xr.DataArray(signal.detrend(da_iizumi_us_det, axis=0), dims=da_iizumi_us_det.dims, coords=da_iizumi_us_det.coords, attrs=da_iizumi_us_det.attrs) + mean3
da_iizumi_us_det_2 = da_iizumi_us_det_1.where(da_iizumi_us > 0, np.nan )

da_iizumi_br_det = da_iizumi_br.where(da_iizumi_br > 0, 0 )
mean_br = da_iizumi_br_det.mean(axis=0)
da_iizumi_br_det_1 = xr.DataArray(signal.detrend(da_iizumi_br_det, axis=0), dims=da_iizumi_br_det.dims, coords=da_iizumi_br_det.coords, attrs=da_iizumi_br_det.attrs) + mean_br
da_iizumi_br_det_2 = da_iizumi_br_det_1.where(da_iizumi_br > 0, np.nan )

ax = plt.axes(projection=ccrs.PlateCarree()) #####plot projection at time T
da_iizumi_us_det_2.isel(time = 1).plot(levels=20,cmap='YlGn', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree());
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-100,-80,28, 49], ccrs.PlateCarree())
#%% #plot of iizumi data for soybean yield data-based

df_sgl_enso_mean = da_sgl_enso.to_dataframe().groupby(['time']).mean()
da_sgl_enso_mean = df_sgl_enso_mean.to_xarray()

la_nina = da_sgl_enso_mean.where(da_sgl_enso_mean < -1.0, drop=True).sst
la_nina_years = la_nina.groupby('time.year').mean('time').year.values

df_iizumi_us_mean = da_iizumi_us.to_dataframe().groupby(['time']).mean()
df_iizumi_br_mean = da_iizumi_br.to_dataframe().groupby(['time']).mean()
# plt.plot(df_iizumi_mean) ### plot time series

# ax = plt.axes(projection=ccrs.PlateCarree()) #####plot projection at time T
# da_iizumi.isel(time = 1).plot(levels=20,cmap='YlGn', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree());
# ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-63,-42,-35,-8], ccrs.PlateCarree())
#%% comparing linear data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

scaler=StandardScaler()

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
p1=plt.plot(df_iizumi_br_mean_scaled, label='Model yield')
p2=plt.plot(da_sgl_enso_mean, color='r', label='ENSO (SST)',linestyle='--')
p3=plt.plot( df_y_mean_sc, color='black', label='ENSO (SST)')


#%% Correlation ENSO and yields - Needs to be fixed!!
#brasil
cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=da_cli_us_month,y=da_iizumi_br_det_2)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(),robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu',levels=10)
ax.set_title('Comparison ENSO signal and soybean yield',fontsize='x-large')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-63,-43,-35,-8], ccrs.PlateCarree())
plt.show()
#usa
cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=da_cli_us_month,y=da_iizumi_us_det_2)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(),robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu',levels=10)
ax.set_title('Comparison ENSO signal and soybean yield',fontsize='x-large')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-100,-80,28, 49], ccrs.PlateCarree())
plt.show()


#%% sst plots globally - multiple plots
import seaborn as sns

df_iizumi_us_mean_scaled.index = pd.to_datetime(list(range(1982, 2017)), format='%Y')
df_iizumi_us_mean_scaled.index =df_iizumi_us_mean_scaled.index + pd.offsets.MonthEnd(11) 
df_iizumi_br_mean_scaled.index = pd.to_datetime(list(range(1982, 2017)), format='%Y')
df_iizumi_br_mean_scaled.index =df_iizumi_br_mean_scaled.index + pd.offsets.MonthEnd(5)
df_iizumi_us_mean_scaled_scatter = df_iizumi_us_mean_scaled[df_iizumi_us_mean_scaled.index < '2013']
df_iizumi_us_mean_scaled_scatter.index = df_iizumi_us_mean_scaled_scatter.index.year 
df_iizumi_br_mean_scaled_scatter = df_iizumi_br_mean_scaled[df_iizumi_br_mean_scaled.index < '2013']
df_iizumi_br_mean_scaled_scatter.index = df_iizumi_br_mean_scaled_scatter.index.year

col = np.where(df_iizumi_us_mean_scaled_scatter.index.isin(['1988','1998','1999','2007','2008','2011','2012']) ,'r','k')
plt.scatter(da_cli_us_month, df_iizumi_us_mean_scaled_scatter, c=col)

# Line plot comparing enso with yields
sns.set_style("ticks")
sns.set_context("talk")
sns.despine()
plt.figure(figsize=(15,10)) 
plt.title("ENSO index")
plt.ylabel("SST Anomaly in Celsius degree")
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=-0.5, color='r', linestyle='--')
plt.axhline(y=0.5, color='r', linestyle='--')
sns.lineplot(data=da_sgl_enso_mean.to_dataframe(), palette = "PuBuGn_d", linewidth=2.5)
sns.despine();

# IOD plot comparing enso with yields
plt.figure(figsize=(15,10)) 
plt.title("Indian Ocean Dipole ")
plt.ylabel("SST anomaly difference in Celsius degree")
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=-0.5, color='r', linestyle='--')
plt.axhline(y=0.5, color='g', linestyle='--')
sns.lineplot(data=dmi_3m.to_series(), palette = "PuBuGn_d", linewidth=2.5)
sns.despine();

# NAO plot comparing enso with yields
plt.figure(figsize=(15,10)) 
plt.title("North Atlantic Oscillation")
plt.ylabel("Difference of normalized sea level pressure (SLP)")
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=-0.5, color='r', linestyle='--')
plt.axhline(y=0.5, color='g', linestyle='--')
sns.lineplot(data=da_nao_3.to_series(), palette = "PuBuGn_d", linewidth=2.5)
sns.despine();

# sst around the world
plt.figure(figsize=(20,10))
ax = plt.axes(projection=ccrs.PlateCarree())
da_sgl.sel(time='2012-07-16').plot(levels=10,cmap='RdBu_r', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree(),cbar_kwargs={'label': 'Temperature anomaly in Celsius degree'})
ax.set_title("SST anomaliy in December 2011 with respect to the 1960-1990 period")
ax.set_global(); ax.coastlines();

#temperature in the US
plt.figure(figsize=(20,10)) 
ax = plt.axes(projection=ccrs.PlateCarree())
da_tempg.sel(time='2012-07-16').plot(levels=10,cmap='RdBu_r', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree(),cbar_kwargs={'label': 'Temperature anomaly in Celsius degree'})
ax.set_title("Temperature anomaliy in July 2012 with respect to the 1960-1990 period")
ax.set_global(); ax.coastlines();
ax.set_global(), ccrs.PlateCarree()

# temperature in south america
plt.figure(figsize=(20,10)) 
ax = plt.axes(projection=ccrs.PlateCarree())
da_tempg.sel(time='2012-1-16').plot(levels=5,cmap='RdBu_r', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree(),cbar_kwargs={'label': 'Temperature anomaly in Celsius degree'})
ax.set_title("Temperature anomaliy in December 2011 with respect to the 1960-1990 period")
ax.coastlines();
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.set_global(); ax.coastlines();


