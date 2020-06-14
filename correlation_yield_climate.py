# Correlation script
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as plticker
from  scipy import stats, signal #Required for detrending data and computing regression
from lag_linregress_3D import lag_linregress_3D
from mask_shape_border import mask_shape_border

adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())

#model yield
DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(lon=slice(-61.25,-44.25),lat=slice(-5.25,-32.75), time=slice(1,31))
DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y = DS_y.rename({'lon': 'longitude','lat': 'latitude'})

#data yield iizumi
ds_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc").sel(time = slice(1982,2016))
da_iizumi =ds_iizumi['yield']
da_iizumi_us = ds_iizumi['yield']
da_iizumi_br = ds_iizumi['yield'].sel(longitude=slice(-61.25,-44.25),latitude=slice(-5.25,-32.75))

ax = plt.axes(projection=ccrs.PlateCarree()) #####plot projection at time T
da_iizumi_us.isel(time = 34).plot(levels=20,cmap='YlGn', x='longitude',y='latitude',robust=True,ax=ax, transform=ccrs.PlateCarree());
# ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-63,-42,-35,-8], ccrs.PlateCarree())

#climate
DS_cli=xr.open_dataset("temp_evp_prec_era5_monthly.nc").sel(time=slice('1979-09-01','2012-03-31'),longitude=slice(-61.25,-44.25),latitude=slice(-5.25,-32.75))
DS_cli['t2m']=DS_cli.t2m -273.15
DS_cli['tp']=DS_cli.tp * 1000
DS_cli['e']=DS_cli.e * 1000
DS_cli.t2m.attrs = {'units': 'Celcius degree', 'long_name': '2 metre temperature'}
DS_cli.tp.attrs = {'units': 'mm', 'long_name': 'Total precipitation'}
DS_cli.e.attrs = {'units': 'mm of water equivalent', 'long_name': 'Evaporation', 'standard_name': 'lwe_thickness_of_water_evaporation_amount'}
DS_cli['growing_month'] = DS_cli["time.month"]
DS_cli['growing_year'] = DS_cli["time.year"]
for i in range(len(DS_cli["time"].values)):
    if DS_cli["time.month"].values[i] < 9:
        DS_cli['growing_year'][i] = DS_cli["time.year"][i]
    else:
        DS_cli["growing_year"][i] = DS_cli["time.year"][i]+1    
DS_cli['growing_month'].values = np.array(list(np.arange(1,8))*(DS_cli['time.year'].values[-1]-DS_cli['time.year'].values[0]))

DS_cli.t2m.mean()

# YEARLY AVERAGE VALUES
# DS_cli['time'] = DS_cli['growing_year']
# DS_cli_year = DS_cli.groupby(DS_cli['time']).mean(keep_attrs=True)
# DS_cli_year['time'] = pd.to_datetime(DS_cli_year.time, format='%Y')
# # only use below if sure it's yearly averaged
# DS_cli=DS_cli_year
# SPECIFIC MONTH VALUES
# and convert data of climate to yield so the two datasets are aligned and the choice of month is indepedent.
# DS_cli = DS_cli.sel(time=DS_cli['time.month']==11)
# DS_cli['time'] = DS_y['time'] 
#%% Operations for correlation, covariance, etc...

#regularize data
data_prec = DS_cli.tp
mean_prec = data_prec.mean(axis=0)
data_prec = xr.DataArray(signal.detrend(data_prec, axis=0), dims=data_prec.dims, coords=data_prec.coords, attrs = data_prec.attrs) + mean_prec

data_temp=DS_cli.t2m
mean_temp = data_temp.mean(axis=0)
data_temp = xr.DataArray(signal.detrend(data_temp, axis=0), dims=data_temp.dims, coords=data_temp.coords, attrs = data_temp.attrs) + mean_temp

data_e=DS_cli.e
mean_e = data_e.mean(axis=0)
data_e = xr.DataArray(signal.detrend(data_e, axis=0), dims=data_e.dims, coords=data_e.coords, attrs = data_e.attrs) + mean_e
#tests
ts1 = data_temp.sel(latitude=-20.25, longitude=-53.25)
ts2 = data_temp.sel(latitude=-30.25, longitude=-55.25)
ts3 = data_temp.sel(latitude=-30.25, longitude=-53.25)

data3=da_iizumi_br
data3 = data3.fillna(99999999999 )
mean3 = data3.mean(axis=0)
data3 = xr.DataArray(signal.detrend(data3, axis=0), dims=data3.dims, coords=data3.coords, attrs=data3.attrs) + mean3
data3 = data3.where(data3<30)


#%% Plotting results different climate

cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=ts2,y=ts1)
print("correlation: " "%.4f" % cor.values, "R2: "  "%.4f" % cor.values**2, "P-value: " "%.4f" % pval.values)
plt.plot(ts2, ts1, 'o', label='original data')
plt.plot(ts2, intercept + slope*ts2, 'r', label='fitted line')
plt.xlabel('Celsius degree')
plt.ylabel('Celsius degree')
plt.legend()
plt.show()
cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=ts2,y=ts3)
print("correlation: " "%.4f" % cor.values, "R2: "  "%.4f" % cor.values**2, "P-value: " "%.4f" % pval.values)
plt.plot(ts2, ts3, 'o', label='original data')
plt.plot(ts2, intercept + slope*ts2, 'r', label='fitted line')
plt.xlabel('Celsius degree')
plt.ylabel('Celsius degree')
plt.legend()
plt.show()

cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=ts2,y=data_temp)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation of point [-30, -44]'}, cmap='RdBu_r')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
plt.show()

cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_temp,y=data_prec)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation precpitation and temperature'}, cmap='RdBu')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
plt.show()

cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_prec,y=data_e)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation precpitation and evaporation'}, cmap='RdBu')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
plt.show()

cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_temp,y=data_e)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation temperature and evaporation'},  cmap='RdBu')
loc = plticker.MultipleLocator(base=2.0) # this locator puts ticks at regular intervals
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
plt.show()

# code is right, but its not considering which month, so commented out
# # yield and climate
# cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_prec,y=data3)
# plt.figure(figsize=(20,10)) 
# ax=plt.axes(projection=ccrs.PlateCarree())
# adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
# cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu')
# ax.set_title('Precpitation and yield at grid level')
# ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
# ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
# plt.show()

# cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_temp,y=data3)
# plt.figure(figsize=(20,10)) 
# ax=plt.axes(projection=ccrs.PlateCarree())
# adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
# cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation temperature and yield at grid level'}, cmap='RdBu')
# ax.set_title('Temperature and yield at grid level')
# ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
# ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
# plt.show()

# cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_e,y=data3)
# plt.figure(figsize=(20,10)) 
# ax=plt.axes(projection=ccrs.PlateCarree())
# adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
# cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu')
# ax.set_title('Correlation evaporation and yield at grid level')
# ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
# ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
# plt.show()

#%% define 1-4 x1, 4-7 x2 and weight shceme 0, 0.25, 0.5, 0.75 - 1-alpha
alpha=[0, 0.25, 0.5, 0.75, 1]
data_temp_sem1 = data_temp.sel(time=DS_cli['time.month'] >= 9 )
data_temp_sem1 = data_temp_sem1.groupby(data_temp_sem1['time.year']).mean(keep_attrs=True)
data_temp_sem1 = data_temp_sem1.rename({'year': 'time'})
data_temp_sem2 = data_temp.sel(time=DS_cli['time.month'] < 9 )
data_temp_sem2 = data_temp_sem2.groupby(data_temp_sem2['time.year']).mean(keep_attrs=True)
data_temp_sem2 = data_temp_sem2.rename({'year': 'time'})

values=[]
for i in alpha:
    data_temp_w = data_temp_sem1*i + data_temp_sem2*(1-i)
    cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_temp_w,y=data3)
    values.append(cor.mean())
    print(cor.mean())
    plt.figure(figsize=(20,10)) 
    ax=plt.axes(projection=ccrs.PlateCarree())
    adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
    cor.plot(levels=10,x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu')
    ax.set_title(f'Weighted temperature and yield - Growing {i}; Harvesting {1-i}')
    ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
    plt.show()

# plot sacatterplot temperature and precipitation conditioned by la nina

plt.scatter(data_temp_w, data3, c='black')



#%% Climate for each month and see correlation
monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
values=[]
for i in [12]:
    data_temp_month = data_temp.sel(time=DS_cli['time.month'] == i )
    data_temp_month['time'] = DS_y['time'] 
    cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_temp_month,y=data3)
    cor_mean=format(cor.mean().values, '.4f')
    values.append(cor_mean)
    plt.figure(figsize=(10,5)) 
    ax=plt.axes(projection=ccrs.PlateCarree())
    adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
    cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu')
    ax.set_title(f'Temperature and yield - Month: {monthDict[i]}; Mean cor:{cor_mean} ')
    ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
    plt.show()
