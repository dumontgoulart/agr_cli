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

#%% data import
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
us1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())

#model yield
DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel( time=slice(1,31))
DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y = DS_y.rename({'lon': 'longitude','lat': 'latitude'})
DS_y_us = mask_shape_border(DS_y,soy_us_states ) #clipping for us

DS_y_br = mask_shape_border(DS_y,soy_br_states ) #clipping for us

#data yield iizumi
ds_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc").sel(time = slice(1982,2016))
ds_iizumi_us= mask_shape_border(ds_iizumi,'gadm36_USA_0.shp' ) #clipping for us
ds_iizumi_br= mask_shape_border(ds_iizumi,'gadm36_BRA_0.shp' ) #clipping for br

da_iizumi_us = ds_iizumi_us['yield'].sel(longitude=slice(-100.25,-80.25),latitude=slice(50.25,30.25))
da_iizumi_br = ds_iizumi_br['yield'].sel(longitude=slice(-63.25,-40.25),latitude=slice(-5.25,-35.25))

#climate
DS_cli=xr.open_dataset("temp_evp_prec_era5_monthly.nc").sel(time=slice('1981-09-01','2012-03-31'),longitude=slice(-63.25,-40.25),latitude=slice(-5.25,-32.75))
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

#local climate
DS_cli=xr.open_dataset("global_cli_2018.nc")
DS_cli = DS_cli.sel(time=slice('1982-01-01','2016-12-31'),longitude=slice(-150,0), latitude = slice(60,-60))
#US-shape
da_cli_us = mask_shape_border(DS_cli,'gadm36_USA_0.shp' ).t2m
da_cli_us_mean = (da_cli_us.to_dataframe().groupby(['time']).mean()).to_xarray().t2m
da_cli_us_mean = da_cli_us_mean  - 273.15

da_cli_br = mask_shape_border(DS_cli,'gadm36_BRA_0.shp' ).t2m
da_cli_br_mean = (da_cli_br.to_dataframe().groupby(['time']).mean()).to_xarray().t2m
da_cli_br_mean = da_cli_br_mean  - 273.15

# YEARLY AVERAGE VALUES
DS_cli['time'] = DS_cli['growing_year']
DS_cli_year = DS_cli.groupby(DS_cli['time']).mean(keep_attrs=True)
DS_cli_year['time'] = pd.to_datetime(DS_cli_year.time, format='%Y')

data_temp_month = DS_cli.t2m.sel(time=DS_cli['time.month'] == 1 )
data_temp_month = data_temp_month.to_dataframe().groupby(['time']).mean()

# # only use below if sure it's yearly averaged
# DS_cli=DS_cli_year
# SPECIFIC MONTH VALUES
# and convert data of climate to yield so the two datasets are aligned and the choice of month is indepedent.
# DS_cli = DS_cli.sel(time=DS_cli['time.month']==11)
# DS_cli['time'] = DS_y['time'] 

#CRU data
DS_t_max=xr.open_dataset("cru/cru_tmx.nc",decode_times=True).sel(time=slice('1980-01-01','2016-12-31'))
DS_t_max = DS_t_max.rename({'lon': 'longitude','lat': 'latitude'})
DS_t_max.tmx.isel(time=0).plot(x='longitude', y = 'latitude')
DS_t_max_us = mask_shape_border(DS_t_max, soy_us_states) #US-shape
DS_t_max_br = mask_shape_border(DS_t_max, soy_br_states) #US-shape

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

data_tmax= DS_t_max_us.tmx.where(DS_t_max_us.tmx > -100, -110)
mean_temp = np.nanmean(data_tmax, axis=0)
data_tmax = xr.DataArray(signal.detrend(data_tmax, axis=0), dims=data_tmax.dims, coords=data_tmax.coords, attrs = data_tmax.attrs) + mean_temp
data_tmax_us = data_tmax.where(data_tmax > -100, np.nan ).sel(time = data_tmax.indexes['time'].month.isin([6, 7, 8,9,10])) 

data_tmax= DS_t_max_br.tmx.where(DS_t_max_br.tmx > -100, -110)
mean_temp = np.nanmean(data_tmax, axis=0)
data_tmax = xr.DataArray(signal.detrend(data_tmax, axis=0), dims=data_tmax.dims, coords=data_tmax.coords, attrs = data_tmax.attrs) + mean_temp
data_tmax_br = data_tmax.where(data_tmax > -100, np.nan ).sel(time = data_tmax.indexes['time'].month.isin([1,2,3,4])) 


#tests
ts1 = data_temp.sel(latitude=-20.25, longitude=-53.25)
ts2 = data_temp.sel(latitude=-30.25, longitude=-55.25)
ts3 = data_temp.sel(latitude=-30.25, longitude=-53.25)

#detrend climate us
da_cli_us_det = da_cli_us.where(da_cli_us > 0, 0 )
mean_cli = da_cli_us_det.mean(axis=0)
da_cli_us_det1 =  xr.DataArray(signal.detrend(da_cli_us_det, axis=0), dims=da_cli_us_det.dims, coords=da_cli_us_det.coords, attrs=da_cli_us_det.attrs) + mean_cli
da_cli_us_det2 = da_cli_us_det1.where(da_cli_us_det1 > 0, np.nan ).sel(time = slice('1982','2015'))

#detrend climate br
da_cli_br_det = da_cli_br.where(da_cli_br > 0, 0 )
mean_cli_br = da_cli_br_det.mean(axis=0)
da_cli_br_det1 =  xr.DataArray(signal.detrend(da_cli_br_det, axis=0), dims=da_cli_br_det.dims, coords=da_cli_br_det.coords, attrs=da_cli_br_det.attrs) + mean_cli_br
da_cli_br_det2 = da_cli_br_det1.where(da_cli_br_det1 > 0, np.nan ).sel(time = slice('1982','2015'))

# detrend yield
da_iizumi_us_det = da_iizumi_us.where(da_iizumi_us > 0, 0 )
mean_us = da_iizumi_us_det.mean(axis=0)
da_iizumi_us_det_1 = xr.DataArray(signal.detrend(da_iizumi_us_det, axis=0), dims=da_iizumi_us_det.dims, coords=da_iizumi_us_det.coords, attrs=da_iizumi_us_det.attrs) + mean_us
da_iizumi_us_det_2 = da_iizumi_us_det_1.where(da_iizumi_us_det_1 > 0, np.nan ).sel(time = slice('1982','2015'))

da_iizumi_br_det = da_iizumi_br.where(da_iizumi_br > 0, 0 )
mean_br = da_iizumi_br_det.mean(axis=0)
da_iizumi_br_det_1 = xr.DataArray(signal.detrend(da_iizumi_br_det, axis=0), dims=da_iizumi_br_det.dims, coords=da_iizumi_br_det.coords, attrs=da_iizumi_br_det.attrs) + mean_br
da_iizumi_br_det_2 = da_iizumi_br_det_1.where(da_iizumi_br_det_1 > 0, np.nan ).sel(time = slice('1982','2015'))

#%% Plotting results different climate
df_ii_us = da_iizumi_us_det_2.to_series().groupby(['time']).mean()
df_ii_us.index = da_cli_us_det2.sel(time = da_cli_us_det2.indexes['time'].month.values == 11).indexes['time'].values
df_ii_us.plot()

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

# plot scatterplot temperature and precipitation conditioned by la nina

plt.scatter(data_temp_w, data3, c='black')



#%% Climate for each month and see correlation
da_iizumi_us_det_2 =DS_y_us['yield'].sel(latitude = slice(60.25,10.25))
da_iizumi_br_det_2 = DS_y_br['yield'].sel(latitude = slice(0.25,-50.25))
da_cli_us_det2 = data_tmax_us
da_cli_br_det2 = data_tmax_br

monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
values=[]
for j in [da_iizumi_us_det_2,da_iizumi_br_det_2 ]: #, da_iizumi_us_det_2
    for i in [1,2,3,6,7,8,9]:
        if len(j.latitude) == len(da_iizumi_us_det_2.latitude):
            data_temp_month = da_cli_us_det2.sel(time=da_cli_us_det2['time.month'] == i )
        elif len(j.latitude) == len(da_iizumi_br_det_2.latitude):
            data_temp_month = da_cli_br_det2.sel(time=da_cli_br_det2['time.month'] == i )
        data_temp_month['time'] = data_temp_month.indexes['time'].year
        cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_temp_month,y=j)
        cor_mean=format(cor.mean().values, '.4f')
        values.append(cor_mean)
        plt.figure(figsize=(15,10)) 
        ax=plt.axes(projection=ccrs.PlateCarree())
        cor.plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu',levels=15)
        ax.set_title(f'Temperature and yield - Month: {monthDict[i]}; Mean cor:{cor_mean} ')
        ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
        if len(j.latitude) == len(da_iizumi_us_det_2.latitude):
            ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
            ax.set_extent([-110,-75,25,50], ccrs.PlateCarree())
        elif len(j.latitude) == len(da_iizumi_br_det_2.latitude):
            ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
            ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
        plt.show()
   
data_tmax8 = data_tmax.sel( time=data_tmax['time.month'] == 8, longitude=slice(-100.25,-80.25),latitude=slice(30.25,50.25) )
data_tmax8 = data_tmax8.sel(time = slice('1980','2010'))
data_tmax8['time'] = data_tmax8.indexes['time'].year
cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=data_tmax8,y=da_iizumi_us_det_2)
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
adm1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
cor.plot(levels=10,x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation'}, cmap='RdBu')
ax.set_title(f'Weighted temperature and yield - Growing {i}; Harvesting {1-i}')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-102,-80,30,50], ccrs.PlateCarree())
plt.show()




#%% test on apei and tmax8

#%% climate CRU
DS_t_mean=xr.open_dataset("cru/cru_tmp.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_t_max=xr.open_dataset("cru/cru_tmx.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_t_min=xr.open_dataset("cru/cru_tmn.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_prec=xr.open_dataset("cru/cru_pre.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_evap=xr.open_dataset("cru/cru_vap.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_wet=xr.open_dataset("cru/cru_wet.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))
DS_spei = xr.open_dataset("spei02.nc",decode_times=True).sel(time=slice('1980-01-01','2015-12-31'))

DS_cli = xr.merge([DS_prec.pre,DS_t_max.tmx,DS_evap.vap,DS_wet['wet'].dt.days, DS_spei.spei]).sel(time=slice('1980-01-01','2010-12-31'))
DS_cli_us = mask_shape_border(DS_cli, soy_us_states) #US-shape
DS_cli_det_us = DS_cli_us.where(DS_cli_us.tmx > -300, -40000 )

# df_tmax_f = detrend_dataset(DS_cli_det_us.tmx,months_to_select =[6, 7, 8,9,10] )
# df_prec_f = detrend_dataset(DS_cli_det_us.pre,months_to_select =[6, 7, 8,9,10] )
# df_e_f = detrend_dataset(DS_cli_det_us.vap,months_to_select =[6, 7, 8,9,10] )
# df_wet_f = detrend_dataset(DS_cli_det_us['days'],months_to_select =[6, 7, 8,9,10] )

#%% detrend climate CRU

#temp_max
da_cli_us_det_tmax = DS_cli_us.tmx.where(DS_cli_us.tmx > -300, -30000 )
mean_cli = da_cli_us_det_tmax.mean(axis=0)
da_cli_us_det_tmax1 =  xr.DataArray(signal.detrend(da_cli_us_det_tmax, axis=0), dims=da_cli_us_det_tmax.dims, coords=da_cli_us_det_tmax.coords, attrs=da_cli_us_det_tmax.attrs) + mean_cli
da_cli_us_det_tmax2 = da_cli_us_det_tmax1.where(da_cli_us_det_tmax1 > -100, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin([6, 7, 8,9,10])) 
da_cli_us_det_tmax_mean = da_cli_us_det_tmax2.groupby('time').mean(...)
df_tmax=da_cli_us_det_tmax_mean.to_series()


#SPEI 
da_cli_us_det_spei = DS_cli_us.spei.where(DS_cli_us.tmx > -300, -30000)
mean_cli = da_cli_us_det_spei.mean(axis=0)
da_cli_us_det_spei1 =  xr.DataArray(signal.detrend(da_cli_us_det_spei, axis=0), dims=da_cli_us_det_spei.dims, coords=da_cli_us_det_spei.coords, attrs=da_cli_us_det_spei.attrs) + mean_cli
da_cli_us_det_spei2 = da_cli_us_det_spei1.where(DS_cli_us.tmx > -300, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin([6, 7, 8,9,10])) 
da_cli_us_det_spei_mean = da_cli_us_det_spei2.groupby('time').mean(...)
df_spei=da_cli_us_det_spei_mean.to_series()

da_test = da_cli_us_det_tmax2.sel( time=da_cli_us_det_tmax2['time.month'] == 8)
da_test['time'] = da_test.indexes['time'].year

#tmax8 x spei
cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=da_cli_us_det_tmax2.sel( time=da_cli_us_det_tmax2['time.month'] == 8),y=da_cli_us_det_spei.sel( time=da_cli_us_det_spei['time.month'] == 8))
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
cor.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation temperature max 8 and SPEI 2 months 8 at grid level'}, cmap='RdBu', levels=10)
ax.set_title('Temperature and SPEI at grid level')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()

#yield x tmax8
cov,cor,slope,intercept,pval,stderr = lag_linregress_3D(x=da_test,y=DS_y['yield'])
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.PlateCarree())
cor.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation temperature max 8 and SPEI 2 months 8 at grid level'}, cmap='RdBu', levels=10)
ax.set_title('Temperature and SPEI at grid level')
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()
