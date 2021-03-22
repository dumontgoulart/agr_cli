# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:45:27 2020

@author: morenodu
"""
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import seaborn as sns
from  scipy import signal 
from mask_shape_border import mask_shape_border
from lag_linregress_3D import lag_linregress_3D
import matplotlib.ticker as plticker


# Crop space to either US or soy states
usa = gpd.read_file('gadm36_USA_1.shp', crs="epsg:4326") 
us1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
state_names = ['Iowa','Illinois','Minnesota','Indiana','Nebraska','Ohio', 'South Dakota','North Dakota', 'Missouri','Arkansas']
soy_us_states = usa[usa['NAME_1'].isin( state_names)]
# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_br_names = ['Mato Grosso','Rio Grande do Sul','Paraná']
soy_br_states = bra[bra['NAME_1'].isin(state_br_names)]


DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(time=slice(1,31))
DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
DS_y_us = mask_shape_border(DS_y,soy_us_states ) #clipping for us
df_wofost=DS_y_us.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore

DS_t_mean=xr.open_dataset("cru/cru_tmp.nc",decode_times=True).sel(time=slice('1980-01-01','2016-12-31'))
DS_t_max=xr.open_dataset("cru/cru_tmx.nc",decode_times=True).sel(time=slice('1980-01-01','2016-12-31'))
DS_t_min=xr.open_dataset("cru/cru_tmn.nc",decode_times=True).sel(time=slice('1980-01-01','2016-12-31'))
DS_prec=xr.open_dataset("cru/cru_pre.nc",decode_times=True).sel(time=slice('1980-01-01','2016-12-31'))
DS_evap=xr.open_dataset("cru/cru_vap.nc",decode_times=True).sel(time=slice('1980-01-01','2016-12-31'))
DS_wet=xr.open_dataset("cru/cru_wet.nc",decode_times=True).sel(time=slice('1980-01-01','2016-12-31'))

DS_cli = xr.merge([DS_t_mean.tmp,DS_prec.pre,DS_t_max.tmx,DS_evap.vap,DS_wet]).sel(time=slice('1980-01-01','2010-12-31'))
DS_cli_us = mask_shape_border(DS_cli, soy_us_states) #US-shape


#%% detrend climate CRU
#temp
da_cli_us_det_temp = DS_cli_us.tmp.where(DS_cli_us.tmp > -300, -300 )
mean_cli = da_cli_us_det_temp.mean(axis=0)
da_cli_us_det_temp1 =  xr.DataArray(signal.detrend(da_cli_us_det_temp, axis=0), dims=da_cli_us_det_temp.dims, coords=da_cli_us_det_temp.coords, attrs=da_cli_us_det_temp.attrs) + mean_cli
da_cli_us_det_temp2 = da_cli_us_det_temp1.where(da_cli_us_det_temp1 > -100, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin([6, 7, 8,9,10])) 
da_cli_us_det_temp_mean = da_cli_us_det_temp2.groupby('time').mean(...)
df_temp=da_cli_us_det_temp_mean.to_series()

#temp_max
da_cli_us_det_tmax = DS_cli_us.tmx.where(DS_cli_us.tmx > -300, -300 )
mean_cli = da_cli_us_det_tmax.mean(axis=0)
da_cli_us_det_tmax1 =  xr.DataArray(signal.detrend(da_cli_us_det_tmax, axis=0), dims=da_cli_us_det_tmax.dims, coords=da_cli_us_det_tmax.coords, attrs=da_cli_us_det_tmax.attrs) + mean_cli
da_cli_us_det_tmax2 = da_cli_us_det_tmax1.where(da_cli_us_det_tmax1 > -100, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin([6, 7, 8,9,10])) 
da_cli_us_det_tmax_mean = da_cli_us_det_tmax2.groupby('time').mean(...)
df_tmax=da_cli_us_det_tmax_mean.to_series()

#prec
da_cli_us_det_prec = DS_cli_us.pre.where(DS_cli_us.tmp > -300, -1 )
mean_cli = da_cli_us_det_prec.mean(axis=0)
da_cli_us_det_prec1 =  xr.DataArray(signal.detrend(da_cli_us_det_prec, axis=0), dims=da_cli_us_det_prec.dims, coords=da_cli_us_det_prec.coords, attrs=da_cli_us_det_prec.attrs) + mean_cli
da_cli_us_det_prec2 = da_cli_us_det_prec1.where(da_cli_us_det_prec1 > -1, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin([6, 7, 8,9,10])) 
da_cli_us_det_prec_mean = da_cli_us_det_prec2.groupby('time').mean(...)
df_prec=da_cli_us_det_prec_mean.to_series()

#vap
da_cli_us_det_e = DS_cli_us.vap.where(DS_cli_us.tmp > -300, -300)
mean_cli = da_cli_us_det_e.mean(axis=0)
da_cli_us_det_e1 =  xr.DataArray(signal.detrend(da_cli_us_det_e, axis=0), dims=da_cli_us_det_e.dims, coords=da_cli_us_det_e.coords, attrs=da_cli_us_det_e.attrs) + mean_cli
da_cli_us_det_e2 = da_cli_us_det_e1.where(DS_cli_us.tmp > -300, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin([6, 7, 8,9,10])) 
da_cli_us_det_e_mean = da_cli_us_det_e2.groupby('time').mean(...)
df_e=da_cli_us_det_e_mean.to_series()
da_wet = DS_cli_us.wet
da_wet = DS_cli_us['wet'].dt.days
#wet
da_cli_us_det_wet = DS_cli_us['wet'].dt.days.where(DS_cli_us.tmp > -300, -300)
mean_cli = da_cli_us_det_wet.mean(axis=0)
da_cli_us_det_wet1 =  xr.DataArray(signal.detrend(da_cli_us_det_wet, axis=0), dims=da_cli_us_det_wet.dims, coords=da_cli_us_det_wet.coords, attrs=da_cli_us_det_wet.attrs) + mean_cli
da_cli_us_det_wet2 = da_cli_us_det_wet1.where(DS_cli_us.tmp > -300, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin([6, 7, 8,9,10])) 
da_cli_us_det_wet_mean = da_cli_us_det_wet2.groupby('time').mean(...)
df_wet=da_cli_us_det_wet_mean.to_series()


#%% US states
da_cli_tmax8 = da_cli_us_det_tmax2.sel(time = da_cli_us_det_tmax2.indexes['time'].month.isin([8])) 
da_cli_tmax8['time'] = DS_y_us['time']
da_cli_wet7 = da_cli_us_det_wet2.sel(time = da_cli_us_det_wet2.indexes['time'].month.isin([7])) 
da_cli_wet7['time'] = DS_y_us['time']


cov,cor,slope, intercept,pval,stderr = lag_linregress_3D(x=da_cli_tmax8,y=DS_y_us['yield'])
plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.Mercator())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
r2_score3D.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'R2 tmax8 and yield'},   levels=15)
loc = plticker.MultipleLocator(base=2.0) # this locator puts ticks at regular intervals
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()

r2_score3D = cor**2
predict = intercept + slope*da_cli_tmax8
rss = ((DS_y_us['yield'] - predict)**2).mean('time')

#%% second trial skleran-xarray
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, make_union




from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(X_train.tmax8, y_train.values.ravel())

import statsmodels.api as sm
X_const = pd.concat([X_train.tmax8,X_train.wet7], axis=1, sort=False)
X_const = sm.add_constant(X_const)
model = sm.OLS(y_train,X_const)
results = model.fit()
results.params
print(results.summary())



coeff, r, rank, s = np.linalg.lstsq(X_const, y_train, rcond=0)
print(coeff, r, rank, s)


x1, x2, y = X_train.tmax8, X_train.wet7, y_train.values.ravel()

n = y.notnull().sum(dim='time')
x1mean=np.nanmean(x1,axis=0)
x2mean=np.nanmean(x2,axis=0)
ymean=np.nanmean(y,axis=0)
x1std=np.nanstd(x1,axis=0)
x2std=np.nanstd(x2,axis=0)
ystd=np.nanstd(y,axis=0)

#4. Compute covariance along time axis
cov   =  np.sum((x1 - x1mean)*(x2 - x2mean)*(y - ymean), axis=0) / (n)

#5. Compute correlation along time axis
cor   = cov/(x1std*x2std*ystd)
#6. Compute regression slope and intercept:
slope    = [cov/(x1std**2),cov/(x2std**2)]

intercept = ymean - x1mean*slope[0]  - x2mean*slope[1]  



#%%






def k_cor(x,y, pthres = 0.05, direction = True):
    """
    Uses the scipy stats module to calculate a Kendall correlation test
    :x vector: Input pixel vector to run tests on
    :y vector: The date input vector
    :pthres: Significance of the underlying test
    :direction: output only direction as output (-1 & 1)
    """
    # Check NA values
   
    
    # Run the kendalltau test
    tau, p_value = stats.kendalltau(x, y)

    # Criterium to return results in case of Significance
    if p_value < pthres:
        # Check direction
        if direction:
            if tau < 0:
                return -1
            elif tau > 0:
                return 1
        else:
            return tau
    else:
      return 0  

# The function we are going to use for applying our kendal test per pixel
def kendall_correlation(x,y,dim='year'):
    x,y = xr.align(x,y)
    # x = Pixel value, y = a vector containing the date, dim == dimension
    return xr.apply_ufunc(
        k_cor, x , y,
        input_core_dims=[[dim], [dim]],
        vectorize=True, # !Important!
        output_dtypes=[int]
        )




r = kendall_correlation(da_cli_tmax8, DS_y_us['yield'],'time')      


plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.Mercator())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
stats.sel(linreg_pam=2).plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation temperature and evaporation'},  cmap='RdBu', levels=15)
loc = plticker.MultipleLocator(base=2.0) # this locator puts ticks at regular intervals
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()

import statsmodels.api as sm

def new_linregress(x, y):
    regr = LinearRegression()
    regr.fit(x, y)
    score = regr.score(x, y)
    return score
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #return np.array([slope, intercept, r_value, p_value, std_err])
    
    
               
               
               
da_y_us = DS_y_us['yield']
da_cli_tmax8 = da_cli_tmax8.reindex(lat=da_cli_tmax8.lat[::-1])
x,y = xr.align(da_cli_tmax8, da_y_us, join="exact") #check with they match


stats = xr.apply_ufunc(new_linregress, [da_cli_tmax8,da_cli_tmax8], da_y_us,
                       input_core_dims=[['time'], ['time']],
                       output_core_dims=[["linreg_pam"]],
                       vectorize=True,
                       dask="parallelized",
                       output_dtypes=['float64'],
                       output_sizes={"linreg_pam": 1},
                      )











