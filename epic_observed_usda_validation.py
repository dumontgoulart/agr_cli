# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:53:27 2021

@author: morenodu
"""


import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from  scipy import signal 
import seaborn as sns
from mask_shape_border import mask_shape_border
from failure_probability import feature_importance_selection, failure_probability
from stochastic_optimization_Algorithm import stochastic_optimization_Algorithm
from shap_prop import shap_prop
from bias_correction_masked import *
import matplotlib as mpl
import pickle
#%% Compare EPIC with USDA observed yields
DS_ref_2 = xr.open_dataset("ACY_gswp3-w5e5_obsclim_2015soc_default_soy_noirr.nc", decode_times=True).sel(time=slice('1960-01-01','2016-12-12'), lon=slice(-160,-10))
DS_usda = xr.open_dataset("soy_yields_usda_05x05.nc", decode_times=False).sel(lon=slice(-160,-10))
DS_usda['time'] = pd.date_range(start='1960', periods=DS_usda.sizes['time'], freq='YS').year
DS_usda = DS_usda.sel(time=slice('1960-01-01','2016-12-12'))
DS_usda['usda_yield'] = DS_usda['usda_yield'] * 0.06725106937 

plot_2d_us_map(DS_usda["usda_yield"].mean('time'))

# Mask US states
DS_ref_2 = mask_shape_border(DS_ref_2,soy_us_states) #US_shape

# Mask for MIRCA 2000 each tile >0.9 rainfed
ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")

DS_ref_2 = DS_ref_2.where(ds_mask['soybean_rainfed'] > 0.9 )
DS_ref_2 = DS_ref_2.dropna(dim = 'lon', how='all')
DS_ref_2 = DS_ref_2.dropna(dim = 'lat', how='all')
if len(DS_ref_2.coords) >3 :
    DS_ref_2=DS_ref_2.drop('spatial_ref')
    

# Mask US states
DS_usda = mask_shape_border(DS_usda,soy_us_states) #US_shape

DS_usda = DS_usda.where(ds_mask['soybean_rainfed'] > 0.9 )
DS_usda = DS_usda.dropna(dim = 'lon', how='all')
DS_usda = DS_usda.dropna(dim = 'lat', how='all')
if len(DS_usda.coords) >3 :
    DS_usda=DS_usda.drop('spatial_ref')
    
    
plt.plot(DS_usda["usda_yield"].mean(['lat','lon']), label = 'Observed yield')
plt.plot(DS_ref_2["yield"].mean(['lat','lon']), label = 'EPIC-IIASA yield')
plt.legend()
plt.show()


plt.figure(figsize=(10,5), dpi=250) #plot clusters
ax=plt.axes(projection=ccrs.LambertConformal())
DS_usda['usda_yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-115,-67,25,50], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.6)


def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

DS_ref_2_det = xr.DataArray( detrend_dim(DS_ref_2['yield'], 'time') + DS_ref_2['yield'].mean('time'), name= DS_ref_2['yield'].name, attrs = DS_ref_2['yield'].attrs)
DS_usda_det = xr.DataArray( detrend_dim(DS_usda['usda_yield'], 'time') + DS_usda['usda_yield'].mean('time'), name= DS_usda['usda_yield'].name, attrs = DS_usda['usda_yield'].attrs)

df_ref_2_det = DS_ref_2_det.mean(['lat','lon']).to_dataframe()
df_usda_det = DS_usda_det.mean(['lat','lon']).to_dataframe()


from sklearn import preprocessing
scaler_usda = preprocessing.StandardScaler().fit(df_usda_det)
usda_scaled = scaler_usda.transform(df_usda_det)
df_usda_scaled = pd.DataFrame(usda_scaled, index=df_usda_det.index, columns = df_usda_det.columns)

scaler_epic = preprocessing.StandardScaler().fit(df_ref_2_det)
epic_scaled = scaler_epic.transform(df_ref_2_det)
df_epic_scaled = pd.DataFrame(epic_scaled, index=df_ref_2_det.index, columns = df_ref_2_det.columns)


plt.plot(df_ref_2_det.index,df_ref_2_det['yield'], label = 'EPIC-IIASA yield')
plt.plot(df_usda_det.index, df_usda_det['usda_yield'], label = 'Observed yield')
plt.legend()
plt.show()

plt.figure(figsize=(5,4), dpi=250) #plot clusters
plt.plot(df_epic_scaled.index,df_epic_scaled['yield'], label = 'EPIC-IIASA yield')
plt.plot(df_usda_scaled.index, df_usda_scaled['usda_yield'], label = 'Observed yield')
plt.ylabel('Standardized yield')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
plt.show()


plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_usda_scaled['usda_yield'],df_epic_scaled['yield'])
plt.plot(df_usda_scaled['usda_yield'].sort_values(), df_usda_scaled['usda_yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('Standardized yield')
plt.xlabel('Standardized yield')
plt.legend()
plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
plt.show()


# Pearson's correlation
from scipy.stats import pearsonr

corr_grouped, _ = pearsonr(df_usda_det['usda_yield'], df_ref_2_det['yield'])
print('Pearsons correlation: %.3f' % corr_grouped)

plt.figure(figsize=(10,5), dpi=250) #plot clusters
ax=plt.axes(projection=ccrs.LambertConformal())
DS_usda_det.mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-115,-67,25,50], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.6)

corr_3d_epic = xr.corr(DS_ref_2_det, DS_usda_det, dim="time", )
plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
corr_3d_epic.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, levels = 10)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-115,-67,25,50], ccrs.PlateCarree())
plt.show()

corr_3d_high = corr_3d_epic.where(corr_3d_epic > 0.5)
plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
corr_3d_high.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, levels = 10)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-115,-67,25,50], ccrs.PlateCarree())
plt.show()



#%%
def conversion_clim_yield_test(DS_cli_us, df_co2, months_to_be_used=[7,8], detrend = False):
    
    # Meteorological features considered for this case, transformation to dataframe for the select months of the season
    column_names = [i+str(j) for i in list(DS_cli_us.keys()) for j in months_to_be_used]
    
    df_features_avg_list = []
    for feature in list(DS_cli_us.keys()):        
        df_feature = DS_cli_us[feature].to_dataframe().groupby(['time']).mean()
        df_feature_reshape = reshape_data(df_feature).loc[:,months_to_be_used]
        
        # Detrending or not
        if detrend == True:
            df_feature_det = pd.DataFrame( 
                signal.detrend(df_feature_reshape, axis=0), index=df_feature_reshape.index, 
                columns = df_feature_reshape.columns ) + df_feature_reshape.mean(axis=0)
            df_features_avg_list.append(df_feature_det) 
        elif detrend == False: 
            df_features_avg_list.append(df_feature_reshape)    
    
    df_clim_avg_features = pd.concat(df_features_avg_list, axis=1)
    df_clim_avg_features.columns = column_names
    
    for feature in list(DS_cli_us.keys()):   
        test = DS_cli_us[feature].sel(time=DS_cli_us.time.dt.month.isin([months_to_be_used])).to_dataframe().groupby(['time']).mean()
        test_reshape = reshape_data(test).loc[:,months_to_be_used]
        
        for i in test_reshape.columns:                     
            # plot trend and detrend
            plt.figure(figsize=(5, 5), dpi=144)
            plt.plot(test_reshape[i], label=f'{test.keys()[0]}_{i}', color = 'red',linestyle='--' )
            plt.plot(df_clim_avg_features[f'{test.keys()[0]}{i}'], label =f' detrended {test.keys()[0]}_{i}',  color = 'darkblue', alpha = 1)
            plt.legend(loc="upper left")
            plt.show()
            
    
    return df_clim_avg_features

months_year_test = range(1,13)

df_clim_avg_features_us_test = conversion_clim_yield_test(DS_cli_us, df_co2, months_to_be_used=months_year_test, detrend = True)
df_clim_avg_features_us_test['year'] = df_clim_avg_features_us_test.index

df_clim_avg_features_us_test_tmx = pd.concat([df_clim_avg_features_us_test.iloc[:,0:12], df_clim_avg_features_us_test.iloc[:,-1]], axis = 1)
df_clim_avg_features_us_test_precip = pd.concat([ df_clim_avg_features_us_test.iloc[:,12:24], df_clim_avg_features_us_test.iloc[:,-1]], axis = 1)
df_clim_avg_features_us_test_dtr = pd.concat([df_clim_avg_features_us_test.iloc[:,24:-1],df_clim_avg_features_us_test.iloc[:,-1]], axis = 1)


df_clim_avg_features_us_test_long_tmx = pd.wide_to_long(df_clim_avg_features_us_test_tmx, stubnames='tmx', i ='year', j='month')
df_clim_avg_features_us_test_long_dtr = pd.wide_to_long(df_clim_avg_features_us_test_dtr, stubnames='dtr', i ='year', j='month')
df_clim_avg_features_us_test_long_precip = pd.wide_to_long(df_clim_avg_features_us_test_precip, stubnames='precip', i ='year', j='month')

df_clim_ref = DS_cli_us.to_dataframe().groupby(['time']).mean()

df_clim_avg_features_us_test_long_tmx = df_clim_avg_features_us_test_long_tmx.sort_index(level=['year','month'])
df_clim_avg_features_us_test_long_dtr = df_clim_avg_features_us_test_long_dtr.sort_index(level=['year','month'])
df_clim_avg_features_us_test_long_precip = df_clim_avg_features_us_test_long_precip.sort_index(level=['year','month'])


df_clim_detrended = df_clim_ref.copy()
df_clim_detrended['tmx'] = df_clim_avg_features_us_test_long_tmx.values
df_clim_detrended['dtr']= df_clim_avg_features_us_test_long_dtr.values
df_clim_detrended['precip']= df_clim_avg_features_us_test_long_precip.values

# THEY SHOULD BE IDENTICAL
print(df_clim_agg_chosen.dtr_7_8[2012])
print(df_clim_avg_features_us_test.loc[2012,['dtr7','dtr8']].mean())
print(df_clim_avg_features_us_test_long_dtr.loc[2012].loc[[7,8]].mean().values)

plt.plot(df_3C_test.index.unique(), DS_cli_us['tmx'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
plt.plot(df_3C_test.index.unique(), df_clim_detrended['tmx'].loc[(df_clim_detrended.index > '2012-01-01') & (df_clim_detrended.index < '2012-12-31')], label = '2012 season', color = 'red')

#%% Right graphs
plt.figure(figsize=(5.5,5), dpi=500)
plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
plt.plot(df_3C_test.index.unique(), df_clim_detrended['tmx'].loc[(df_clim_detrended.index > '2012-01-01') & (df_clim_detrended.index < '2012-12-31')], label = '2012 season', color = 'black')
# plt.plot(df_3C_test.index.unique(), DS_cli_us['tmx'].sel(time=(DS_cli_us.time.dt.year != 2012)).groupby('time.month').mean('time').mean(['lat','lon']), label = 'Historical', linestyle='dotted', color = 'black')
sns.lineplot(data = df_PD_nonfail, y = 'tmx', x = df_PD_nonfail.index, linestyle='dotted',label = 'PD climatology',color = 'black')
sns.lineplot(data = df_2C_nonfail, y = 'tmx', x = df_2C_nonfail.index, linestyle='dotted',label = '2C climatology',color = '#fdbb84')
sns.lineplot(data = df_3C_nonfail, y = 'tmx', x = df_3C_nonfail.index, linestyle='dotted',label = '3C climatology',color = '#e34a33')
# sns.lineplot(data = df_hist_test, y = 'tmx', x = df_hist_test.index, label = 'historical analogues',  linestyle='dashed',color = 'black')
# sns.lineplot(data = df_PD_test, y = 'tmx', x = df_PD_test.index, label = 'PD analogues', color = 'black')
sns.lineplot(data = df_2C_test, y = 'tmx', x = df_2C_test.index, label = '2C analogues', color = '#fdbb84')
sns.lineplot(data = df_3C_test, y = 'tmx', x = df_3C_test.index, label = '3C analogues', color = '#e34a33')
plt.ylabel('Temperature (°C)')
plt.legend(frameon=False)
plt.xticks(df_3C_test.index.unique(), ticks)
plt.title("d) Maximum monthly temperature", loc='left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5.5,5), dpi=500)
plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
plt.plot(df_3C_test.index.unique(), df_clim_detrended['precip'].loc[(df_clim_detrended.index > '2012-01-01') & (df_clim_detrended.index < '2012-12-31')], label = '2012 season', color = 'black')
# plt.plot(df_3C_test.index.unique(), DS_cli_us['precip'].sel(time=(DS_cli_us.time.dt.year != 2012)).groupby('time.month').mean('time').mean(['lat','lon']), label = 'PD climatology', linestyle='dotted', color = 'black')
sns.lineplot(data = df_PD_nonfail, y = 'pre', x = df_PD_nonfail.index, linestyle='dotted',label = 'PD climatology',color = 'black', legend = False)
sns.lineplot(data = df_2C_nonfail, y = 'pre', x = df_2C_nonfail.index, linestyle='dotted',label = '2C climatology',color = '#fdbb84', legend = False)
sns.lineplot(data = df_3C_nonfail, y = 'pre', x = df_3C_nonfail.index, linestyle='dotted',label = '3C climatology',color = '#e34a33', legend = False)
# sns.lineplot(data = df_hist_test, y = 'precip', x = df_hist_test.index, label = 'historical analogues',  linestyle='dashed',color = 'black')
# sns.lineplot(data = df_PD_test, y = 'pre', x = df_PD_test.index, label = 'PD analogues', color = 'black')
sns.lineplot(data = df_2C_test, y = 'pre', x = df_2C_test.index, label = '2C analogues', color = '#fdbb84', legend = False)
sns.lineplot(data = df_3C_test, y = 'pre', x = df_3C_test.index, label = '3C analogues', color = '#e34a33', legend = False)
plt.ylabel('Precipitation (mm/month)')
# plt.legend(loc="upper left",  frameon=False)
plt.xticks(df_3C_test.index.unique(), ticks)
plt.title("e) Precipitation", loc='left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5.5,5), dpi=500)
plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
plt.plot(df_3C_test.index.unique(), df_clim_detrended['dtr'].loc[(df_clim_detrended.index > '2012-01-01') & (df_clim_detrended.index < '2012-12-31')], label = '2012 season', color = 'black')
# plt.plot(df_3C_test.index.unique(), DS_cli_us['dtr'].sel(time=(DS_cli_us.time.dt.year != 2012)).groupby('time.month').mean('time').mean(['lat','lon']), label = 'Historical', linestyle='dotted', color = 'black')
sns.lineplot(data = df_PD_nonfail, y = 'dtr', x = df_PD_nonfail.index, linestyle='dotted',label = 'PD climatology',color = 'black', legend = False)
sns.lineplot(data = df_2C_nonfail, y = 'dtr', x = df_2C_nonfail.index, linestyle='dotted',label = '2C climatology',color = '#fdbb84', legend = False)
sns.lineplot(data = df_3C_nonfail, y = 'dtr', x = df_3C_nonfail.index, linestyle='dotted',label = '3C climatology',color = '#e34a33', legend = False)
# sns.lineplot(data = df_PD_test, y = 'dtr', x = df_PD_test.index, label = 'PD analogues', color = 'black')
sns.lineplot(data = df_2C_test, y = 'dtr', x = df_2C_test.index, label = '2C analogues', color = '#fdbb84', legend = False)
sns.lineplot(data = df_3C_test, y = 'dtr', x = df_3C_test.index, label = '3C analogues', color = '#e34a33', legend = False)
plt.ylabel('Temperature (°C)')
# plt.legend(loc="upper left",  frameon=False)
plt.xticks(df_3C_test.index.unique(), ticks)
plt.title("f) Diurnal temperature range", loc='left')
plt.tight_layout()
plt.show()


