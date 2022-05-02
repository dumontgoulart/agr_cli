# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:21:42 2021

@author: morenodu
"""

import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
from sklearnex import patch_sklearn
patch_sklearn()
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

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

def plot_2d_map(dataarray_2d):
    # Plot 2D map of DataArray, remember to average along time or select one temporal interval
    plt.figure(figsize=(12,5)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    dataarray_2d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-80.73,-34,-45,6], ccrs.PlateCarree())
    plt.show()
    
    
# Function
def states_mask(input_gdp_shp, state_names) :
    country = gpd.read_file(input_gdp_shp, crs="epsg:4326") 
    country_shapes = list(shpreader.Reader(input_gdp_shp).geometries())
    soy_states = country[country['NAME_1'].isin(state_names)]
    states_area = soy_states['geometry'].to_crs({'proj':'cea'}) 
    states_area_sum = (sum(states_area.area / 10**6))
    return soy_states, country_shapes, states_area_sum

# Group and detrend - .groupby('time').mean(...)
def detrending(df):
    df_det = pd.DataFrame( 
    signal.detrend(df, axis=0), index=df.index,
    columns = df.columns ) + df.mean(axis=0)
    return df_det



# Detrend Dataset
def detrend_dataset(DS, deg = 'free', dim = 'time', print_res = True, mean_data = None):
            
    if deg == 'free':
        da_list = []
        for feature in list(DS.keys()):
            da = DS[feature]
            print(feature)
            
            if mean_data is None:
                mean_dataarray = da.mean('time')
            else:
                mean_dataarray = mean_data[feature].mean('time') #da.mean('time') - ( da.mean() - mean_data[feature].mean() )
            
            da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )
    
            dict_res = {}
            for degree in [1,2]:
                # detrend along a single dimension
                p = da.polyfit(dim=dim, deg=degree)
                fit = xr.polyval(da[dim], p.polyfit_coefficients)
                
                da_det = da - fit
                
                res_detrend = np.nansum((da_zero_mean.mean(['lat','lon'])-da_det.mean(['lat','lon']))**2)
                dict_res.update({degree:res_detrend})
            if print_res == True:
                print(dict_res)
            deg = min(dict_res, key=dict_res.get) # minimum degree   
            
            # detrend along a single dimension
            print('Chosen degree is ', deg)
            p = da.polyfit(dim=dim, deg=deg)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
        
            da_det = da - fit + mean_dataarray
            da_det.name = feature
            da_list.append(da_det)
        DS_det = xr.merge(da_list) 
    
    else:       
        px= DS.polyfit(dim='time', deg=deg)
        fitx = xr.polyval(DS['time'], px)
        dict_name = dict(zip(list(fitx.keys()), list(DS.keys())))
        fitx = fitx.rename(dict_name)
        DS_det  = (DS - fitx) + mean_data
        
    return DS_det


# Different ways to detrend, select the best one
def detrend_dim(da, dim, deg = 'free', print_res = True):        
    if deg == 'free':
        
        da_zero_mean = da.where( da < np.nanmin(da.values), other = 0 )

        dict_res = {}
        for degree in [1,2]:
            # detrend along a single dimension
            p = da.polyfit(dim=dim, deg=degree)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
            
            da_det = da - fit
            res_detrend = np.nansum((da_zero_mean.mean(['lat','lon'])-da_det.mean(['lat','lon']))**2)
            dict_res_in = {degree:res_detrend}
            dict_res.update(dict_res_in)
        if print_res == True:
            print(dict_res)
        deg = min(dict_res, key=dict_res.get) # minimum degree        
    
    # detrend along a single dimension
    print('Chosen degree is ', deg)
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    
    da_det = da - fit   
    return da_det

def timedelta_to_int(DS, var):
    da_timedelta = DS[var].dt.days
    da_timedelta = da_timedelta.rename(var)
    da_timedelta.attrs["units"] = 'days'
    
    return da_timedelta


#%% First step, load data

# Select states to be considered
# 'Mato Grosso do Sul', 'São Paulo', 'Santa Catarina','Goiás', 'Mato Grosso','Minas Gerais','Tocantins', 'Bahia' , 'Piauí', 'Maranhão'
state_names = ['Rio Grande do Sul', 'Paraná', 'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais','São Paulo','Goiás',  'Piauí','Tocantins', 'Bahia' ]
soy_brs_states, br1_shapes, brs_states_area_sum = states_mask('../../Paper_drought/data/gadm36_BRA_1.shp', state_names)

# Upscaled observed yield--------------------------------------------------------
DS_y_obs_pr = xr.open_dataset("../../paper_hybrid_agri/data/soy_yield_1980_2016_1prc05x05.nc", decode_times=False) / 1000 #soy_yield_1980_2016_1prc05x05 / soy_yield_1980_2016_all_filters05x05
DS_y_obs_pr['time'] = pd.date_range(start='1980', periods=DS_y_obs_pr.sizes['time'], freq='YS').year
DS_y_obs_pr=DS_y_obs_pr.sel(time = slice('1980', '2016'))
plot_2d_map(DS_y_obs_pr['Yield'].mean('time'))
# DS_y_obs_pr = mask_shape_border(DS_y_obs_pr, soy_brs_states)

# #Shift one year for only 2001 - 2007
# DS_y_obs_pr_test = DS_y_obs_pr['Yield'].copy().shift(time = 1) 
# DS_y_obs_pr['Yield'].loc["2001":"2007"] = DS_y_obs_pr_test.loc["2001":"2007"].copy() #DS_y_obs_up_test['Yield'].values

# plt.plot( DS_y_obs_pr.time, DS_y_obs_pr['Yield'].mean(['lat','lon']))
# plt.plot( DS_y_obs_pr_test.time, DS_y_obs_pr_test.mean(['lat','lon']))

# state_names = ['Rio Grande do Sul', 'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais','São Paulo','Goiás',  'Piauí','Tocantins', 'Bahia' ]
# soy_brs_states, br1_shapes, brs_states_area_sum = states_mask('../../Paper_drought/data/gadm36_BRA_1.shp', state_names)

# # Upscaled observed yield--------------------------------------------------------
# DS_y_obs_up = xr.open_dataset("../../paper_hybrid_agri/data/soy_yield_1980_2016_1prc05x05.nc", decode_times=False) / 1000 #soy_yield_1980_2016_1prc05x05 / soy_yield_1980_2016_all_filters05x05
# # DS_y_obs_up = DS_y_obs_up.rename({'__xarray_dataarray_variable__':'Yield'})
# DS_y_obs_up['time'] = pd.date_range(start='1980', periods=DS_y_obs_up.sizes['time'], freq='YS').year
# DS_y_obs_up=DS_y_obs_up.sel(time = slice('1980', '2016'))
# plot_2d_map(DS_y_obs_up['Yield'].mean('time'))
# DS_y_obs_up = mask_shape_border(DS_y_obs_up ,soy_brs_states)

# DS_y_obs_up = DS_y_obs_up.combine_first(DS_y_obs_pr)

DS_y_obs_up = DS_y_obs_pr
DS_y_obs_up_test = DS_y_obs_up['Yield'].copy().shift(time = -1) 
DS_y_obs_up['Yield'] = DS_y_obs_up_test.copy() #DS_y_obs_up_test['Yield'].values


#### Optional if we want to isolate the rainfed 90% soybeans -> Problem is the calendar represents the year 2000, so highly outdated.
DS_mirca_test = xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).sel(longitude=slice(-58.25,-44))
DS_mirca_test = DS_mirca_test.rename({'latitude': 'lat', 'longitude': 'lon'})

plot_2d_map(DS_mirca_test['annual_area_harvested_rfc_crop08_ha_30mn'])
DS_y_obs_up = DS_y_obs_up.where(DS_mirca_test['annual_area_harvested_rfc_crop08_ha_30mn'] > 0 )
plot_2d_map(DS_y_obs_up['Yield'].mean('time'))

# EPIC
DS_y_epic = xr.open_dataset("../../Paper_drought/data/epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc", decode_times=False)
DS_biom_epic = xr.open_dataset("../../Paper_drought/data/epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_biom-soy-noirr_global_annual_1901_2016.nc", decode_times=False)
DS_output_epic = xr.merge([DS_y_epic['yield-soy-noirr'], DS_biom_epic['biom-soy-noirr']])
# Convert time unit
units, reference_date = DS_y_epic.time.attrs['units'].split('since')
DS_output_epic['time'] = pd.date_range(start=reference_date, periods=DS_y_epic.sizes['time'], freq='YS')
DS_output_epic['time'] = DS_output_epic['time'].dt.year #+ 1

# plot to see states level parametrization and forms
plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y_epic["yield-soy-noirr"].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.set_extent([-80.73,-34,-35,6], ccrs.PlateCarree())
plt.show()

plot_2d_map(DS_y_epic["yield-soy-noirr"].mean('time'))

plot_2d_map(DS_output_epic["yield-soy-noirr"].sel(time=2014))


DS_y_epic_br = DS_output_epic.sel(time=slice(1980, 2016))
plot_2d_map(DS_y_epic_br['yield-soy-noirr'].mean('time'))

DS_y_epic_br_clip = DS_y_epic_br.where(DS_y_obs_up['Yield'] >= 0.0 )
DS_y_obs_up_clip = DS_y_obs_up.where(DS_y_epic_br_clip['yield-soy-noirr'] >= 0.0 )
plot_2d_map(DS_y_epic_br_clip['yield-soy-noirr'].mean('time'))

corr_3d = xr.corr(DS_y_epic_br["yield-soy-noirr"], DS_y_obs_up["Yield"], dim="time", )
plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
corr_3d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, levels = 10)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-80.73,-34,-35,6], ccrs.PlateCarree())
plt.show()
corr_3d_high = corr_3d.where(corr_3d > 0.4)
plot_2d_map(corr_3d_high)

# Compare
df_epic = DS_y_epic_br_clip.to_dataframe().dropna()
df_obs = DS_y_obs_up_clip.to_dataframe().dropna()
plot_2d_map(DS_y_obs_up_clip['Yield'].mean('time'))
DS_y_obs_up_clip['Yield'].mean(['lat','lon']).plot()

DS_y_obs_up_clip_det = xr.DataArray( detrend_dim(DS_y_obs_up_clip.Yield, 'time') + DS_y_obs_up_clip.Yield.mean('time'), name= DS_y_obs_up_clip.Yield.name, attrs = DS_y_obs_up_clip.Yield.attrs)
plot_2d_map(DS_y_obs_up_clip_det.mean('time'))
    
DS_y_epic_br_clip_det = xr.DataArray( detrend_dim(DS_y_epic_br_clip["yield-soy-noirr"], 'time') + DS_y_epic_br_clip["yield-soy-noirr"].mean('time'), name= DS_y_epic_br_clip["yield-soy-noirr"].name, attrs = DS_y_epic_br_clip["yield-soy-noirr"].attrs)
# DS_biom_epic_br_clip_det = xr.DataArray( detrend_dim(DS_y_epic_br_clip["biom-soy-noirr"], 'time') + DS_y_epic_br_clip["biom-soy-noirr"].mean('time'), name= DS_y_epic_br_clip["biom-soy-noirr"].name, attrs = DS_y_epic_br_clip["biom-soy-noirr"].attrs)
DS_epic_br_clip_det = xr.merge([DS_y_epic_br_clip_det])

plt.plot(DS_y_obs_up_clip.Yield.mean(['lat','lon']))
plt.plot( DS_y_obs_up_clip_det.mean(['lat','lon']))
plt.title('Observed data detrending')
plt.show()

plt.plot(DS_y_epic_br_clip["yield-soy-noirr"].mean(['lat','lon']))
plt.plot( DS_y_epic_br_clip_det.mean(['lat','lon']))
plt.title('EPIC data detrending')
plt.show()

plt.plot( DS_y_epic_br_clip_det.mean(['lat','lon']))
plt.plot( DS_y_obs_up_clip_det.mean(['lat','lon']))
plt.title('Datasets comparison')
plt.show()

corr_1 = DS_y_epic_br_clip_det.mean(['lat','lon']).to_dataframe()
corr_2 = DS_y_obs_up_clip_det.mean(['lat','lon']).to_dataframe()
print('Correlation epic and observed is', corr_1['yield-soy-noirr'].corr(corr_2['Yield']) )


corr_3d = xr.corr(DS_y_epic_br_clip_det, DS_y_obs_up_clip_det, dim="time")
corr_3d_high = corr_3d.where(corr_3d > 0.4)
plot_2d_map(corr_3d)
plot_2d_map(corr_3d_high)

DS_y_dif_2012 =   DS_y_obs_up_clip_det.sel(time=2005) - DS_y_obs_up_clip_det.mean('time')
DS_y_dif_2012.attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}

plt.figure(figsize=(11,6), dpi=300) #plot clusters
ax=plt.axes(projection=ccrs.PlateCarree())
DS_y_dif_2012.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_title('2005 yield deviation')
ax.set_extent([-80.73,-34,-35,6], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.6)
plt.tight_layout()
# plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
plt.show()


globiom_br_shape = list(shpreader.Reader('../../paper_hybrid_agri/data/soy_br_harvest_area.shp').geometries())

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
ax.add_geometries(globiom_br_shape, ccrs.PlateCarree())
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-80.73,-34,-35,6], ccrs.Geodetic())
plt.show()

# Dataframe of gridded detrended values

df_obs_det = DS_y_obs_up_clip_det.to_dataframe().dropna()
df_epic_det = DS_y_epic_br_clip_det.to_dataframe().dropna()
df_epic_det_2 = DS_epic_br_clip_det.to_dataframe().dropna()
df_epic_det_2 = df_epic_det_2.reorder_levels(['time','lat','lon']).sort_index(ascending = [True,False,True])

#TEST ADJUST EPIC: Resuls it does not change anything for the RF training
DS_epic_meanbiascor = DS_y_epic_br_clip_det.mean(['lat','lon']) - ( DS_y_epic_br_clip_det.mean() - DS_y_obs_up_clip_det.mean())
df_epic_det_3 = DS_epic_meanbiascor.to_dataframe(name='yield-soy-noirr').dropna()


df_epic_grouped = df_epic.groupby('time').mean(...)


# Import CO2 levels globally
DS_co2 = xr.open_dataset("../../Paper_drought/data/ico2_annual_1901 2016.nc",decode_times=False)
DS_co2['time'] = pd.date_range(start='1901', periods=DS_co2.sizes['time'], freq='YS').year

DS_co2 = DS_co2.sel(time=slice(df_epic_grouped.index.get_level_values('time')[0], df_epic_grouped.index.get_level_values('time')[-1]))
df_co2 = DS_co2.to_dataframe()

# removal with a 2nd order based on the CO2 levels
coeff = np.polyfit(df_co2.values.ravel(), df_epic_grouped, 1)
trend = np.polyval(coeff, df_co2.values)
df_epic_grouped_det =  pd.DataFrame( df_epic_grouped - trend, index = df_epic_grouped.index, columns = df_epic_grouped.columns) + df_epic_grouped.mean() 

plt.plot(df_epic_grouped)
plt.plot(df_epic_grouped_det)
plt.show()    

df_obs_mean_det = detrending(df_obs.groupby('time').mean(...))

plt.plot(df_obs_mean_det, label = 'Observed')
plt.plot(df_epic_grouped_det["yield-soy-noirr"]-1, label = 'EPIC')
plt.vlines(df_epic_grouped_det.index, 1,3.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()

# Pearson's correlation
from scipy.stats import pearsonr

corr_grouped, _ = pearsonr(df_epic_grouped_det["yield-soy-noirr"].values.flatten(), df_obs_mean_det.values.flatten())
print('Pearsons correlation: %.3f' % corr_grouped)

corr_batch, _ = pearsonr(df_epic_det["yield-soy-noirr"].values.flatten(), df_obs_det.values.flatten())
print('Pearsons correlation: %.3f' % corr_batch)


df_fao = pd.read_csv('FAOSTAT_data_6-18-2021.csv')
df_fao.index = df_fao.Year
df_fao = pd.DataFrame( df_fao.Value/10000 )
df_fao_subset = df_fao.loc[DS_y_obs_up_clip.time]
df_fao_det = detrending(df_fao_subset)

df_fao_cardinal = df_fao_subset
df_fao_cardinal.index = range(len(df_fao_cardinal.index))


DS_y_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc", decode_times=True)
DS_y_iizumi = DS_y_iizumi.rename({'latitude': 'lat', 'longitude': 'lon'})
plot_2d_map(DS_y_iizumi['yield'].mean('time'))

DS_y_iizumi_test = DS_y_iizumi.where(DS_y_obs_up_clip['Yield'] >= 0.0 )
df_iizumi = DS_y_iizumi_test.to_dataframe().dropna()
df_iizumi_mean = df_iizumi.groupby('time').mean(...)
df_iizumi_mean_det = detrending(df_iizumi_mean)

# Plot time series
plt.figure(figsize=(10,6))
plt.plot(df_epic_grouped_det["yield-soy-noirr"], label = 'EPIC')
plt.plot(df_obs_mean_det, label = 'Obs (subset)')
plt.plot(df_fao_det, label = 'FAO')
plt.plot(df_iizumi_mean_det, label = 'Iizumi')
plt.vlines(df_epic_grouped_det.index, 1,3.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()


#%% Machine learning model training
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,  explained_variance_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from keras.layers import Activation
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
import lightgbm as lgb

import os
os.environ['PYTHONHASHSEED']= '123'
os.environ['TF_CUDNN_DETERMINISTIC']= '1'
import random as python_random
np.random.seed(1)
python_random.seed(1)
tf.random.set_seed(1)

def calibration(X,y,type_of_model='RF', params = None, stack_model = False):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
       
    if params is None:
        if type_of_model == 'RF':
            model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
                                              max_depth = 20, max_features = 'auto',
                                              min_samples_leaf = 1, min_samples_split=2)
            
            full_model_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,
                                              max_depth = 20, max_features = 'auto',
                                              min_samples_leaf = 1, min_samples_split=2)
            
        elif type_of_model == 'lightgbm':
            model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
                                                random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
            ])
            
            
            full_model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', lgb.LGBMRegressor(linear_tree= True, max_depth = 20, num_leaves = 50, min_data_in_leaf = 100, 
                                                random_state=0, learning_rate = 0.01, n_estimators = 1000 ) )
            ])
            
        
        elif type_of_model == 'DNN':
            def create_model():
                model = Sequential()
                model.add(Dense(200, input_dim=len(X_train.columns))) 
                # model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.1))
    
                model.add(Dense(200))
                # model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.1))
    
                model.add(Dense(200))
                # model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.1))
                
                model.add(Dense(200))
                # model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(0.1))
    
                model.add(Dense(1, activation='linear'))
                # compile the keras model
                model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','mean_absolute_error'])
                return model
                    
            model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', KerasRegressor(model=create_model, epochs=200, batch_size= 1024, verbose=1))
            ])
    
            full_model_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', KerasRegressor(model=create_model, epochs=200, batch_size= 1024, verbose=1))
            ])
            

        elif type_of_model == 'MLP':
            model_rf = make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (200,200,200),batch_size= 256,learning_rate_init = 0.01, alpha = 0.0001, verbose=1, max_iter=400,learning_rate = 'adaptive') ) #
            full_model_rf = make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (200,200,200), batch_size= 256,learning_rate_init = 0.01, alpha = 0.0001, verbose=1, max_iter=400,learning_rate = 'adaptive') ) #
            
            
    
    elif params is not None:
        model_rf = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split'])
        
        full_model_rf = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split'])
        
    if stack_model is False:
        model = model_rf.fit(X_train, y_train)
        
        full_model = full_model_rf.fit(X, y)
        
    elif stack_model is True:
        print('Model: stacked')
        
        estimators = [
            ('RF', make_pipeline(StandardScaler(), model_rf)), #make_pipeline(StandardScaler(),
            ('MLP', make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (100,100,100), alpha = 0.0001,verbose=1, max_iter=400,learning_rate = 'adaptive')) )
            ]
        
        estimators_full = [
            ('RF', make_pipeline(StandardScaler(), full_model_rf)), #make_pipeline(StandardScaler(),
            ('MLP', make_pipeline(StandardScaler(), MLPRegressor(random_state=0, hidden_layer_sizes= (100,100,100), alpha = 0.0001, max_iter=400))) #
            ]
        
        # Get together the models:
        stacking_regressor = StackingRegressor(estimators=estimators, final_estimator = LinearRegression() ) # MLPRegressor(random_state=0, max_iter=500) #SVR() # GaussianProcessRegressor(kernel = 1**2 * RationalQuadratic(alpha=1, length_scale=1)) #StackingRegressor(estimators=estimators)
        stacking_regressor_full = StackingRegressor(estimators=estimators_full, final_estimator = LinearRegression() ) # MLPRegressor(random_state=0, max_iter=500) #SVR() # GaussianProcessRegressor(kernel = 1**2 * RationalQuadratic(alpha=1, length_scale=1)) #StackingRegressor(estimators=estimators)

        model = stacking_regressor.fit(X_train, y_train)

        full_model = stacking_regressor_full.fit(X, y)
        
       
        
    def MBE(y_true, y_pred):
        '''
        Parameters:
            y_true (array): Array of observed values
            y_pred (array): Array of prediction values
    
        Returns:
            mbe (float): Bias score
        '''
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true = y_true.reshape(len(y_true),1)
        y_pred = y_pred.reshape(len(y_pred),1)   
        diff = (y_true-y_pred)
        mbe = diff.mean()
        return mbe
    
    # Test performance
    y_pred = model.predict(X_test)
    
    # report performance
    print("R2 on test set:", round(r2_score(y_test, y_pred),2))
    print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
    print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),3))
    print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),3))
    print("MBE on test set:", round(MBE(y_test, y_pred),3))
    print("______")
    
    y_pred_total = full_model.predict(X)
    
    
    plt.figure(figsize=(5,5), dpi=250) #plot clusters
    plt.scatter(y_test, y_pred)
    plt.plot(y_test, y_test, color = 'black', label = '1:1 line')
    plt.ylabel('Predicted yield')
    plt.xlabel('Observed yield')
    plt.title('Scatter plot - test set')
    plt.legend()
    # plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
    plt.show()

    # # perform permutation importance
    # results = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=5, random_state=0, n_jobs=-1)
    # # get importance
    # df_importance = pd.DataFrame(results.importances_mean)
    # df_importance.index = X.columns
    # print("Mutual importance:",df_importance)
    # # summarize feature importance
    # plt.figure(figsize=(12,5)) #plot clusters
    # plt.bar(df_importance.index, df_importance[0])
    # plt.show()
    
    return y_pred, y_pred_total, model, full_model 


from sklearn.model_selection import GridSearchCV

def hyper_param_tuning(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'max_depth': [5,6,10,15,20], #list(range(5,15))
        'max_features': ['auto'],
        'min_samples_leaf': [1,2,3,4],
        'min_samples_split': [2,3,4,5],
        'n_estimators': [100, 200, 300,500]
    }
    # Create a based model
    rf = RandomForestRegressor()# Instantiate the grid search model #scoring='neg_mean_absolute_error',
    grid_search = GridSearchCV(estimator = rf,  param_grid = param_grid, scoring = 'neg_root_mean_squared_error', cv = 5, n_jobs = -1, verbose = 2) 
    grid_search.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    means = grid_search.cv_results_["mean_test_score"]
    stds = grid_search.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
    params_cv_chosen = grid_search.best_params_
    best_grid = grid_search.best_estimator_
    
    return params_cv_chosen, best_grid

#%% EPIC RF

# X, y = df_epic_grouped_det.values.reshape(-1, 1), df_obs_mean_det.values.ravel()
# y_pred_epic, y_pred_total_epic = calibration(X,y)

# df_pred_epic = pd.DataFrame(y_pred_epic, index = df_obs_test.index)
# df_pred_epic_total = pd.DataFrame(y_pred_total_epic, index = df_epic_grouped_det.index)


# # Batch process
# feature_importance_selection(df_epic_det, df_obs_det)

X, y = df_epic_det, df_obs_det.values.flatten().ravel()

# # Tune hyper-parameters --------------------------------------------------
# params_cv_chosen_epic_br, best_grid_epic = hyper_param_tuning(X,y)

# # Save hyper-parameters
# with open('params_cv_chosen_epic_br.pickle', 'wb') as f:
#     pickle.dump(params_cv_chosen_epic_br, f)
# -------------------------------------------------------------------------    
# for test_size in [0.1,0.2,0.3,0.4,0.5]:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
#     regr_rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
#                                       max_depth = 20, max_features = 'auto',
#                                            min_samples_leaf = 1, min_samples_split=2)
#     regr_rf.fit(X_train.values, y_train)
    
#     y_rf = regr_rf.predict(X_test)
    
#     print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_rf),2))

# Standard model
y_pred_epic, y_pred_total_epic, model_epic_br, full_model_epic = calibration(X, y, type_of_model= 'lightgbm')

# # Tunned model
# with open('../../Paper_drought/data/params_cv_chosen_epic_br.pickle', 'rb') as f:
#     params_cv_chosen_epic_br = pickle.load(f)
    
# y_pred_epic, y_pred_total_epic, model_epic, full_model_epic = calibration(X, y, params = params_cv_chosen_epic_br )


DS_y_test = DS_y_obs_up_clip_det.to_dataset()
DS_y_test = DS_y_test.rename({'Yield':'yield'})


#%% HADEX V3 - NEW TEST

start_date, end_date = '01-01-1979','30-12-2016'

# DS_hadex = xr.open_mfdataset('HadEX3_ref1961-1990_mon/*.nc').sel(time=slice(start_date, end_date))
DS_hadex = xr.open_mfdataset('../../paper_hybrid_agri/data/climpact-master/climpact-master/www/output_gswp3/monthly_data/*.nc').sel(time=slice(start_date, end_date))
# DS_hadex = DS_hadex.drop(['latitude_bnds','longitude_bnds'])

# New dataset
# DS_hadex = xr.open_dataset('DS_hadex_all_hr.nc').sel(time=slice(start_date, end_date)).sel(lon=slice(-79,-30), lat=slice(0,-39))
DS_hadex = DS_hadex.drop_vars('fd') # Always zero
DS_hadex = DS_hadex.drop_vars('id') # Always zero
DS_hadex = DS_hadex.drop_vars('time_bnds') # Always zero
DS_hadex = DS_hadex.drop_vars('spi') # Always zero
DS_hadex = DS_hadex.drop_vars('spei') # Always zero
DS_hadex = DS_hadex.drop('scale') # Always zero

list_features_br = ['prcptot', 'r10mm', 'txm' ]# 'dtr', 'tnm', 'txge35', 'tr', 'txm', 'tmm', 'tnn'
DS_hadex = DS_hadex[list_features_br] 


plot_2d_map(DS_hadex['prcptot'].mean('time')) 

da_list = []
for feature in list(DS_hadex.keys()):
    if (type(DS_hadex[feature].values[0,0,0]) == type(DS_hadex.r10mm.values[0,0,0])):
        print('Time')
        DS = timedelta_to_int(DS_hadex, feature)
    else:
        print('Integer')
        DS = DS_hadex[feature]
    
    da_list.append(DS)

DS_hadex_combined = xr.merge(da_list)    
DS_hadex_combined = DS_hadex_combined.drop_vars('r10mm') # Always zero
if len(DS_hadex_combined.coords) >3 :
    DS_hadex_combined=DS_hadex_combined.drop('spatial_ref')
    
# DS_hadex_combined = DS_hadex_combined.rename({'latitude': 'lat', 'longitude': 'lon'})
DS_hadex_combined.coords['lon'] = (DS_hadex_combined.coords['lon'] + 180) % 360 - 180
DS_hadex_combined = DS_hadex_combined.sortby(DS_hadex_combined.lon)
DS_hadex_combined = DS_hadex_combined.reindex(lat=DS_hadex_combined.lat[::-1])
if len(DS_hadex_combined.coords) >3 :
    DS_hadex_combined=DS_hadex_combined.drop('spatial_ref')
    
DS_hadex_combined_br = mask_shape_border(DS_hadex_combined, soy_brs_states)
DS_hadex_combined_br = DS_hadex_combined.where(DS_y_obs_up_clip_det.mean('time') > -10)
if len(DS_hadex_combined_br.coords) >3 :
    DS_hadex_combined_br=DS_hadex_combined_br.drop('spatial_ref')
plot_2d_map(DS_hadex_combined_br['prcptot'].mean('time'))

# Select data according to months
def is_month(month, ref_in, ref_out):
    return (month >= ref_in) & (month <= ref_out)

DS_hadex_combined_br_det = detrend_dataset(DS_hadex_combined_br)

# Check if it works:
DS_hadex_combined_br.prcptot.groupby('time').mean(...).plot()
DS_hadex_combined_br_det.prcptot.groupby('time').mean(...).plot()

# Select months
DS_hadex_combined_br_season = DS_hadex_combined_br_det.sel(time=is_month(DS_hadex_combined_br_det['time.month'], 1, 2))
DS_hadex_combined_br_season = DS_hadex_combined_br_season.transpose("time", "lat", "lon")

# Average across season
DS_hadex_combined_br_season = DS_hadex_combined_br_season.groupby('time.year').mean('time')
DS_hadex_combined_br_season = DS_hadex_combined_br_season.rename({'year': 'time'})



DS_hadex_combined_br_season = DS_hadex_combined_br_season.where(DS_y_obs_up_clip_det >= -5.0 )
DS_y_obs_up_clip_det_test = DS_y_obs_up_clip_det.where(DS_hadex_combined_br_season['prcptot'] >= -5.0 )
DS_y_obs_up_clip_det_test = DS_y_obs_up_clip_det_test.where(DS_hadex_combined_br_season['prcptot'].mean('time') >= -5.0 )


#%% ########## BATCH
DS_hadex_combined_br_det = DS_hadex_combined_br # detrend_dataset(DS_hadex_combined_br)
# DS_hadex_combined_br_det = DS_hadex_combined_br_det[['ETR','DTR', 'R10mm', 'Rx5day']]
# Select months
DS_hadex_combined_br_season = DS_hadex_combined_br_det.sel(time=is_month(DS_hadex_combined_br_det['time.month'], 1,2))
# Average across season
DS_hadex_combined_br_season = DS_hadex_combined_br_season.groupby('time.year').mean('time')
DS_hadex_combined_br_season = DS_hadex_combined_br_season.rename({'year':'time'})
DS_hadex_combined_br_season = DS_hadex_combined_br_season.reindex(lat=DS_hadex_combined_br_season.lat[::-1])
DS_hadex_combined_br_season = DS_hadex_combined_br_season.where(DS_y_obs_up_clip_det >= -5.0 )
# DS_hadex_combined_br_season.to_netcdf('ds_clim.nc')

# Find variable with minimum length in the dataset and use it as the mask variable
def minimum_variable_dataset(DS_clim):
    feature_len_list = pd.DataFrame(np.empty([len(list(DS_clim.keys())),2]), columns = ['feature','length'])
    i=0
    for feature in list(DS_clim.keys()):
        # print(feature, len(DS_clim[feature].to_dataframe().dropna() ))
        feature_len_list.iloc[i,0] = [feature]
        feature_len_list.iloc[i,1] = [len(DS_clim[feature].to_dataframe().dropna())]
        i=i+1
    
    feature_name = feature_len_list[feature_len_list.iloc[:,1] == feature_len_list.iloc[:,1].min()].feature.values[0]
    print(feature_name)
    return feature_name

feature_name = minimum_variable_dataset(DS_hadex_combined_br_season)

test2 = DS_y_obs_up_clip_det.where(DS_hadex_combined_br_season[feature_name] >= DS_hadex_combined_br_season[feature_name].min() )
DS_epic_br_det_forhybrid = DS_y_epic_br_clip_det.where(DS_hadex_combined_br_season[feature_name] >= DS_hadex_combined_br_season[feature_name].min() )

df_hadex_combined_br_season = DS_hadex_combined_br_season.to_dataframe().dropna()
df_test2 = test2.to_dataframe().dropna()
df_epic_det_forhybrid = DS_epic_br_det_forhybrid.to_dataframe().dropna()
# test2.to_netcdf('ds_yield.nc')

plot_2d_map(DS_hadex_combined_br_season['prcptot'].mean('time'))
plot_2d_map(test2.sel(time=2016))
plot_2d_map(test2.mean('time'))

# corr_3d_had = xr.corr(DS_hadex_combined_br_season['prcptot'], DS_y_obs_up_clip_det, dim="time", )
# plt.figure(figsize=(12,5)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# corr_3d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, levels = 10)
# ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-80.73,-34,-35,6], ccrs.PlateCarree())
# plt.show()


df_hadex_combined_br_season = df_hadex_combined_br_season.reorder_levels(['time','lat','lon']).sort_index()
df_test2 = df_test2.reorder_levels(['time','lat','lon']).sort_index()

feature_importance_selection(df_hadex_combined_br_season, df_test2)

print('Extreme climatic indices results:')
X, y = df_hadex_combined_br_season, df_test2.values.ravel()
y_pred_clim, y_pred_total_clim, model_clim, full_model_clim = calibration(X,y)

#%% Relative dates calendar

# Functions
# Reshape to have each calendar year on the columns (1..12)
def reshape_data(dataarray):  #converts and reshape data
    if isinstance(dataarray, pd.DataFrame): #If already dataframe, skip the convertsion
        dataframe = dataarray
    elif isinstance(dataarray, pd.Series):    
        dataframe = dataarray.to_frame()
        
    dataframe['month'] = dataframe.index.get_level_values('time').month
    dataframe['year'] = dataframe.index.get_level_values('time').year
    dataframe.set_index('month', append=True, inplace=True)
    dataframe.set_index('year', append=True, inplace=True)
    # dataframe = dataframe.reorder_levels(['time', 'year','month'])
    dataframe.index = dataframe.index.droplevel('time')
    dataframe = dataframe.unstack('month')
    dataframe.columns = dataframe.columns.droplevel()
    
    return dataframe


def reshape_shift(dataset, shift_time=0):
    ### Convert to dataframe
    if shift_time == 0:     
        dataframe_1 = dataset.to_dataframe()
        # Define the column names
        column_names = [dataset.name +"_"+str(j) for j in range(1,13)]   
    else:
        dataframe_1 = dataset.shift(time=-shift_time).to_dataframe()    
        # Define the column names
        column_names = [dataset.name +"_"+str(j) for j in range(1+shift_time,13+shift_time)]
    # Reshape
    dataframe_reshape = reshape_data(dataframe_1)
    dataframe_reshape.columns = column_names
    
    return dataframe_reshape

# DS_calendar_plant = xr.open_dataset('../../Paper_drought/data/soy_rf_pd_2015soc.nc').mean('time') / (365/12)
# plot_2d_map(DS_calendar_plant['Calendar'])

# DS_calendar_mature = xr.open_dataset('../../paper_hybrid_agri/data/soy_rf_md_2015soc_2.nc').mean('time') / (365/12)
# plot_2d_map(DS_calendar_mature['Calendar'])

# DS_cal_sachs = xr.open_dataset('../../paper_hybrid_agri/data/Soybeans.crop.calendar_sachs_05x05.nc') / (365/12)
# plot_2d_map( (DS_cal_sachs['plant'] ))

# DS_cal_abr = xr.open_dataset('../../paper_hybrid_agri/data/calendar_soybeans/calendar_v15_05x05_2.nc')
# DS_cal_abr['time'] = pd.date_range(start='1973', periods=DS_cal_abr.sizes['time'], freq='YS').year
# # crop to obs yield values
# DS_cal_abr_mean = DS_cal_abr.mean('time') / (365/12)
# plot_2d_map( (DS_cal_abr_mean['cliendcal'] ))

DS_cal_sachs = xr.open_dataset('../../paper_hybrid_agri/data/Soybeans.crop.calendar_sachs_05x05.nc') / (365/12) # 0.72 for Sachs // best type of calendar is plant
DS_cal_mirca = xr.open_dataset('../../paper_hybrid_agri/data/mirca2000_soy_calendar.nc') # 
# DS_cal_plant = xr.open_dataset('../../Paper_drought/data/soy_rf_pd_2015soc.nc').mean('time') / (365/12)
DS_cal_ggcmi = xr.open_dataset('../../paper_hybrid_agri/data/soy_rf_ggcmi_crop_calendar_phase3_v1.01.nc4') / (365/12)

DS_cal_mirca_subset = DS_cal_mirca.where(DS_y_obs_up_clip_det.mean('time') >= -5.0 )
DS_cal_sachs_month_subset = DS_cal_sachs.where(DS_y_obs_up_clip_det.mean('time') >= -5.0)
# DS_cal_plant = DS_cal_plant.where(DS_exclim_us_det['prcptot'].mean('time') >= -10)
DS_cal_ggcmi_subset = DS_cal_ggcmi.where(DS_y_obs_up_clip_det.mean('time') >= -5.0 )


### Chose calendar:
DS_chosen_calendar_br = DS_cal_mirca_subset['start']   #  DS_cal_mirca_subset['start'] DS_cal_ggcmi['planting_day']
if DS_chosen_calendar_br.name != 'plant':
    DS_chosen_calendar_br = DS_chosen_calendar_br.rename('plant')
# Convert DS to df
df_chosen_calendar = DS_chosen_calendar_br.to_dataframe().dropna()
# Convert planting days to beginning of the month
df_calendar_month_br = df_chosen_calendar[['plant']].apply(np.rint).astype('Int64')


### LOAD climate date and clip to the calendar cells    
DS_exclim_br_det_clip = DS_hadex_combined_br_det.sel(time=slice('1979-12-31','2016-12-16')).where(DS_chosen_calendar_br >= 0 )

# For loop along features to obtain 24 months of climatic data for each year
list_features_reshape_shift = []
for feature in list(DS_exclim_br_det_clip.keys()):
    ### Reshape and shift for 24 months for every year.
    df_test_shift = reshape_shift(DS_exclim_br_det_clip[feature])
    df_test_shift_12 = reshape_shift(DS_exclim_br_det_clip[feature], shift_time = 12)
    # Combine both dataframes
    df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)
    # Remove last year, because we do not have two years for it
    df_test_reshape_twoyears = df_test_reshape_twoyears.query('year <= 2016')
    ### Join and change name to S for the shift values
    df_feature_reshape_shift = (df_test_reshape_twoyears.dropna().join(df_calendar_month_br)
                                .rename(columns={'plant':'s'}))
    # Move 
    col = df_feature_reshape_shift.pop("s")
    df_feature_reshape_shift.insert(0, col.name, col)
    # Activate this if error "TypeError: int() argument must be a string, a bytes-like object or a number, not 'NAType'" occurs
    
    # print(df_feature_reshape_shift[['s']].isna().sum())

    # Shift accoording to month indicator (hence +1)
    df_feature_reshape_shift = (df_feature_reshape_shift.apply(lambda x : x.shift(-(int(x['s']))+1) , axis=1)
                                .drop(columns=['s']))
    
    
    list_features_reshape_shift.append(df_feature_reshape_shift)

# Transform into dataframe
df_features_reshape_2years = pd.concat(list_features_reshape_shift, axis=1)

### Select specific months ###################################################
suffixes = tuple(["_"+str(j) for j in range(3,6)])
df_feature_season_6mon = df_features_reshape_2years.loc[:,df_features_reshape_2years.columns.str.endswith(suffixes)]

df_feature_season_6mon_br = df_feature_season_6mon.copy()

# # Shift 1 year
# df_feature_season_6mon_br.index = df_feature_season_6mon_br.index.set_levels(df_feature_season_6mon_br.index.levels[2] + 1, level=2)


df_feature_season_6mon_br = df_feature_season_6mon_br.rename_axis(index={'year':'time'})
df_feature_season_6mon_br = df_feature_season_6mon_br.reorder_levels(['time','lat','lon']).sort_index()
df_feature_season_6mon_br = df_feature_season_6mon_br.where(df_hadex_combined_br_season['prcptot']>=0).dropna().astype(float)
df_test3 = df_test2.where(df_feature_season_6mon_br['prcptot'+suffixes[0]]>=-1000).dropna()
df_feature_season_6mon_br = df_feature_season_6mon_br.where(df_test3['Yield']>=-5).dropna()

# SECOND DETRENDING PART - SEASONAL
DS_feature_season_6mon_br = xr.Dataset.from_dataframe(df_feature_season_6mon_br)
DS_feature_season_6mon_br_det = detrend_dataset(DS_feature_season_6mon_br, deg = 'free')
df_feature_season_6mon_br_det = DS_feature_season_6mon_br_det.to_dataframe().dropna()

for feature in df_feature_season_6mon_br.columns:
    df_feature_season_6mon_br[feature].groupby('time').mean().plot(label = 'old')
    df_feature_season_6mon_br_det[feature].groupby('time').mean().plot(label = 'detrend')
    plt.title(f'{feature}')
    plt.legend()
    plt.show()

# =============================================================================
# # ATTENTION HERE - update second detrending scheme
# =============================================================================
df_feature_season_6mon_br = df_feature_season_6mon_br_det

for feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
    df_feature_season_6mon_br[feature][df_feature_season_6mon_br[feature] < 0] = 0

feature_importance_selection(df_feature_season_6mon_br, df_test3)

print('Dynamic clim data')
X, y = df_feature_season_6mon_br, df_test3.values.ravel()
y_pred_exclim_dyn_br, y_pred_total_exclim_dyn_br, model_exclim_dyn_br, full_model_exclim_dyn_br = calibration(X, y, type_of_model='RF', stack_model = False)


#### BENCHMARK FOR CALENDAR CHANGES ####################################################################
# For loop along features to obtain 24 months of climatic data for each year
list_static_calendar = []
for feature in list(DS_exclim_br_det_clip.keys()):
    ### Reshape and shift for 24 months for every year.
    df_test_shift = reshape_shift(DS_exclim_br_det_clip[feature])
    df_test_shift_12 = reshape_shift(DS_exclim_br_det_clip[feature], shift_time = 12)
    # Combine both dataframes
    df_test_reshape_twoyears = df_test_shift.dropna().join(df_test_shift_12)
    list_static_calendar.append(df_test_reshape_twoyears)

# Transform into dataframe
df_cal_benchmark = pd.concat(list_static_calendar, axis=1)

### Select specific months
suffixes_stat = tuple(["_"+str(j) for j in range(11,14)])
df_cal_benchmark_season = df_cal_benchmark.loc[:,df_cal_benchmark.columns.str.endswith(suffixes_stat)]
df_cal_benchmark_season = df_cal_benchmark_season.rename_axis(index={'year':'time'}).reorder_levels(['time','lat','lon']).sort_index()
df_cal_benchmark_season = df_cal_benchmark_season.where(df_test2['Yield'] >= -5).dropna().astype(float)
df_test4 = df_test2.where(df_cal_benchmark_season['prcptot_12']>=-100).dropna()

feature_importance_selection(df_cal_benchmark_season, df_test4)

print('Static clim data')
X, y = df_cal_benchmark_season, df_test4.values.flatten().ravel()
y_pred_exclim_stat, y_pred_total_exclim_stat, model_exclim_stat, full_model_exclim_stat = calibration(X, y, type_of_model='RF', stack_model = False)
###################################################################################################




#%% HYBRID
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')

# Define hybrid as:
# df_hybrid = pd.concat([df_epic_grouped_det, df_clim_mon_brs_sub_agg], axis =1 )
# X, y = df_hybrid.values, df_obs_mean_det.values.ravel()

df_epic_det_forhybrid = df_epic_det_forhybrid.reorder_levels(['time','lat','lon']).sort_index(ascending = [True,True,True])
df_epic_det_forhybrid = df_epic_det_forhybrid.where(df_feature_season_6mon_br['prcptot'+suffixes[0]]>=-100).dropna()

df_hybrid_batch = pd.concat([df_epic_det_forhybrid, df_feature_season_6mon_br], axis = 1 )
X, y = df_hybrid_batch, df_test3.values.ravel()

# Save this for future operations:
df_hybrid_batch.to_csv('dataset_input_hybrid_forML_br.csv')
df_test3.to_csv('dataset_obs_yield_forML_br.csv')

# Feature selection
feature_importance_selection(df_hybrid_batch, df_test3)


for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
    regr_rf.fit(X_train, y_train)
    
    y_rf = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_rf),2))

# Evaluate Model
print('Hybrid results for Deep neural network:')
X, y = df_hybrid_batch, df_test3.values.ravel()
y_pred_hyb_rf, y_pred_total_hyb_rf, model_hyb_rf, full_model_hyb_rf = calibration(X, y, type_of_model='RF', stack_model = False) #DNN for neural network
y_pred_hyb_br, y_pred_total_hyb_br, model_hyb_br, full_model_hyb_br = calibration(X, y, type_of_model='DNN', stack_model = False)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Test performance
y_pred = model_hyb_am2.predict(X_test)

# report performance
print("R2 on test set:", round(r2_score(y_test, y_pred),2))
print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))

### TEST LIGHTGBM with linear trees on


# ################ saving just the keras model, still needs to be scaled. <<<<<<<<<<<<<<<<
# full_model_hyb_br['estimator'].model_.save('/hybrid_model/hybrid_br_ANN_61_ext')

# # Save the Keras model first:
# full_model_hyb_br['estimator'].model_.save('keras_model_br_noBN_61_ext.h5')

# # This hack allows us to save the sklearn pipeline:
# full_model_hyb_br['estimator'].model = None

# import joblib
# from keras.models import load_model
# # Finally, save the pipeline:
# joblib.dump(full_model_hyb_br, 'sklearn_pipeline_br_noBN_61_ext.pkl')


#%%
# Compare timelines
df_predict_test_hist = df_test3.copy()
predict_test_hist = model_hyb_br.predict(df_hybrid_batch)
df_predict_test_hist.loc[:,"Yield"] = predict_test_hist
DS_predict_test_hist = xr.Dataset.from_dataframe(df_predict_test_hist)
DS_predict_test_hist.to_netcdf('netcdf_present_BR.nc')
DS_predict_test_hist = DS_predict_test_hist.sortby('lat')
DS_predict_test_hist = DS_predict_test_hist.sortby('lon')

### uncomment to save shifters
# shift_2012 = DS_predict_test_hist['Yield'].sel(time=2012) / DS_predict_test_hist['Yield'].mean(['time']) 
# Reindex to avoid missnig coordinates and dimension values
# new_lat = np.arange(shift_2012.lat[0], shift_2012.lat[-1], 0.5)
# new_lon = np.arange(shift_2012.lon[0], shift_2012.lon[-1], 0.5)
# shift_2012 = shift_2012.reindex({'lat':new_lat})
# shift_2012 = shift_2012.reindex({'lon':new_lon})
# shift_2012.to_netcdf('shifter_2012_BR.nc')

# RF:EPIC
df_predict_epic_hist = df_test3.copy()
predict_epic_hist = model_epic_br.predict(df_epic_det_forhybrid)
df_predict_epic_hist.loc[:,"Yield"] = predict_epic_hist
DS_predict_epic_hist = xr.Dataset.from_dataframe(df_predict_epic_hist)

# RF:ECE
df_predict_clim_hist = df_test3.copy()
predict_clim_hist = model_exclim_dyn_br.predict(df_feature_season_6mon_br)
df_predict_clim_hist.loc[:,"Yield"] = predict_clim_hist
DS_predict_clim_hist = xr.Dataset.from_dataframe(df_predict_clim_hist)

print("Correlation is:", xr.corr(DS_predict_test_hist["Yield"], DS_predict_clim_hist["Yield"]) )


# PLOTS
plt.figure(figsize=(10,6), dpi=300) #plot clusters
plt.plot(DS_y_epic_br_clip_det.time, DS_y_epic_br_clip_det.mean(['lat','lon']), label = 'Original EPIC',linestyle='dashed',linewidth=3)
plt.plot(DS_y_obs_up_clip_det.time, DS_y_obs_up_clip_det.mean(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_predict_epic_hist['Yield'].mean(['lat','lon']), label = 'RF:EPIC')
plt.plot(DS_predict_clim_hist.time, DS_predict_clim_hist['Yield'].mean(['lat','lon']), label = 'RF:ECE')
plt.plot(DS_predict_test_hist.time, DS_predict_test_hist['Yield'].mean(['lat','lon']), label = 'RF:Hybrid')
plt.legend()
plt.show()


### WIEGHTED ANALYSIS
DS_harvest_area_globiom =  xr.open_dataset("../../paper_hybrid_agri/data/americas_mask_ha.nc", decode_times=False).rename({'latitude': 'lat', 'longitude': 'lon'}) # xr.open_dataset('../../paper_hybrid_agri/data/soy_harvest_area_globiom_05x05_2b.nc').mean('time')
DS_harvest_area_globiom = DS_harvest_area_globiom.rename({'annual_area_harvested_rfc_crop08_ha_30mn':'harvest_area'}).where(DS_predict_test_hist['Yield']>=0)
plot_2d_am_map(DS_harvest_area_globiom['harvest_area'].isel(time = 0))
plot_2d_am_map(DS_harvest_area_globiom['harvest_area'].isel(time = -1))

total_area = DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
DS_obs_weighted = ((DS_y_obs_up_clip_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_hybrid_weighted = ((DS_predict_test_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_epic_weighted = ((DS_predict_epic_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_clim_weighted = ((DS_predict_clim_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_epic_orig_weighted =((DS_y_epic_br_clip_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield') 

# Weighted plot
plt.figure(figsize=(8,5), dpi=300) #plot clusters
# plt.plot(DS_predict_epic_hist.time, DS_epic_orig_weighted['Yield'].sum(['lat','lon']), label = 'Original epic', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_obs_weighted['Yield'].sum(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_epic_weighted['Yield'].sum(['lat','lon']), label = 'RF:EPIC')
plt.plot(DS_predict_epic_hist.time, DS_clim_weighted['Yield'].sum(['lat','lon']), label = 'RF:CLIM')
plt.plot(DS_predict_epic_hist.time, DS_hybrid_weighted['Yield'].sum(['lat','lon']), label = 'RF:hybrid')
plt.ylabel('Yield (ton/ha)')
plt.xlabel('Years')
plt.legend()
plt.show()

print("Weighted R2 OBS-EPIC:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_epic_orig_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 OBS-RF:EPIC:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_epic_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 OBS-Clim:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_clim_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 OBS-Hybrid:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_hybrid_weighted['Yield'].sum(['lat','lon'])),2))
print("Weighted R2 Hybrid-clim:",round(r2_score(DS_hybrid_weighted['Yield'].sum(['lat','lon']), DS_clim_weighted['Yield'].sum(['lat','lon'])),2))


# Scatter plots
plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_test3['Yield'],df_epic_det_forhybrid['yield-soy-noirr'])
plt.plot(df_test3['Yield'].sort_values(), df_test3['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('Original EPIC predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
plt.show()

plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_test3['Yield'],df_predict_test_hist['Yield'])
plt.plot(df_test3['Yield'].sort_values(), df_test3['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('Hybrid predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
plt.show()


plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_test3['Yield'],df_predict_epic_hist['Yield'])
plt.plot(df_test3['Yield'].sort_values(), df_test3['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('EPIC predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
plt.show()


plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_test3['Yield'],df_predict_clim_hist['Yield'])
plt.plot(df_test3['Yield'].sort_values(), df_test3['Yield'].sort_values(), linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('RF clim predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
plt.show()

print("R2 OBS-EPIC_original:",round(r2_score(df_test3['Yield'], df_epic_det_forhybrid['yield-soy-noirr']),2))
print("R2 OBS-RF:EPIC:",round(r2_score(df_test3['Yield'].sort_values(), df_predict_epic_hist['Yield'].sort_values()),2))
print("R2 OBS-RF:Clim:",round(r2_score(df_test3['Yield'].sort_values(), df_predict_clim_hist['Yield'].sort_values()),2))
print("R2 OBS-Hybrid:",round(r2_score(df_test3['Yield'].sort_values(), df_predict_test_hist['Yield'].sort_values()),2))

#%%
 

# Test on aggregated data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# define the model
model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, max_depth = 20)
model.fit(X_train.values, y_train)

df_hybrid_aggregated = df_hybrid_batch.groupby(['time']).mean()
X_test = df_hybrid_aggregated.values
y_pred = model.predict(X_test)
y_test = df_test2.groupby(['time']).mean().values.ravel()

print("R2 on test set:",round(r2_score(y_test, y_pred),2))
print("Var score on test set:",round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:",round(mean_absolute_error(y_test, y_pred),5))
print("RMSE  on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
print("______")

# Correct mean bias pure EPIC
DS_epic_meanbiascor = DS_y_epic_br_clip_det.mean(['lat','lon']) - ( DS_y_epic_br_clip_det.mean() - DS_y_obs_up_clip_det.mean())

# Compare timelines
df_predict_test_hist = df_test2.copy()
predict_test_hist = model_hyb.predict(df_hybrid_batch)
df_predict_test_hist.loc[:,"Yield"] = predict_test_hist
DS_predict_test_hist = xr.Dataset.from_dataframe(df_predict_test_hist)


# RF:EPIC
df_predict_epic_hist = df_test2.copy()
predict_epic_hist = model_epic.predict(df_epic_det_forhybrid)
df_predict_epic_hist.loc[:,"Yield"] = predict_epic_hist
DS_predict_epic_hist = xr.Dataset.from_dataframe(df_predict_epic_hist)


# RF:ECE
df_predict_clim_hist = df_test2.copy()
predict_clim_hist = model_clim.predict(df_feature_season_6mon_br)
df_predict_clim_hist.loc[:,"Yield"] = predict_clim_hist
DS_predict_clim_hist = xr.Dataset.from_dataframe(df_predict_clim_hist)

# PLOTS
plt.figure(figsize=(10,6), dpi=300) #plot clusters
plt.plot(DS_y_obs_up_clip_det.time, DS_y_obs_up_clip_det.mean(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_y_epic_br_clip_det.time, DS_epic_meanbiascor, label = 'Original EPIC',linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_predict_epic_hist['Yield'].mean(['lat','lon']), label = 'RF:EPIC')
plt.plot(DS_predict_clim_hist.time, DS_predict_clim_hist['Yield'].mean(['lat','lon']), label = 'RF:ECE')
plt.plot(DS_predict_test_hist.time, DS_predict_test_hist['Yield'].mean(['lat','lon']), label = 'RF:Hybrid')
plt.legend()
plt.show()

### WIEGHTED ANALYSIS
DS_harvest_area_globiom = xr.open_dataset('../../paper_hybrid_agri/data/soy_harvest_area_globiom_05x05_2b.nc').mean('time')
DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['harvest_area'].where(DS_predict_test_hist['Yield']>0)

total_area = DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
DS_obs_weighted = ((DS_y_obs_up_clip_det * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_hybrid_weighted = ((DS_predict_test_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_epic_weighted = ((DS_predict_epic_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_clim_weighted = ((DS_predict_clim_hist['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')

# Weighted plot
plt.figure(figsize=(8,5), dpi=300) #plot clusters
plt.plot(DS_predict_epic_hist.time, DS_obs_weighted['Yield'].sum(['lat','lon']), label = 'Obs', linestyle='dashed',linewidth=3)
plt.plot(DS_predict_epic_hist.time, DS_epic_weighted['Yield'].sum(['lat','lon']), label = 'RF:EPIC')
plt.plot(DS_predict_epic_hist.time, DS_clim_weighted['Yield'].sum(['lat','lon']), label = 'RF:CLIM')
plt.plot(DS_predict_epic_hist.time, DS_hybrid_weighted['Yield'].sum(['lat','lon']), label = 'RF:hybrid')
plt.ylabel('Yield (ton/ha)')
plt.xlabel('Years')
plt.legend()
plt.show()

print("R2 OBS-Hybrid:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_hybrid_weighted['Yield'].sum(['lat','lon'])),2))
print("R2 OBS-Clim:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_clim_weighted['Yield'].sum(['lat','lon'])),2))
print("R2 OBS-EPIC:",round(r2_score(DS_obs_weighted['Yield'].sum(['lat','lon']), DS_epic_weighted['Yield'].sum(['lat','lon'])),2))
print("R2 Hybrid-clim:",round(r2_score(DS_hybrid_weighted['Yield'].sum(['lat','lon']), DS_clim_weighted['Yield'].sum(['lat','lon'])),2))

def plots_deviation(year_to_use = 2012):    
    
    # predict 2012 year
    v_max = 2
    v_min = -2
    DS_y_dif_2012 = DS_y_obs_up_clip_det.sel(time=year_to_use) 
    DS_y_dif_2012.attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}
    
    # Predict 2012 hybrid model
    df_predict_2012 = df_test2.loc[year_to_use].copy() #df_test2.query('time==2012').copy()
    df_predict_2012.loc[:,'Yield'] =  model.predict(df_hybrid_batch.query(f'time=={year_to_use}')).copy()
    df_predict_2012 = df_predict_2012.reorder_levels(['lat','lon']).sort_index(ascending = [True,True])
    
    DS_predict_2012 = xr.Dataset.from_dataframe(df_predict_2012)
    DS_predict_2012 = DS_predict_2012.sortby('lat')
    DS_predict_2012 = DS_predict_2012.sortby('lon')
    
    DS_y_dif_predict_2012 = DS_predict_2012
    DS_y_dif_predict_2012['Yield'].attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}
    
    # EPIC prediction for 2012
    df_pred_epic_2012 = df_test2.loc[year_to_use].copy() #df_test2.query('time==2012').copy()
    df_pred_epic_2012.loc[:,'Yield'] =  model_epic.predict(df_epic_det_forhybrid.query(f'time=={year_to_use}')).copy()
    df_pred_epic_2012 = df_pred_epic_2012.reorder_levels(['lat','lon']).sort_index(ascending = [True,True])
    
    DS_pred_epic_2012 = xr.Dataset.from_dataframe(df_pred_epic_2012)
    DS_pred_epic_2012 = DS_pred_epic_2012.sortby('lat')
    DS_pred_epic_2012 = DS_pred_epic_2012.sortby('lon')
    
    DS_y_dif_pred_epic_2012 = DS_pred_epic_2012 
    DS_y_dif_pred_epic_2012['Yield'].attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}
    
    # EPIC without RF
    df_epic_2012 = df_epic_det_forhybrid.query(f'time=={year_to_use}').copy()
    
    DS_epic_2012 = xr.Dataset.from_dataframe(df_epic_2012)
    DS_epic_2012 = DS_epic_2012.sortby('lat')
    DS_epic_2012 = DS_epic_2012.sortby('lon')
    
    DS_y_dif_epic_2012 = DS_epic_2012
    DS_y_dif_epic_2012['yield-soy-noirr'].attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}
        
    # plt.figure(figsize=(11,6), dpi=300) #plot clusters
    # ax=plt.axes(projection=ccrs.PlateCarree())
    # DS_y_dif_2012.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
    # ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    # ax.set_title(f'{year_to_use} yield deviation')
    # ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
    # plt.tight_layout()
    # # plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
    # plt.show()
    
    # plt.figure(figsize=(11,6), dpi=300) #plot clusters
    # ax=plt.axes(projection=ccrs.PlateCarree())
    # DS_y_dif_pred_epic_2012['Yield'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
    # ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    # ax.set_title(f'{year_to_use} yield EPIC deviation')
    # ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
    # plt.tight_layout()
    # # plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
    # plt.show()
    
    
    # plt.figure(figsize=(11,6), dpi=300) #plot clusters
    # ax=plt.axes(projection=ccrs.PlateCarree())
    # DS_y_dif_predict_2012['Yield'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
    # ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    # ax.set_title(f'{year_to_use} yield predicted deviation')
    # ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
    # plt.tight_layout()
    # # plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
    # plt.show()
    
    #TEST
       
    DS_epic_2012 = xr.Dataset.from_dataframe(df_epic_det)
    DS_epic_2012 = DS_epic_2012.sortby('lat')
    DS_epic_2012 = DS_epic_2012.sortby('lon')
    
    # plot_2d_map(DS_epic_2012['yield-soy-noirr'].sel(time = 2012))


    
    # difference prediction and real data
    DS_error_prediction = DS_y_dif_predict_2012 - DS_y_dif_2012
    DS_error_prediction['Yield'].attrs = {'long_name': 'Yield error prediction', 'units':'ton/ha'}
    
    
    plt.figure(figsize=(11,6), dpi=300) #plot clusters
    ax=plt.axes(projection=ccrs.PlateCarree())
    DS_error_prediction['Yield'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
    ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_title(f'Predicted Hybrid - Observed ({year_to_use})')
    ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
    plt.tight_layout()
    # plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
    plt.show()
    
    
    # difference prediction and real data
    DS_error_prediction = DS_y_dif_pred_epic_2012 - DS_y_dif_2012
    DS_error_prediction['Yield'].attrs = {'long_name': 'Yield error prediction', 'units':'ton/ha'}
    
    
    plt.figure(figsize=(11,6), dpi=300) #plot clusters
    ax=plt.axes(projection=ccrs.PlateCarree())
    DS_error_prediction['Yield'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
    ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_title(f'Predicted RF:EPIC - Observed ({year_to_use})')
    ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
    plt.tight_layout()
    # plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
    plt.show()
    
    # difference prediction and real data
    DS_error_prediction = DS_y_dif_epic_2012 - DS_y_dif_2012
    DS_error_prediction['yield-soy-noirr'].attrs = {'long_name': 'Yield error prediction', 'units':'ton/ha'}
    
    
    plt.figure(figsize=(11,6), dpi=300) #plot clusters
    ax=plt.axes(projection=ccrs.PlateCarree())
    DS_error_prediction['yield-soy-noirr'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
    ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_title(f'Predicted original EPIC - Observed ({year_to_use})')
    ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
    plt.tight_layout()
    # plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
    plt.show()
    
    
    return DS_y_dif_predict_2012, DS_y_dif_epic_2012
    
    
# 2012
DS_y_predict_2012, DS_y_pure_epic_2012 = plots_deviation(2012)
DS_y_predict_shift_2012 = DS_y_predict_2012 / DS_y_obs_up_clip_det.mean('time')
DS_y_pure_epic_dif_2012 = DS_y_pure_epic_2012 / DS_y_epic_br_clip_det.mean('time')

plot_2d_map(DS_y_predict_shift_2012['Yield'])
plot_2d_map( DS_y_obs_up_clip_det.sel(time=2012) / DS_y_obs_up_clip_det.mean('time') )

DS_y_predict_shift_2012.to_netcdf("hybrid_yield_soybean_2012_shift.nc")
DS_y_pure_epic_dif_2012.to_netcdf("epic_yield_soybean_2012_shift.nc")

# 2005
DS_y_predict_shift_2005, DS_y_pure_epic_2005 = plots_deviation(2005)
DS_y_predict_shift_2005 = DS_y_predict_shift_2005 / DS_y_obs_up_clip_det.mean('time')
DS_y_pure_epic_dif_2005 = DS_y_pure_epic_2005 / DS_y_epic_br_clip_det.mean('time')

DS_y_predict_shift_2005.to_netcdf("ds_yield_soybean_2005_shift.nc")
DS_y_pure_epic_dif_2005.to_netcdf("epic_yield_soybean_2005_shift.nc")
# ADD EPIC ANOTHER COLUMN< USE DIVISION + 2005







v_min = -2
v_max = 2
# predict 2005 year
DS_y_dif_2012 = DS_y_obs_up_clip_det.sel(time=2005) - DS_y_obs_up_clip_det.mean('time')
DS_y_dif_2012.attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}

# Predict 2005
df_predict_2012 = df_test2.loc[2005].copy() #df_test2.query('time==2012').copy()
df_predict_2012.loc[:,'Yield'] =  model.predict(df_hybrid_batch.query('time==2005')).copy()
df_predict_2012 = df_predict_2012.reorder_levels(['lat','lon']).sort_index(ascending = [True,True])

DS_predict_2012 = xr.Dataset.from_dataframe(df_predict_2012)
DS_predict_2012 = DS_predict_2012.sortby('lat')
DS_predict_2012 = DS_predict_2012.sortby('lon')

DS_y_dif_predict_2012 = DS_predict_2012 - DS_y_obs_up_clip_det.mean('time')
DS_y_dif_predict_2012['Yield'].attrs = {'long_name': 'Yield anomaly', 'units':'ton/ha'}

plt.figure(figsize=(11,6), dpi=300) #plot clusters
ax=plt.axes(projection=ccrs.PlateCarree())
DS_y_dif_2012.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_title('2005 yield deviation')
ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
plt.tight_layout()
# plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
plt.show()

plt.figure(figsize=(11,6), dpi=300) #plot clusters
ax=plt.axes(projection=ccrs.PlateCarree())
DS_y_dif_predict_2012['Yield'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_title('2005 yield predicted deviation')
ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
plt.tight_layout()
# plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
plt.show()

# difference prediction and real data
DS_error_prediction = DS_y_dif_predict_2012 - DS_y_dif_2012
DS_error_prediction['Yield'].attrs = {'long_name': 'Yield error prediction', 'units':'ton/ha'}


plt.figure(figsize=(11,6), dpi=300) #plot clusters
ax=plt.axes(projection=ccrs.PlateCarree())
DS_error_prediction['Yield'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, cmap=plt.cm.seismic_r, vmin=v_min, vmax=v_max)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_title('Predicted - Observed (2005)')
ax.set_extent([-65,-32,-5,-35], ccrs.Geodetic())
plt.tight_layout()
# plt.savefig('paper_figures/us_map_2012_yield.png', format='png', dpi=300)
plt.show()



#%% TEST SINGLE VARIABLE TEMPERATURE

# Evaluate Model
print('TEST results for Deep neural network:')
X, y = df_hybrid_batch[['yield-soy-noirr','tnx_4']], df_test3.values.ravel()
# y_pred_hyb_rf, y_pred_total_hyb_rf, model_hyb_rf, full_model_hyb_rf = calibration(X, y, type_of_model='RF', stack_model = False)
y_pred_hyb_test, y_pred_total_hyb_test, model_hyb_test, full_model_hyb_test = calibration(X, y, type_of_model='DNN', stack_model = False)


X_fut = df_hybrid_proj_ukesm_585[['yield-soy-noirr_2015co2','tnx_4']]
X_fut = X_fut.rename(columns={'yield-soy-noirr_2015co2':'yield-soy-noirr'})


df_y_pred_hist = X[['yield-soy-noirr']].copy()
df_y_pred_hist['yield-soy-noirr'] = full_model_hyb_test.predict(X)

df_y_pred_fut = X_fut[['yield-soy-noirr']].copy()
df_y_pred_fut['yield-soy-noirr'] = full_model_hyb_test.predict(X_fut)

plt.plot(df_y_pred_hist['yield-soy-noirr'].groupby('time').mean(), label = 'History')
plt.plot(df_y_pred_fut['yield-soy-noirr'].groupby('time').mean(), label = 'Future')
plt.legend()
plt.plot()

# Plots for points distribution
for feature in X_fut.columns:
    df_clim_extrapolated = X_fut[feature].where(X_fut[feature] > X[feature].max()).dropna()
    df_y_extrapolated = df_y_pred_fut['yield-soy-noirr'].where(X_fut[feature] > X[feature].max()).dropna()

    plt.scatter(X[feature], df_y_pred_hist['yield-soy-noirr'], color = 'k', label = 'History')    
    plt.scatter(X_fut[feature], df_y_pred_fut['yield-soy-noirr'], alpha = 0.8, label = 'Projection')
    plt.hlines(df_y_pred_hist['yield-soy-noirr'].mean(), X[feature].min(), X[feature].max(), color = 'k')
    plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8, label = 'Extrapolation')
    plt.legend(loc="upper right")
    plt.title(f'Scatterplot of {feature} for GCM-RCPs')
    plt.ylabel('Yield')
    if feature in ['tnx_3','tnx_4','tnx_5']:
        x_label = 'Temperature (°C)'
    elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
        x_label = 'Precipitation (mm/month)'
    else:
        x_label = 'Yield (ton/ha)'
          
    plt.xlabel(x_label)
    plt.show()

for feature in X_fut.columns:   
    sns.kdeplot(X[feature],fill=True, alpha = 0.3, label = 'History')
    sns.kdeplot(X_fut[feature],fill=True, alpha = 0.3, label = 'Proj')
    plt.legend()
    plt.show()
    
sns.kdeplot(df_y_pred_hist['yield-soy-noirr'], fill=True, alpha = 0.3, label = 'History')
sns.kdeplot(df_y_pred_fut['yield-soy-noirr'],fill=True, alpha = 0.3, label = 'Proj')
plt.legend()
plt.show()

# for feature in df_hybrid_proj_ukesm_585.columns:
#     df_clim_extrapolated = df_hybrid_proj_ukesm_585[feature].where(df_hybrid_proj_ukesm_585[feature] < df_hybrid_us_2_br[feature].min()).dropna()
#     df_y_extrapolated = df_prediction_proj_ukesm_585['yield-soy-noirr'].where(df_hybrid_proj_ukesm_585[feature] < df_hybrid_us_2_br[feature].min()).dropna()

#     plt.scatter(df_hybrid_us_2_br[feature], df_predict_test_hist['Yield'], color = 'k')    
#     plt.scatter(df_hybrid_proj_ukesm_585[feature], df_prediction_proj_ukesm_585['yield-soy-noirr'], alpha = 0.8)
#     plt.hlines(df_predict_test_hist['Yield'].mean(), df_hybrid_us_2_br[feature].min(), df_hybrid_us_2_br[feature].max(), color = 'k')
#     plt.scatter(df_clim_extrapolated, df_y_extrapolated, alpha = 0.8)
#     # plt.legend(loc="upper right")
#     plt.title(f'Scatterplot of {feature} for GCM-RCPs')
#     plt.ylabel('Yield')
#     if feature in ['tnx_3','tnx_4','tnx_5']:
#         x_label = 'Temperature (°C)'
#     elif feature in ['prcptot_3', 'prcptot_4', 'prcptot_5']:
#         x_label = 'Precipitation (mm/month)'
#     else:
#         x_label = 'Yield (ton/ha)'
          
#     plt.xlabel(x_label)
#     plt.show()
    

### Partial dependence plots
from sklearn.inspection import PartialDependenceDisplay

features_to_plot = [0,1]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(full_model_hyb_test, X, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0,1), ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(full_model_hyb_test, X_fut, features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99), pd_line_kw={'color':'k'})
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(1.0, 2.8)
ax2.set_ylim(1.0, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()

#%%
DS_predict_test_hist
DS_prcptot_4 = df_hybrid_batch[['tnx_5']].to_xarray()
DS_prcptot_4 = DS_prcptot_4.sortby('lat')

corr_3d = xr.corr(DS_prcptot_4["tnx_5"], DS_y_obs_up_test["Yield"], dim="time", )
plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
corr_3d.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True, levels = 10)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-80.73,-34,-35,6], ccrs.PlateCarree())
plt.show()
corr_3d_high = corr_3d.where(corr_3d > 0.4)
plot_2d_map(corr_3d_high)




