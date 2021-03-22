'''
For UNIX / CDO

STEP 1) - Locate folder:
$ cd /mnt/c/Users/morenodu/"OneDrive - Stichting Deltares"/Documents/PhD/Paper_drought/data/ec_earth_sample

STEP 2) - for ordering the whole ensemble in a single timeseries, run:
    Complete ensemble nested loop
$ for s_number in {1..16} ; do s_tag='s'$(printf '%02d' "$s_number"); for r_number in {0..24} ; do r_tag='r'$(printf '%02d' "$r_number") ; echo "$s_tag";  echo "$r_tag" ; cdo cat -apply,shifttime,$(( ($s_number-1) * 125 + $r_number * 5 ))year [ dtr_m_ECEarth_PD_"$s_tag""$r_tag"_20??.nc ] dtr_m_ECEarth_PD_ensemble_2035-4035.nc ; done; done
    One for loop
$ for r_number in {0..24} ; do r_tag='r'$(printf '%02d' "$r_number") ; echo "$r_tag" ; cdo cat -apply,-shifttime,$(( $r_number * 5 ))year [ dtr_m_ECEarth_PD_s01"$r_tag"_20??.nc ] dtr_m_ECEarth_PD_s01r00-24_2035-2159.nc ; done

STEP 3) - for increasing the resolution
$  cdo -remapycon,cru_ts4.04.1901.2019.pre.dat.nc tasmax_m_ECEarth_PD_ensemble_2035-4035.nc tasmax_m_ECEarth_PD_ensemble_2035_
4035_hr.nc
    or for decreasing resolution
$ cdo -remapycon,tasmax_m_ECEarth_PD_ensemble_2035-4035.nc cru_ts4.04.1901.2019.tmx.dat.nc cru_ts4.04.1901.2019.tmx.dat_lr.nc
    
'''

import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from  scipy import signal 
import seaborn as sns
from mask_shape_border import mask_shape_border
from failure_probability import failure_probability
from shap_prop import shap_prop
import glob
from xclim import ensembles
#%% Openinig and cleaning data

# Detrending data
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

# Needs to import DS_Yield from other script
DS_y_lr = xr.open_dataset("epic_soy_yield_us_lr.nc",decode_times=True)
mask_ref = DS_y_lr['yield'].mean('time')
# CRU data
DS_t_max_cru = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.tmx.dat_lr.nc",decode_times=True).sel(time=slice('31-12-1959', '31-12-2020'))
DS_t_max_cru_us = DS_t_max_cru.where(mask_ref > -0.1 ) # mask

DS_dtr_cru = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.dtr.dat_lr.nc",decode_times=True).sel(time=slice('31-12-1959', '31-12-2020'))
DS_dtr_cru_us = DS_dtr_cru.where(mask_ref > -0.1 )

DS_pre_cru = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pre.dat_lr.nc",decode_times=True).sel(time=slice('31-12-1959', '31-12-2020'))
DS_pre_cru_us = DS_pre_cru.where(mask_ref > -0.1 )

# detrend 
DS_tmx_cru_us_det = xr.DataArray( detrend_dim(DS_t_max_cru_us.tmx, 'time') + DS_t_max_cru_us.tmx.mean('time'), name= DS_t_max_cru_us.tmx.name, attrs = DS_t_max_cru_us.tmx.attrs)
DS_dtr_cru_us_det = xr.DataArray( detrend_dim(DS_dtr_cru_us.dtr, 'time') + DS_dtr_cru_us.dtr.mean('time'), name= DS_dtr_cru_us.dtr.name)
DS_pre_cru_us_det = xr.DataArray( detrend_dim(DS_pre_cru_us.pre, 'time') + DS_pre_cru_us.pre.mean('time'), name= DS_pre_cru_us.pre.name)


DS_cru_merge = xr.merge([DS_tmx_cru_us_det, DS_dtr_cru_us_det, DS_pre_cru_us_det])
DS_cru_merge.coords['lon'] = (DS_cru_merge.coords['lon'] + 180) % 360 - 180
DS_cru_merge = DS_cru_merge.sortby(DS_cru_merge.lon)
DS_cru_merge = DS_cru_merge.dropna(dim = 'lon', how='all')
DS_cru_merge = DS_cru_merge.dropna(dim = 'lat', how='all')

# Test detrending
df_dtr = DS_dtr_cru_us.dtr.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_dtr_8 = df_dtr[df_dtr.index.month == 8]

df_dtr_det = DS_dtr_cru_us_det.to_dataset().to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_dtr_det_8 = df_dtr_det[df_dtr_det.index.month == 8]

plt.plot(df_dtr_8, label = '8-normal')
plt.plot(df_dtr_det_8, label = "8-det")
plt.legend()
plt.show()

# EC-Earth
def open_regularize(address_file, reference_file):    
    DS_ec = xr.open_dataset(address_file,decode_times=True) 
    if list(DS_ec.keys())[1] == 'tasmax':
        da_ec = DS_ec[list(DS_ec.keys())[1]] - 273.15
    elif list(DS_ec.keys())[1] == 'pr':
        da_ec = DS_ec[list(DS_ec.keys())[1]] * 1000 
    else:
        da_ec = DS_ec[list(DS_ec.keys())[1]]
    DS_ec = da_ec.to_dataset() 
    DS_ec_crop = DS_ec.where(reference_file.mean('time') > -300 )
    return DS_ec_crop

#Temp - Kelvin to celisus
DS_tmx_ec = open_regularize("EC_earth_PD/tasmax_m_ECEarth_PD_ensemble_2035-4035.nc", DS_pre_cru_us['pre']) 
# precipitation
DS_pre_ec = open_regularize("EC_earth_PD/pr_m_ECEarth_PD_ensemble_2035-4035.nc", DS_pre_cru_us['pre']) 
# dtr
DS_dtr_ec = open_regularize("EC_earth_PD/dtr_m_ECEarth_PD_ensemble_2035-4035.nc", DS_pre_cru_us['pre']) 

# Test plot to see if it's good    
plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_t_max_cru_us['tmx'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

# Test plot to see if it's good    
plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_dtr_cru_us['dtr'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

# Measure diference between model and observed data
subt = DS_tmx_ec['tasmax'].mean('time') - DS_t_max_cru_us['tmx'].mean('time')

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
subt.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
ax.set_title('Difference between CRU and EC-earth')
plt.show()

#%% BIAS CORRECTION - Convert and tests
from xclim import sdba

def bias_analysis(obs_data, model_data):
    df_cru_cli=obs_data[list(obs_data.keys())[0]].to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
    df_ec=model_data[list(model_data.keys())[0]].to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
    
    # Compare mean annual cycle
    df_cru_year = df_cru_cli.groupby(df_cru_cli.index.month)[list(obs_data.keys())[0]].mean()
    df_ec_year = df_ec.groupby(df_ec.index.month).mean()
    # plot
    plt.plot(df_cru_year, label = 'CRU', color = 'darkblue')
    plt.plot(df_ec_year, label = 'EC-Earth_0', color = 'red' )
    plt.ylabel(obs_data[list(obs_data.keys())[0]].attrs['units'])
    plt.title(f'Mean annual cycle - {list(obs_data.keys())[0]}') 
    plt.legend(loc="lower left")
    plt.show()
    
    # Compare std each year
    df_cru_year_std = df_cru_cli.groupby(df_cru_cli.index.month)[list(obs_data.keys())[0]].std()
    df_ec_year_std = df_ec.groupby(df_ec.index.month).std()
    # plot
    plt.plot(df_cru_year_std, label = 'CRU', color = 'darkblue')
    plt.plot(df_ec_year_std, label = 'EC-Earth_0', color = 'red' )
    plt.ylabel(obs_data[list(obs_data.keys())[0]].attrs['units'])
    plt.title(f'Variability around the mean - {list(obs_data.keys())[0]}')
    plt.legend(loc="lower left")
    plt.show()
    
    # Compare Q-Q plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
    stats.probplot(df_ec.iloc[:,0], dist=stats.beta, sparams=(2,3),plot=plt,fit=False)
    stats.probplot(df_cru_cli.iloc[:,0], dist=stats.beta, sparams=(2,3), plot=plt,fit=False)
    ax.get_lines()[0].set_color('C1')
    plt.show()

    
# bias correction for tasmax Quantile mapping - First compare the original, then adjust the bias and compare the bias corrected version
bias_analysis(DS_t_max_cru_us, DS_tmx_ec)
dqm_tmx = sdba.adjustment.EmpiricalQuantileMapping( group='time.month', kind='+')
dqm_tmx.train(DS_t_max_cru_us['tmx'],DS_tmx_ec['tasmax'])
DS_tmx_ec_cor = dqm_tmx.adjust(DS_tmx_ec['tasmax'], interp='linear')
DS_tmx_ec_cor = DS_tmx_ec_cor.to_dataset(name= 'tmx')
bias_analysis(DS_t_max_cru_us, DS_tmx_ec_cor)

# bias correction for dtr Quantile mapping - First compare the original, then adjust the bias and compare the bias corrected version
bias_analysis(DS_dtr_cru_us, DS_dtr_ec)
dqm_dtr = sdba.adjustment.EmpiricalQuantileMapping( group='time.month', kind='+')
dqm_dtr.train(DS_dtr_cru_us['dtr'],DS_dtr_ec['dtr'])
DS_dtr_ec_cor = dqm_dtr.adjust(DS_dtr_ec['dtr'], interp='linear')
DS_dtr_ec_cor = DS_dtr_ec_cor.to_dataset(name= 'dtr')
bias_analysis(DS_dtr_cru_us, DS_dtr_ec_cor)

# bias correction for precipitation Quantile mapping - First compare the original, then adjust the bias and compare the bias corrected version
bias_analysis(DS_pre_cru_us, DS_pre_ec)
dqm_pr = sdba.adjustment.DetrendedQuantileMapping(nquantiles=100, group='time.month', kind='+')
dqm_pr.train(DS_pre_cru_us['pre'],DS_pre_ec['pr'])
DS_pre_ec_cor = dqm_pr.adjust(DS_pre_ec['pr'], interp='linear')
DS_pre_ec_cor = DS_pre_ec_cor.to_dataset(name= 'pre')
bias_analysis(DS_pre_cru_us, DS_pre_ec_cor)

# Merge in one dataset
DS_cli_ec = xr.merge([DS_tmx_ec_cor.tmx, DS_dtr_ec_cor.dtr, DS_pre_ec_cor.pre])

# new diference
for feature in list(DS_cli_ec.keys()):
    subt_cor = DS_cli_ec[feature].mean('time') - DS_cru_merge[feature].mean('time')
    
    plt.figure(figsize=(20,10)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    subt_cor.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,levels=10)
    ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-105,-65,20,50], ccrs.PlateCarree())
    ax.set_title(f'Difference between CRU and EC-earth for {feature}')
    plt.show()

# Features considered for this case
months_to_be_used = [7,8]
column_names = [i+str(j) for i in list(DS_cli_ec.keys()) for j in months_to_be_used]

df_features_ec = []
for feature in list(DS_cli_ec.keys()):        
    df_feature_2 = DS_cli_ec[feature].to_dataframe().groupby(['time']).mean()
    df_feature_2_reshape = reshape_data(df_feature_2).loc[:,months_to_be_used]
    df_features_ec.append(df_feature_2_reshape)    

df_features_ec = pd.concat(df_features_ec, axis=1) # format data
df_features_ec.columns = column_names

# Adapt the structure to match the RF structure
df_features_ec_season = pd.concat( [df_features_ec.iloc[:,0:2].mean(axis=1),df_features_ec.iloc[:,2:4].mean(axis=1),df_features_ec.iloc[:,4:6].mean(axis=1)], axis=1 )
df_features_ec_season.columns=['tmx_7_8','dtr_7_8', 'precip_7_8']

#%% 2C simulation - future data

#Temp - Kelvin to celisus
DS_tmx_ec_2C = open_regularize("EC_earth_2C/tasmax_m_ECEarth_2C_ensemble_2062-4062.nc", DS_pre_cru_us['pre']) 
# precipitation
DS_pre_ec_2C = open_regularize("EC_earth_2C/pr_m_ECEarth_2C_ensemble_2062-4062.nc", DS_pre_cru_us['pre']) 
# dtr
DS_dtr_ec_2C = open_regularize("EC_earth_2C/dtr_m_ECEarth_2C_ensemble_2062-4062.nc", DS_pre_cru_us['pre']) 

# bias correction for tasmax
bias_analysis(DS_t_max_cru_us, DS_tmx_ec_2C)
DS_tmx_ec_2C_cor = dqm_tmx.adjust(DS_tmx_ec_2C['tasmax'], interp='linear')
DS_tmx_ec_2C_cor = DS_tmx_ec_2C_cor.to_dataset(name= 'tmx')
bias_analysis(DS_t_max_cru_us, DS_tmx_ec_2C_cor)

# bias correction for dtr Quantile mapping
bias_analysis(DS_dtr_cru_us, DS_dtr_ec_2C)
DS_dtr_ec_2C_cor = dqm_dtr.adjust(DS_dtr_ec_2C['dtr'], interp='linear')
DS_dtr_ec_2C_cor = DS_dtr_ec_2C_cor.to_dataset(name= 'dtr')
bias_analysis(DS_dtr_cru_us, DS_dtr_ec_2C_cor)

# bias correction for precipitation Quantile mapping
bias_analysis(DS_pre_cru_us, DS_pre_ec_2C)
DS_pre_ec_2C_cor = dqm_pr.adjust(DS_pre_ec_2C['pr'], interp='linear')
DS_pre_ec_2C_cor = DS_pre_ec_2C_cor.to_dataset(name= 'pre')
bias_analysis(DS_pre_cru_us, DS_pre_ec_2C_cor)

# Merge in one dataset
DS_cli_ec_2C = xr.merge([DS_tmx_ec_2C_cor.tmx, DS_dtr_ec_2C_cor.dtr, DS_pre_ec_2C_cor.pre])

# Prepare data
df_features_ec_2C = []
for feature in list(DS_cli_ec_2C.keys()):        
    df_feature_2 = DS_cli_ec_2C[feature].to_dataframe().groupby(['time']).mean()
    df_feature_2_reshape = reshape_data(df_feature_2).loc[:,months_to_be_used]
    df_features_ec_2C.append(df_feature_2_reshape)    

df_features_ec_2C = pd.concat(df_features_ec_2C, axis=1)
df_features_ec_2C.columns = column_names

# Adapt the structure to match the RF structure
df_features_ec_2C_season = pd.concat( [df_features_ec_2C.iloc[:,0:2].mean(axis=1),df_features_ec_2C.iloc[:,2:4].mean(axis=1),df_features_ec_2C.iloc[:,4:6].mean(axis=1)], axis=1 )
df_features_ec_2C_season.columns=['tmx_7_8','dtr_7_8', 'precip_7_8']
    
#%% DONE WITH BIAS ADJUSTMENT AND CLIMATE DATA ##################
#                                                               #
# Start from below as above it takes a long time to bias adjust #
#################################################################
#%% DEPLOYMENT OF MODEL - clean and apply for ML prediction

# Predictions for Observed data PD
y_pred = brf_model.predict(df_features_ec_season)
score_prc = sum(y_pred)/len(y_pred) 
print("The ratio of failure seasons by total seasons is:", score_prc)
probs = brf_model.predict_proba(df_features_ec_season)

# PERMUTATION ---------- Preliminary analysis shows that precipitation is such an important variable that it alone can predict a good amount of the failures, irrespective of the other variables
df_features_ec_season_permuted = df_features_ec_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
df_features_ec_season_permuted2 = pd.concat( [df_features_ec_season['tmx_7_8'], df_features_ec_season['dtr_7_8'], df_features_ec_season['precip_7_8'][::-1] ],axis=1)

# Predictions for permuted
y_pred_perm = brf_model.predict(df_features_ec_season_permuted)
score_prc_perm = sum(y_pred_perm)/len(y_pred_perm) 
print("The ratio of failure seasons PERMUTED by total seasons is:", score_prc_perm)
probs_perm = brf_model.predict_proba(df_features_ec_season_permuted)

# Difference between obs. and permuted
print("The difference between predicted failures in observed data and permuted data is:", score_prc - score_prc_perm)

# put them together in the same dataframe for plotting
probs_agg=pd.DataFrame( [probs[:,1],probs_perm[:,1]]).T
probs_agg.columns=['Ordered','Permuted']

# plots comparing prediction confidence for obs and perumuted
probs_agg_melt = probs_agg.melt(value_name='Failure probability').assign(data='Density')
ax = sns.violinplot(data=probs_agg_melt, x="data", y='Failure probability',hue='variable', split=True, inner="quartile",bw=.1)
sns.displot(probs_agg_melt, x="Failure probability",hue="variable", kind="kde", fill='True')
sns.displot(probs_agg_melt, x="Failure probability",hue="variable",kde=False)
plt.show()

# Compare the number of cases above a failure threshold
thresholds=range(0,105,5)
fails_prob_together = np.empty([len(thresholds),2])
i=0
for prc in thresholds: 
    print(f'The number of observed seasons with failure probability over {prc}% is:', len(probs[:,1][probs[:,1]>prc/100]), 'and permuted is: ',len(probs_perm[:,1][probs_perm[:,1]>prc/100]))
    fails_prob_together[i,:] = (len(probs[:,1][probs[:,1]>prc/100]),len(probs_perm[:,1][probs_perm[:,1]>prc/100]))
    i=i+1
df_fails_prob_together = pd.DataFrame( fails_prob_together, index = thresholds, columns = probs_agg.columns)

# Plot figure to compare the amount of cases above the thresholds
sns.lineplot(data=df_fails_prob_together)
plt.axvline(x=50, alpha=0.5,c='k',linestyle=(0, (5, 5)))
plt.ylabel('Amount of cases')
plt.xlabel('Threshold (%)')
plt.title('Number of cases above a failure prediction level')

#%% FURTHER EXPLORATION: train explainer shap, explore predictions and how decisions are called
import shap
explainer = shap.TreeExplainer(brf_model)
for mode in [df_features_ec_season,df_features_ec_season_permuted]:
    shap_values = explainer.shap_values(mode, approximate=True, check_additivity=True)
    
    # dependence plots
    for name in mode.columns:
        shap.dependence_plot(name, shap_values[1], mode)
       
    # Summary plots
    shap.summary_plot(shap_values, mode, plot_type="bar")
    shap.summary_plot(shap_values[1], mode, plot_type="bar")
    shap.summary_plot(shap_values[1], mode) # Failure
    
    # Decision plots explaining decisions to classify
    shap.decision_plot(explainer.expected_value[1], shap_values[1], mode)
    shap.decision_plot(explainer.expected_value[1], shap_values[1][1], mode.iloc[1]) #2012 year
    
    # Calculate force plot for a given value 2012
    shap.initjs() 
    shap_values_2012 = explainer.shap_values( mode.iloc[[4]])
    shap_display = shap.force_plot(explainer.expected_value[1], shap_values_2012[1], mode.iloc[[4]],matplotlib=True)
 
#%% Predictions for 2C degree
y_pred_2C = brf_model.predict(df_features_ec_2C_season)
score_prc_2C = sum(y_pred_2C)/len(y_pred_2C) 
print("The ratio of failure seasons by total seasons for 2C is:", score_prc_2C)
probs_2C = brf_model.predict_proba(df_features_ec_2C_season)

print("The ratio between failures in 2C and PD is",score_prc_2C/score_prc)
print("The increase in failures between 2C and PD is", (score_prc_2C - score_prc)*100,"%")

# put them together in the same dataframe for plotting
probs_agg_t2=pd.DataFrame( [probs[:,1],probs_2C[:,1]]).T
probs_agg_t2.columns=['Present','2C']

# plots comparing prediction confidence for each arrangement of data
probs_agg_t2_melt = probs_agg_t2.melt(value_name='Failure probability').assign(data='Density')
ax = sns.violinplot(data=probs_agg_t2_melt, x="data", y='Failure probability',hue='variable', split=True, inner="quartile",bw=.1)
sns.displot(probs_agg_t2_melt, x="Failure probability",hue="variable", kind="kde", fill='True')
sns.displot(probs_agg_t2_melt, x="Failure probability",hue="variable",kde=False)
plt.show()
# Compare the number of cases above a failure threshold
fails_prob_together_2C = np.empty([len(thresholds),2])
i=0
for prc in thresholds: 
    print(f'The number of PD seasons with failure probability over {prc}% is:', len(probs[:,1][probs[:,1]>prc/100]), 'and 2C is: ',len(probs_2C[:,1][probs_2C[:,1]>prc/100]))
    fails_prob_together_2C[i,:] = (len(probs[:,1][probs[:,1]>prc/100]),len(probs_2C[:,1][probs_2C[:,1]>prc/100]))
    i=i+1
df_fails_prob_together_2C = pd.DataFrame( fails_prob_together_2C, index = thresholds, columns = probs_agg_t2.columns)

# Plot figure ti cinoare the amount of cases above the thresholds
sns.lineplot(data=df_fails_prob_together_2C)
plt.axvline(x=50, alpha=0.5,c='k',linestyle=(0, (5, 5)))
plt.ylabel('Amount of cases')
plt.xlabel('Threshold (%)')
plt.title('Number of cases above a failure prediction level')
#%% Compound analysis 

# filtering average weather conditions where the prediction points to failure
mean_cond = np.mean( df_features_ec_season[y_pred == 1] , axis=0)
#mean_cond_perm = np.mean( df_features_ec_season_permuted[y_pred_perm == 1] , axis=0)
    
def compound_analysis(df_features_ec_season, y_pred):
    # Scatter plot to understand the shape of the variables (correlation)
    df_features_ec_season_fail=pd.concat([df_features_ec_season,pd.DataFrame(np.array([y_pred == 1]).T,index=df_features_ec_season.index, columns=['Failure'])],axis=1)
    sns.lmplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[1], x=df_features_ec_season.columns[2],fit_reg=True, scatter_kws={"s": 10}, hue='Failure',legend_out=False)
    # plt.vlines(mean_cond['precip_7_8'],ymin = mean_cond['dtr_7_8'],ymax=np.max(df_features_ec_season_fail['dtr_7_8']), colors ='k', ls='--', alpha=0.6)
    # plt.hlines(mean_cond['dtr_7_8'],xmax = mean_cond['precip_7_8'],xmin=np.min(df_features_ec_season_fail['precip_7_8']),colors='k', ls='--', alpha=0.6 , label =f'Failure prediction domain')
    plt.title("Scatter plot and regression line")
    plt.show()
    sns.lmplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[0], x=df_features_ec_season.columns[2],fit_reg=True, scatter_kws={"s": 10}, hue='Failure',legend_out=False)
    # plt.vlines(mean_cond['precip_7_8'],ymin = mean_cond['tmx_7_8'],ymax=np.max(df_features_ec_season_fail['tmx_7_8']), colors ='k', ls='--', alpha=0.6)
    # plt.hlines(mean_cond['tmx_7_8'],xmax = mean_cond['precip_7_8'],xmin=np.min(df_features_ec_season_fail['precip_7_8']),colors='k', ls='--', alpha=0.6 , label =f'Failure prediction domain')
    plt.title("Scatter plot and regression line")
    plt.show()
    sns.lmplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[0], x=df_features_ec_season.columns[1],fit_reg=True, scatter_kws={"s": 10}, hue='Failure',legend_out=False)
    # plt.vlines(mean_cond['dtr_7_8'],ymin = mean_cond['tmx_7_8'],ymax=np.max(df_features_ec_season_fail['tmx_7_8']), colors ='k', ls='--', alpha=0.6)
    # plt.hlines(mean_cond['tmx_7_8'],xmin = mean_cond['dtr_7_8'],xmax=np.max(df_features_ec_season_fail['dtr_7_8']),colors='k', ls='--', alpha=0.6 , label =f'Failure prediction domain')
    plt.title("Scatter plot and regression line")
    plt.show()
    
    # 3D - turn off
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import   
    fig = plt.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = df_features_ec_season['tmx_7_8']
    sequence_containing_y_vals = df_features_ec_season['dtr_7_8']
    sequence_containing_z_vals = df_features_ec_season['precip_7_8']
    ax.set_xlabel('Max temp')
    ax.set_ylabel('Diurnal temp. range')
    ax.set_zlabel('Precipitation')
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    plt.show()
        
    # Correlation between variables
    corrmat = df_features_ec_season.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize = (6,6), dpi=144)
    g = sns.heatmap(df_features_ec_season[top_corr_features].corr(),annot=True, cmap="RdYlGn",vmin=-1, vmax=1)
    plt.title("Pearson's correlation")
    plt.show()
   
    # PERCENTILES - Count the frequency of 50, 78, 86 and 90prc for both variables at the same time.
    print("__________________________________________________________________________")
    for prc in [50, 78, 86, 90]:
        joint_prc = (sum(np.where( (df_features_ec_season['tmx_7_8'] > np.percentile(df_features_ec_season['tmx_7_8'],prc)) & (df_features_ec_season['dtr_7_8'] > np.percentile(df_features_ec_season['dtr_7_8'],prc)) & (df_features_ec_season['precip_7_8'] < np.percentile(df_features_ec_season['precip_7_8'],(100-prc))), 1, 0)))
        print(f"Extreme events (>{prc};<{100-prc}): The ratio of joint-occurrence ({joint_prc}) with respect to independent variables ({(len(df_features_ec_season)*((100-prc)/100)**3)}) is about:",joint_prc/(len(df_features_ec_season)*((100-prc)/100)**3))
        print(f"Extreme events (>{prc};<{100-prc}): The ratio of joint occurrences per univariate extreme level is:",joint_prc/ len(df_features_ec_season[df_features_ec_season['dtr_7_8'] > np.percentile(df_features_ec_season['dtr_7_8'],prc)]))
        
    # MEAN FAILURE CONDITIONS - Define each percentile
    from scipy import stats
    for feature in df_features_ec_season.columns:
        percentile = stats.percentileofscore(df_features_ec_season[feature], mean_cond[feature])
        print(f"The percentile of variable {feature} for mean failure conditions is:",percentile )
    
    # Occurences of conditions for mean value of each variable
    fail_tmx = len(df_features_ec_season[df_features_ec_season['tmx_7_8'] > mean_cond['tmx_7_8']])
    fail_pre = len(df_features_ec_season[df_features_ec_season['precip_7_8'] < mean_cond['precip_7_8']])
    fail_dtr = len(df_features_ec_season[df_features_ec_season['dtr_7_8'] > mean_cond['dtr_7_8']])
    
    # Comparison simultaneous occurences with univariate extremes 
    fail_joint = sum(np.where( (df_features_ec_season['tmx_7_8'] >  mean_cond['tmx_7_8']) & (df_features_ec_season['dtr_7_8'] >  mean_cond['dtr_7_8']) & (df_features_ec_season['precip_7_8'] < mean_cond['precip_7_8']), 1, 0))
    print("The joint occurrences of extreme conditions for mean failure conditions are:",fail_joint)
    print("The ratio of joint occurrences by extreme univariate occurrences (pre):",fail_joint/fail_pre)
    print("The ratio of joint occurrences by extreme univariate occurrences (dtr):",fail_joint/fail_dtr)
    print("The ratio of joint occurrences by extreme univariate occurrences (tmx):",fail_joint/fail_tmx)
    print("__________________________________________________________________________","\n")
    return fail_joint

#%% Compound analysis climate data original structure
print("Ordered---")
fail_joint_obs = compound_analysis(df_features_ec_season,y_pred)

# permutation joint analysis
print("Permuted---")
fail_joint_obs_perm = compound_analysis(df_features_ec_season_permuted,y_pred_perm)

#%% Compound analysis 2C

# PERMUTATION
df_features_ec_2C_season_permuted = df_features_ec_2C_season.apply(np.random.RandomState(seed=00).permutation, axis=0)  

# Predictions
y_pred_2C_perm = brf_model.predict(df_features_ec_2C_season_permuted)
score_prc_2C_perm = sum(y_pred_2C_perm)/len(y_pred_2C_perm) 
print("The ratio of failure seasons PERMUTED by total seasons is:", score_prc_2C_perm)
probs_2C_perm = brf_model.predict_proba(df_features_ec_2C_season_permuted)

# climate data original structure
print("2C---")
fail_joint_obs_2C = compound_analysis(df_features_ec_2C_season,y_pred_2C)
 #permutation joint analysis
print("2C permuted---")
fail_joint_obs_perm_2C = compound_analysis(df_features_ec_2C_season_permuted,y_pred_2C_perm)

# Amount of times more frequent 
print("The ratio between PD correlated joint occurrences and permuted joint occurrences is",fail_joint_obs/fail_joint_obs_perm)
print("The ratio between 2C correlated joint occurrences and permuted joint occurrences is",fail_joint_obs_2C/fail_joint_obs_perm_2C)
print("The ratio between correlated joint occurrences in the future and present",fail_joint_obs_2C/fail_joint_obs)
print("The ratio between correlated joint occurrences in the future and present PERMUTED",fail_joint_obs_perm_2C/fail_joint_obs_perm)

# Plot graphs comparing the difference between 2C and PD
for (df_features_ec_season_1,df_features_ec_2C_season_1) in zip([df_features_ec_season,df_features_ec_season_permuted],[df_features_ec_2C_season,df_features_ec_2C_season_permuted]):
    df_features_ec_season_fail_PD =pd.concat([df_features_ec_season_1,pd.DataFrame(np.array([y_pred < -1]).T,index=df_features_ec_season_1.index, columns=['Scenario'])],axis=1)
    df_features_ec_season_fail_PD['Scenario'] = 'Present day'
    df_features_ec_season_fail_2C =pd.concat([df_features_ec_2C_season_1,pd.DataFrame(np.array([y_pred_2C > -1 ]).T,index=df_features_ec_2C_season_1.index, columns=['Scenario'])],axis=1)
    df_features_ec_season_fail_2C['Scenario'] = '2C'
    df_features_ec_season_scenarios = pd.concat([df_features_ec_season_fail_PD, df_features_ec_season_fail_2C], axis= 0)
    
    sns.lmplot(data=df_features_ec_season_scenarios, y="dtr_7_8", x="precip_7_8",fit_reg=True, scatter_kws={"s": 10}, hue='Scenario',legend_out=False)
    plt.title("Climatic variables for each scenario")
    sns.lmplot(data=df_features_ec_season_scenarios, y="tmx_7_8", x="precip_7_8",fit_reg=True, scatter_kws={"s": 10}, hue='Scenario',legend_out=False)
    plt.title("Climatic variables for each scenario")
    sns.lmplot(data=df_features_ec_season_scenarios, y="tmx_7_8", x="dtr_7_8",fit_reg=True, scatter_kws={"s": 10}, hue='Scenario',legend_out=False)
    plt.title("Climatic variables for each scenario")



