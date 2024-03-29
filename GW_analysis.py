'''
Main block for climate change analysis
'''

import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
import xarray as xr 
import matplotlib.pyplot as plt
import pandas as pd
from bias_correction_masked import *
from return_period_storyline import return_period_storyline
from extrapolation_test import *
import pickle

#%% 1st - load data generated by bias correction

DS_ec_earth_PD_us = xr.open_dataset("ds_ec_earth_PD_us_lr.nc",decode_times=True, use_cftime=True)
DS_ec_earth_2C_us = xr.open_dataset("ds_ec_earth_2C_us_lr.nc",decode_times=True, use_cftime=True)
DS_ec_earth_3C_us = xr.open_dataset("ds_ec_earth_3C_us_lr.nc",decode_times=True, use_cftime=True)

# load and treat bias-adjusted ec earth projections
DS_ec_earth_PD_brs = xr.open_dataset("ds_ec_earth_PD_brs.nc",decode_times=True, use_cftime=True)
DS_ec_earth_2C_brs = xr.open_dataset("ds_ec_earth_2C_brs.nc",decode_times=True, use_cftime=True)
DS_ec_earth_3C_brs = xr.open_dataset("ds_ec_earth_3C_brs.nc",decode_times=True, use_cftime=True)

# load and treat bias-adjusted ec earth projections
DS_ec_earth_PD_ar = xr.open_dataset("ds_ec_earth_PD_ar.nc",decode_times=True, use_cftime=True)
DS_ec_earth_2C_ar = xr.open_dataset("ds_ec_earth_2C_ar.nc",decode_times=True, use_cftime=True)
DS_ec_earth_3C_ar = xr.open_dataset("ds_ec_earth_3C_ar.nc",decode_times=True, use_cftime=True)

# load and treat bias-adjusted ec earth projections
DS_ec_earth_PD_brc = xr.open_dataset("ds_ec_earth_PD_brc.nc",decode_times=True, use_cftime=True)
DS_ec_earth_2C_brc = xr.open_dataset("ds_ec_earth_2C_brc.nc",decode_times=True, use_cftime=True)


#%% 2nd - Convert datasets to dataframe adjusted for ML
df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us = function_conversion(DS_ec_earth_PD_us, DS_ec_earth_2C_us,DS_cli_ec_3C=DS_ec_earth_3C_us, months_to_be_used=[7,8])

df_features_ec_season_brs, df_features_ec_season_2C_brs, df_features_ec_season_3C_brs = function_conversion(DS_ec_earth_PD_brs, DS_ec_earth_2C_brs,DS_cli_ec_3C=DS_ec_earth_3C_brs, months_to_be_used=[1,2], water_year= True)

df_features_ec_season_ar, df_features_ec_season_2C_ar, df_features_ec_season_3C_ar = function_conversion(DS_ec_earth_PD_ar, DS_ec_earth_2C_ar,DS_cli_ec_3C=DS_ec_earth_3C_ar, months_to_be_used=[1,2,3], water_year= True)

# Careful with this dataset as it needs better handling of water year and low performance - TRY AGAIN
df_features_ec_season_brc, df_features_ec_season_2C_brc = function_conversion(DS_ec_earth_PD_brc, DS_ec_earth_2C_brc, months_to_be_used=[1,2], water_year= True)

#%% PDFs for all regions
list_feautures_names = ['temperature Celisus Degrees','Diurnal Temperature Range (C)', 'Precipitation mm/month']
for scenario in ['present', '2C']:
    if scenario == 'present':
        df_feature_scen_us, df_feature_scen_brs, df_feature_scen_ar, df_feature_scen_brc = df_features_ec_season_us, df_features_ec_season_brs, df_features_ec_season_ar, df_features_ec_season_brc
    
    elif scenario == '2C':
        df_feature_scen_us, df_feature_scen_brs, df_feature_scen_ar, df_feature_scen_brc = df_features_ec_season_2C_us, df_features_ec_season_2C_brs, df_features_ec_season_2C_ar, df_features_ec_season_2C_brc

    for feature in range(len(df_features_ec_season_us.columns)):
        df_con_hist = pd.concat( [df_feature_scen_us.iloc[:,feature],df_feature_scen_brs.iloc[:,feature],df_feature_scen_ar.iloc[:,feature], df_feature_scen_brc.iloc[:,feature] ],axis=1)
        df_con_hist.columns = ['us','south brazil','argentina','central brazil']
        
        plt.figure(figsize = (6,6), dpi=144)
        fig = sns.displot(df_con_hist,kind="kde", aspect=1, linewidth=3,fill=True,alpha=.2)
        fig.set(xlabel=df_feature_scen_us.columns[feature])
    
        fig._legend.set_bbox_to_anchor((.6, 0.9))
        # plt.tight_layout()
        plt.show()
        
        
#%% Comparison between EC-Earth bias adjusted and CRU for a single region
# PDFs
for feature in range(len(df_clim_agg_chosen.columns)):
    df_cru_hist = pd.DataFrame( df_clim_agg_chosen.iloc[:,feature])
    df_cru_hist['Scenario'] = 'CRU'
    df_ec_hist = pd.DataFrame( df_features_ec_season_us.iloc[:,feature])
    df_ec_hist['Scenario'] = 'EC-earth'
    df_hist_us = pd.concat( [df_cru_hist,df_ec_hist],axis=0)
    df_hist_us.index = range(len(df_hist_us))
    
    plt.figure(figsize = (6,6), dpi=144)
    fig = sns.kdeplot( data = df_hist_us, x=df_clim_agg_chosen.columns[feature], hue="Scenario", fill=True, alpha=.2, common_norm = False)
    fig.set(xlabel=df_clim_agg_chosen.columns[feature])
    plt.show()
    
    df_runs = pd.DataFrame(np.repeat(range(0, 400), 5), index = df_ec_hist.index, columns = ['run'])  
    df_feat = pd.concat( [df_ec_hist.iloc[:,0], df_runs], axis = 1)
    
    df_feat2 = pd.DataFrame(df_ec_hist.iloc[:,0])
    df_feat2.index = np.tile([2011,2012,2013,2014,2015], 400) 
    df_feat2.sort_index(inplace=True)
    
    df_ensemble = pd.DataFrame( np.reshape(df_feat.iloc[:,0].values, (5,400)))
    df_ensemble.index.name = 'index' 
    
    # sns.lineplot(x=df_feat2.index, y=df_feat2.columns[0], data=df_feat2, ci=95, estimator='mean')
    
    plt.figure(figsize = (6,6), dpi=144)
    fig = sns.lineplot( data = df_cru_hist[42:], x=df_cru_hist[42:].index, y=df_clim_agg_chosen.columns[feature])
    fig = sns.lineplot( data = df_feat2, x=df_feat2.index, y=df_feat2.columns[0], legend= False)
    fig.set(xlabel=df_clim_agg_chosen.columns[feature])
    plt.show()
    
# Quantil quantile mapping
import scipy.stats as stats
for i in [0,1,2]:   
    stats.probplot(df_clim_agg_chosen.iloc[62:,i], dist=stats.beta, sparams=(2,3),plot=plt,fit=False)
    stats.probplot(df_features_ec_season_us.iloc[:,i], dist=stats.beta, sparams=(2,3),plot=plt,fit=False)
    plt.show()
       
#%% ####3RD part - Random Forest predictions 
with open('rf_model_us.pickle', 'rb') as f:
    brf_model_us = pickle.load(f)
table_scores_us, table_events_prob2012_us = predictions_permutation(brf_model_us, df_clim_agg_chosen, df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us, df_clim_2012_us  )

table_scores_brs, table_events_prob2012_brs = predictions_permutation(brf_model_brs, input_features_brs, df_features_ec_season_brs, df_features_ec_season_2C_brs, df_features_ec_season_3C_brs, df_clim_2012 = df_clim_2012_brs )

table_scores_ar, table_events_prob2012_ar = predictions_permutation(brf_model_ar, input_features_ar,df_features_ec_season_ar, df_features_ec_season_2C_ar, df_features_ec_season_3C_ar,  df_clim_2012 = df_clim_2012_ar  )

#really weird results - poor ML performance
table_scores_brc, table_events_prob2012_brc = predictions_permutation(brf_model_brc, df_features_ec_season_brc, df_features_ec_season_2C_brc )

#%% ############### 4TH PART - Compound analysis

#US
df_joint_or_rf_us, table_JO_prob2012_us = compound_exploration(brf_model_us, df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us, df_clim_2012_us )

#BRS
df_joint_or_rf_brs, table_JO_prob2012_brs = compound_exploration(brf_model_brs, df_features_ec_season_brs, df_features_ec_season_2C_brs, df_features_ec_season_3C_brs, df_clim_2012 = df_clim_2012_brs )

#AR
df_joint_or_rf_ar, table_JO_prob2012_ar = compound_exploration(brf_model_ar, df_features_ec_season_ar, df_features_ec_season_2C_ar, df_features_ec_season_3C_ar, df_clim_2012 = df_clim_2012_ar )

#BRC CAUTION - results unstable
df_joint_or_rf_brc, table_JO_prob2012_brc = compound_exploration(brf_model_brc, df_features_ec_season_brc, df_features_ec_season_2C_brc )

#%%% Return period figure with storyline
# US
mean_conditions_similar_2012_2C_us, mean_conditions_similar_2012_3C_us = return_period_storyline(
    df_features_ec_season_us, df_features_ec_season_2C_us, df_clim_agg_chosen,
    table_JO_prob2012_us, table_events_prob2012_us, brf_model_us,
    df_clim_2012_us, df_joint_or_rf_us, proof_total_us, df_features_ec_season_3C_us)

# BRS
mean_conditions_similar_2012_2C_brs, mean_conditions_similar_2012_3C_brs = return_period_storyline(
    df_features_ec_season_brs, df_features_ec_season_2C_brs, input_features_brs,
    table_JO_prob2012_brs, table_events_prob2012_brs, brf_model_brs,
    df_clim_2012_brs, df_joint_or_rf_brs, proof_total_brs, df_features_ec_season_3C_brs)

# ARG
mean_conditions_similar_2012_2C_ar, mean_conditions_similar_2012_3C_ar = return_period_storyline(
    df_features_ec_season_ar, df_features_ec_season_2C_ar, input_features_ar,
    table_JO_prob2012_ar, table_events_prob2012_ar, brf_model_ar,
    df_clim_2012_ar, df_joint_or_rf_ar, compoundness_obs_ar, df_features_ec_season_3C_ar)


#%% Extrapolation of historical data

max_PD = df_features_ec_season_us.max(axis=0) -df_clim_agg_chosen.max(axis=0)
min_PD = df_features_ec_season_us.min(axis=0) - df_clim_agg_chosen.min(axis=0)

max_2C = df_features_ec_season_2C_us.max(axis=0) -df_clim_agg_chosen.max(axis=0)
min_2C = df_features_ec_season_2C_us.min(axis=0) - df_clim_agg_chosen.min(axis=0)

max_3C = df_features_ec_season_3C_us.max(axis=0) - df_clim_agg_chosen.max(axis=0)
min_3C = df_features_ec_season_3C_us.min(axis=0) - df_clim_agg_chosen.min(axis=0)

max_min_def = pd.DataFrame([max_2C, max_3C, min_2C, min_3C], index = ['Max 2C','Max 3C', 'Min 2C', 'Min 3C'])

# ------------------------------------------------------------------------------------------------------------------
# EXTRAPOLATION TEST
# ------------------------------------------------------------------------------------------------------------------

df_features_ec_season_us_new, df_features_ec_season_2C_us_new, df_features_ec_season_3C_us_new = extrapolation_test(df_clim_agg_chosen, df_features_ec_season_us, df_features_ec_season_2C_us, df_features_ec_season_3C_us,
                       brf_model_us, df_clim_2012_us, table_JO_prob2012_us, table_events_prob2012_us,
                       df_joint_or_rf_us, proof_total_us)

def out_in_range(out_data, in_data, scenario):
        print("MAX",scenario, out_data[out_data > in_data.max(axis=0)].count())
        print("MIN",scenario, out_data[out_data < in_data.min(axis=0)].count())
        
        max_count = out_data[out_data > in_data.max(axis=0)].count()
        min_count = out_data[out_data < in_data.min(axis=0)].count()
        max_min = pd.concat([max_count, min_count], axis = 1)    
        
        for feature in in_data:
            out_column = out_data.loc[:,feature]
            in_column = in_data.loc[:,feature]
            
            out_column[out_column > in_column.max()] = in_column.max()
            out_column[out_column < in_column.min()] = in_column.min()
        
        print("MAX CORRECTED",scenario, out_data[out_data > in_data.max(axis=0)].count())
        print("MIN CORRECTED",scenario, out_data[out_data < in_data.min(axis=0)].count())
        
        return max_min
        
    
max_min_PD = out_in_range(df_features_ec_season_us, df_clim_agg_chosen, scenario = 'PD')
max_min_2C = out_in_range(df_features_ec_season_2C_us, df_clim_agg_chosen, scenario = '2C')
max_min_3C = out_in_range(df_features_ec_season_3C_us, df_clim_agg_chosen, scenario = '3C')

table_exceedance = pd.DataFrame( pd.concat([max_min_PD, max_min_2C, max_min_3C], axis = 1) )
table_exceedance.columns = ['PD Above', 'PD Below', '2C Above','2C Below', '3C Above','3C Below']



#############

df_features_ec_season_2C_us.mean(axis=0) - df_clim_agg_chosen.max(axis=0)

dev_2012 = (df_clim_2012_us - df_clim_agg_chosen.mean(axis=0)) / df_clim_agg_chosen.std(axis=0, ddof=0)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(df_clim_agg_chosen)
ar_scaled = scaler.transform(df_clim_agg_chosen)
ar_2C_scaled = scaler.transform(df_features_ec_season_us)
ar_3C_scaled = scaler.transform(df_features_ec_season_3C_us)

df_scaled = pd.DataFrame(ar_scaled, index=df_clim_agg_chosen.index, columns = df_clim_agg_chosen.columns)
df_2C_scaled = pd.DataFrame(ar_2C_scaled, index=df_features_ec_season_us.index, columns = df_clim_agg_chosen.columns)
df_3C_scaled = pd.DataFrame(ar_3C_scaled, index=df_features_ec_season_us.index, columns = df_clim_agg_chosen.columns)

plt.figure(figsize=(9,5), dpi=500)
fig = sns.boxplot(x='variable', y='value', data=pd.melt(df_scaled))
fig = sns.boxplot(x='variable', y='value', data=pd.melt(df_3C_scaled))


fig = sns.scatterplot(x = df_scaled.columns, y = df_scaled.loc[2012,:], data =df_scaled.loc[2012,:], 
                color = 'red', alpha = 1, s = 80, zorder=100, label = '2012 season')
plt.xlabel('')
plt.legend(bbox_to_anchor=(0.95, 0.5), loc=2, borderaxespad=0.)
fig.set_ylabel('STD units')
plt.tight_layout()



compound_occurrences_plot = table_scores_us.iloc[:2,:3] * 2000
plt.bar(x = compound_occurrences_plot.iloc[0].index, height= compound_occurrences_plot.iloc[0])
plt.bar(x = compound_occurrences_plot.iloc[0].index, height = compound_occurrences_plot.iloc[1])
plt.ylabel('Failure cases')
plt.show()


#%% EXPLORING THE 2012 ANALOGUES FOR DIFFERENT CHAIN OF EVENTS

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

DS_cli_us_tmx_det = xr.DataArray( detrend_dim(DS_cli_us['tmx'], 'time') + DS_cli_us['tmx'].mean('time'), name= DS_cli_us['tmx'].name, attrs = DS_cli_us['tmx'].attrs)
DS_cli_us_pre_det = xr.DataArray( detrend_dim(DS_cli_us['precip'], 'time') + DS_cli_us['precip'].mean('time'), name= DS_cli_us['precip'].name, attrs = DS_cli_us['precip'].attrs)
DS_cli_us_dtr_det = xr.DataArray( detrend_dim(DS_cli_us['dtr'], 'time') + DS_cli_us['dtr'].mean('time'), name= DS_cli_us['dtr'].name, attrs = DS_cli_us['dtr'].attrs)

DS_cli_us['tmx'].mean(['lat','lon']).plot()
DS_cli_us_tmx_det.mean(['lat','lon']).plot()

def detrend_dim_co2(x):
    coeff = np.polyfit(df_co2.values.ravel(), x, 1)
    trend = np.polyval(coeff, df_co2.values.ravel())
    # need to return an xr.DataArray for groupby
    df_epic_det =  pd.DataFrame( x - trend, index = x.index, columns = x.columns) + x.mean() 
    return df_epic_det

# # stack lat and lon into a single dimension called allpoints
# stacked = DS_cli_us['tmx'].stack(allpoints=['lat','lon'])
# stacked_det = detrend_dim_co2(stacked.groupby('allpoints'))

# # apply the function over allpoints to calculate the trend at each point
# trend = stacked.groupby('allpoints').apply(linear_trend)
# # unstack back to lat lon coordinates
# trend_unstacked = trend.unstack('allpoints')

# removal with a 2nd order based on the CO2 levels
# coeff = np.polyfit(df_co2.values.ravel(), df_epic.values, 1)
# trend = np.polyval(coeff, df_co2.values.ravel())

# plt.plot(df_epic.index, df_co2, 'k')
# plt.plot(df_epic.index, trend, 'r')
# plt.show()

# df_epic_det =  pd.DataFrame( df_epic['yield'] - trend, index = df_epic.index, columns = df_epic.columns) + df_epic.mean() 
# plt.plot(df_epic.index, df_epic_det, 'r-')
# plt.plot(df_epic.index, trend, 'k')
# plt.scatter(df_epic_det.loc[2012].name, df_epic_det.loc[2012].values, label = '2012 season', color = 'red')
# plt.show()

print(df_clim_agg_chosen.dtr_7_8[2012])
print(DS_cli_us['dtr'].sel(time='2012').mean(['lat','lon'])[6:8].mean().values)
print(DS_cli_us_dtr_det.sel(time='2012').mean(['lat','lon'])[6:8].mean().values)

# Plots for storylines and analogues
y_pred_2012 = brf_model_us.predict_proba(df_clim_2012_us.values.reshape(1, -1))[0][1]

def predictions(brf_model,df_features_ec_season):
        
        y_pred = brf_model.predict(df_features_ec_season)
        score_prc = sum(y_pred)/len(y_pred) 
        print("\n The total failures are:", sum(y_pred),
              " And the ratio of failure seasons by total seasons is:", score_prc, "\n")     
        probs = brf_model.predict_proba(df_features_ec_season)        
        if df_clim_2012_us is not None:
            seasons_over_2012 = df_features_ec_season[probs[:,1]>=y_pred_2012]
            print(f"\n Number of >= {y_pred_2012} probability failure events: {len(seasons_over_2012)} and mean conditions are:", 
                  np.mean(seasons_over_2012))
        
        return y_pred, score_prc, probs, seasons_over_2012

years_PD = np.unique(DS_ec_earth_PD_us.time.dt.year.values)   
df_features_ec_season_us_test = df_features_ec_season_us.copy()
df_features_ec_season_us_test.index = years_PD

years_2C = np.unique(DS_ec_earth_2C_us.time.dt.year.values)   
df_features_ec_season_2C_us_test = df_features_ec_season_2C_us.copy()
df_features_ec_season_2C_us_test.index = years_2C

years_3C = np.unique(DS_ec_earth_3C_us.time.dt.year.values)   
df_features_ec_season_3C_us_test = df_features_ec_season_3C_us.copy()
df_features_ec_season_3C_us_test.index = years_3C

y_pred_hist, score_prc_hist, probs_hist, seasons_over_2012_hist = predictions(brf_model_us, df_clim_agg_chosen)
y_pred_PD, score_prc_PD, probs_PD, seasons_over_2012_PD = predictions(brf_model_us, df_features_ec_season_us_test)
y_pred_2C, score_prc_2C, probs_2C, seasons_over_2012_2C = predictions(brf_model_us, df_features_ec_season_2C_us_test)
y_pred_3C, score_prc_3C, probs_3C, seasons_over_2012_3C = predictions(brf_model_us, df_features_ec_season_3C_us_test)

index_nonfail_hist = df_clim_agg_chosen.index.drop(seasons_over_2012_hist.index)
index_nonfail_PD = df_features_ec_season_us_test.index.drop(seasons_over_2012_PD.index)
index_nonfail_2C = df_features_ec_season_2C_us_test.index.drop(seasons_over_2012_2C.index)
index_nonfail_3C = df_features_ec_season_3C_us_test.index.drop(seasons_over_2012_3C.index)

DS_cli_us_sel = DS_cli_us.sel(time=DS_cli_us.time.dt.year.isin(seasons_over_2012_hist.index))
DS_ec_earth_PD_us_sel = DS_ec_earth_PD_us.sel(time=DS_ec_earth_PD_us.time.dt.year.isin(seasons_over_2012_PD.index))
DS_ec_earth_2C_us_sel = DS_ec_earth_2C_us.sel(time=DS_ec_earth_2C_us.time.dt.year.isin(seasons_over_2012_2C.index))
DS_ec_earth_3C_us_sel = DS_ec_earth_3C_us.sel(time=DS_ec_earth_3C_us.time.dt.year.isin(seasons_over_2012_3C.index))

DS_nonfail_PD = DS_ec_earth_PD_us.sel(time=DS_ec_earth_PD_us.time.dt.year.isin(index_nonfail_PD))
DS_nonfail_2C = DS_ec_earth_2C_us.sel(time=DS_ec_earth_2C_us.time.dt.year.isin(index_nonfail_2C))
DS_nonfail_3C = DS_ec_earth_3C_us.sel(time=DS_ec_earth_3C_us.time.dt.year.isin(index_nonfail_3C))

df_hist_test = DS_cli_us_sel.mean(['lat','lon']).to_dataframe() #.groupby('time.month').mean(dim='time')
df_hist_test['year'] = df_hist_test.index.year
df_hist_test.index = df_hist_test.index.month

df_PD_test = DS_ec_earth_PD_us_sel.mean(['lat','lon']).to_dataframe() #.groupby('time.month').mean(dim='time')
df_PD_test['year'] = df_PD_test.index.year
df_PD_test.index = df_PD_test.index.month

df_2C_test = DS_ec_earth_2C_us_sel.mean(['lat','lon']).to_dataframe() #.groupby('time.month').mean(dim='time')
df_2C_test['year'] = df_2C_test.index.year
df_2C_test.index = df_2C_test.index.month

df_3C_test = DS_ec_earth_3C_us_sel.mean(['lat','lon']).to_dataframe() #.groupby('time.month').mean(dim='time')
df_3C_test['year'] = df_3C_test.index.year
df_3C_test.index = df_3C_test.index.month


df_PD_nonfail = DS_nonfail_PD.mean(['lat','lon']).to_dataframe() #.groupby('time.month').mean(dim='time')
df_PD_nonfail['year'] = df_PD_nonfail.index.year
df_PD_nonfail.index = df_PD_nonfail.index.month

df_2C_nonfail = DS_nonfail_2C.mean(['lat','lon']).to_dataframe() #.groupby('time.month').mean(dim='time')
df_2C_nonfail['year'] = df_2C_nonfail.index.year
df_2C_nonfail.index = df_2C_nonfail.index.month

df_3C_nonfail = DS_nonfail_3C.mean(['lat','lon']).to_dataframe() #.groupby('time.month').mean(dim='time')
df_3C_nonfail['year'] = df_3C_nonfail.index.year
df_3C_nonfail.index = df_3C_nonfail.index.month

# MAIN PLOTS COMPARING 2012 ANALOGUES TO SCENARIOS
ticks = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(9,4.5), dpi=500)
plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
# plt.plot(df_3C_test.index.unique(), DS_cli_us['tmx'].sel(time=(DS_cli_us.time.dt.year != 2012)).groupby('time.month').mean('time').mean(['lat','lon']), label = 'Historical', linestyle='dotted', color = 'black')
sns.lineplot(data = df_PD_nonfail, y = 'tmx', x = df_PD_nonfail.index, linestyle='dotted',label = 'PD climatology',color = 'black')
sns.lineplot(data = df_2C_nonfail, y = 'tmx', x = df_2C_nonfail.index, linestyle='dotted',label = '2C climatology',color = '#fdbb84')
sns.lineplot(data = df_3C_nonfail, y = 'tmx', x = df_3C_nonfail.index, linestyle='dotted',label = '3C climatology',color = '#e34a33')
plt.plot(df_3C_test.index.unique(), DS_cli_us['tmx'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
sns.lineplot(data = df_hist_test, y = 'tmx', x = df_hist_test.index, label = 'historical analogues',  linestyle='dashed',color = 'black')
sns.lineplot(data = df_PD_test, y = 'tmx', x = df_PD_test.index, label = 'PD analogues', color = 'black')
sns.lineplot(data = df_2C_test, y = 'tmx', x = df_2C_test.index, label = '2C analogues', color = '#fdbb84')
sns.lineplot(data = df_3C_test, y = 'tmx', x = df_3C_test.index, label = '3C analogues', color = '#e34a33')
plt.ylabel('Temperature (°C)')
plt.legend(loc="upper left",  frameon=False)
plt.xticks(df_3C_test.index.unique(), ticks)
plt.title("a) Maximum monthly temperature", loc='left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,4.5), dpi=500)
plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
# plt.plot(df_3C_test.index.unique(), DS_cli_us['precip'].sel(time=(DS_cli_us.time.dt.year != 2012)).groupby('time.month').mean('time').mean(['lat','lon']), label = 'Historical', linestyle='dotted', color = 'black')
sns.lineplot(data = df_PD_nonfail, y = 'pre', x = df_PD_nonfail.index, linestyle='dotted',label = 'PD climatology',color = 'black')
sns.lineplot(data = df_2C_nonfail, y = 'pre', x = df_2C_nonfail.index, linestyle='dotted',label = '2C climatology',color = '#fdbb84')
sns.lineplot(data = df_3C_nonfail, y = 'pre', x = df_3C_nonfail.index, linestyle='dotted',label = '3C climatology',color = '#e34a33')
# plt.plot(df_3C_test.index.unique(), DS_cli_us['precip'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
# sns.lineplot(data = df_hist_test, y = 'precip', x = df_hist_test.index, label = 'historical analogues',  linestyle='dashed',color = 'black')
sns.lineplot(data = df_PD_test, y = 'pre', x = df_PD_test.index, label = 'PD analogues', color = 'black')
sns.lineplot(data = df_2C_test, y = 'pre', x = df_2C_test.index, label = '2C analogues', color = '#fdbb84')
sns.lineplot(data = df_3C_test, y = 'pre', x = df_3C_test.index, label = '3C analogues', color = '#e34a33')
plt.ylabel('Precipitation (mm/month)')
plt.legend(loc="upper left",  frameon=False)
plt.xticks(df_3C_test.index.unique(), ticks)
plt.title("b) Precipitation", loc='left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,4.5), dpi=500)
plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
# plt.plot(df_3C_test.index.unique(), DS_cli_us['dtr'].sel(time=(DS_cli_us.time.dt.year != 2012)).groupby('time.month').mean('time').mean(['lat','lon']), label = 'Historical', linestyle='dotted', color = 'black')
sns.lineplot(data = df_PD_nonfail, y = 'dtr', x = df_PD_nonfail.index, linestyle='dotted',label = 'PD climatology',color = 'black')
sns.lineplot(data = df_2C_nonfail, y = 'dtr', x = df_2C_nonfail.index, linestyle='dotted',label = '2C climatology',color = '#fdbb84')
sns.lineplot(data = df_3C_nonfail, y = 'dtr', x = df_3C_nonfail.index, linestyle='dotted',label = '3C climatology',color = '#e34a33')
# plt.plot(df_3C_test.index.unique(), DS_cli_us['dtr'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
sns.lineplot(data = df_PD_test, y = 'dtr', x = df_PD_test.index, label = 'PD analogues', color = 'black')
sns.lineplot(data = df_2C_test, y = 'dtr', x = df_2C_test.index, label = '2C analogues', color = '#fdbb84')
sns.lineplot(data = df_3C_test, y = 'dtr', x = df_3C_test.index, label = '3C analogues', color = '#e34a33')
plt.ylabel('Temperature (°C)')
plt.legend(loc="upper left",  frameon=False)
plt.xticks(df_3C_test.index.unique(), ticks)
plt.title("c) Diurnal temperature range", loc='left')
plt.tight_layout()
plt.show()

sns.lineplot(data = df_2C_nonfail, y = 'tmx', x = df_2C_nonfail.index,legend = False, units = 'year', estimator = None, label = '2C analogues', color = '#fdbb84', linestyle = 'dotted')
sns.lineplot(data = df_3C_nonfail, y = 'tmx', x = df_3C_nonfail.index,legend = False, units = 'year', estimator = None, label = '3C analogues', color = '#e34a33', linestyle = 'dotted')

# Figure with spam of realizations
plt.figure(figsize=(9,4.5), dpi=500)
plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
sns.lineplot(data = df_3C_test, y = 'dtr', x = df_3C_test.index,legend = False, units = 'year', estimator = None, label = '3C analogues', color = '#e34a33')
sns.lineplot(data = df_3C_test, y = 'dtr', x = df_3C_test.index,legend = False, units = 'year', estimator = None, label = '3C analogues', color = '#e34a33')
sns.lineplot(data = df_hist_test, y = 'dtr', x = df_hist_test.index,legend = False, units = 'year', estimator = None, label = 'historical analogues', linestyle = 'dashed', color = 'black',)
plt.plot(df_3C_test.index.unique(), DS_cli_us['dtr'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
plt.xticks(df_3C_test.index.unique(), ticks)
# plt.legend(loc="upper left",  frameon=False)
plt.show()

# Figure with spam of realizations
def plots_ensembles_analogues(dataframe_fail, dataframe_nonfail, scen = '3C'):
    plt.figure(figsize=(9, 4.5), dpi=500)
    plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3,zorder=1)
    # plt.fill_betweenx([df_3C_test['tmx'].max(), df_3C_test['tmx'].min()], 6.5, 8.5, color='gray',alpha=0.3,linewidth=0)
    plt.fill_between(dataframe_fail.index.unique(), dataframe_nonfail[['tmx','year']].pivot(columns='year').max(axis=1), dataframe_nonfail[['tmx','year']].pivot(columns='year').min(axis=1), label = scen+' climatology', color='orange',alpha=0.9,zorder=2,linewidth=0)
    plt.fill_between(dataframe_fail.index.unique(), dataframe_fail[['tmx','year']].pivot(columns='year').max(axis=1), dataframe_fail[['tmx','year']].pivot(columns='year').min(axis=1), label = scen+' analogues', color='#e34a33',alpha=0.9,zorder=2,linewidth=0)
    plt.plot(dataframe_fail.index.unique(), DS_cli_us['tmx'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
    plt.ylabel('Temperature (°C)')
    plt.legend(loc="upper left",  frameon=False)
    plt.xticks(df_3C_test.index.unique(), ticks)
    plt.title("a) Maximum monthly temperature", loc='left')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(9, 4.5), dpi=500)
    plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
    # plt.fill_betweenx([df_3C_test['tmx'].max(), df_3C_test['tmx'].min()], 6.5, 8.5, color='gray',alpha=0.3,linewidth=0)
    plt.fill_between(dataframe_fail.index.unique(), dataframe_nonfail[['pre','year']].pivot(columns='year').min(axis=1), dataframe_nonfail[['pre','year']].pivot(columns='year').max(axis=1), label = scen+' climatology', color='orange',alpha=0.9,zorder=2,linewidth=0)
    plt.fill_between(dataframe_fail.index.unique(), dataframe_fail[['pre','year']].pivot(columns='year').min(axis=1), dataframe_fail[['pre','year']].pivot(columns='year').max(axis=1), label = scen+' analogues', color='#e34a33',alpha=0.9,zorder=2,linewidth=0)
    plt.plot(dataframe_fail.index.unique(), DS_cli_us['precip'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
    plt.ylabel('Precipitation (mm/month)')
    plt.legend(loc="upper left",  frameon=False)
    plt.xticks(df_3C_test.index.unique(), ticks)
    plt.title("b) Precipitation", loc='left')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(9, 4.5), dpi=500)
    plt.axvspan(6.5, 8.5, facecolor='0.2', alpha=0.3)
    # plt.fill_betweenx([df_3C_test['dtr'].max(), df_3C_test['dtr'].min()], 6.5, 8.5, color='gray',alpha=0.3,linewidth=0)
    plt.fill_between(dataframe_fail.index.unique(), dataframe_nonfail[['dtr','year']].pivot(columns='year').min(axis=1), dataframe_nonfail[['dtr','year']].pivot(columns='year').max(axis=1), label = scen+' climatology', color='orange',alpha=0.9,zorder=2,linewidth=0)
    plt.fill_between(dataframe_fail.index.unique(), dataframe_fail[['dtr','year']].pivot(columns='year').min(axis=1), dataframe_fail[['dtr','year']].pivot(columns='year').max(axis=1), label = scen+' analogues', color='#e34a33',alpha=0.9,zorder=2,linewidth=0)
    plt.plot(dataframe_fail.index.unique(), DS_cli_us['dtr'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
    plt.ylabel('Temperature (°C)')
    plt.legend(loc="upper left",  frameon=False)
    plt.xticks(df_3C_test.index.unique(), ticks)
    plt.title("c) Diurnal temperature range", loc='left')
    plt.tight_layout()
    plt.show()

plots_ensembles_analogues(df_3C_test, df_3C_nonfail)
plots_ensembles_analogues(df_2C_test, df_2C_nonfail, '2C')
plots_ensembles_analogues(df_PD_test, df_PD_nonfail, 'PD')

##### INDIVIDUAL SCENARIOS

max_tmx= seasons_over_2012_2C.tmx_7_8.idxmax()
min_tmx= seasons_over_2012_2C.tmx_7_8.idxmin()
max_dtr =seasons_over_2012_2C.dtr_7_8.idxmax()
min_dtr =seasons_over_2012_2C.dtr_7_8.idxmin()
max_pre =seasons_over_2012_2C.precip_7_8.idxmax()
min_pre =seasons_over_2012_2C.precip_7_8.idxmin()

plt.plot(df_3C_test.index.unique(), DS_cli_us['tmx'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['tmx'].sel(time=str(max_tmx)).mean(['lat','lon']), label = 'Highest temperature', linestyle = 'dashed')
plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['tmx'].sel(time=str(min_tmx)).mean(['lat','lon']), label = 'Lowest temperature', linestyle = 'dashed')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['tmx'].sel(time=str(max_dtr)).mean(['lat','lon']), label = 'Highest DTR', linestyle = 'dotted')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['tmx'].sel(time=str(min_dtr)).mean(['lat','lon']), label = 'Lowest DTR', linestyle = 'dotted')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['tmx'].sel(time=str(max_pre)).mean(['lat','lon']), label = 'Highest precipitation')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['tmx'].sel(time=str(min_pre)).mean(['lat','lon']), label = 'Lowest precipitation')
plt.ylabel('Temperature (°C)')
plt.legend(loc="upper left",  frameon=False)
plt.show()

plt.plot(df_3C_test.index.unique(), DS_cli_us['dtr'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['dtr'].sel(time=str(max_tmx)).mean(['lat','lon']), label = 'Highest temperature', linestyle = 'dashed')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['dtr'].sel(time=str(min_tmx)).mean(['lat','lon']), label = 'Lowest temperature', linestyle = 'dashed')
plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['dtr'].sel(time=str(max_dtr)).mean(['lat','lon']), label = 'Highest DTR', linestyle = 'dotted')
plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['dtr'].sel(time=str(min_dtr)).mean(['lat','lon']), label = 'Lowest DTR', linestyle = 'dotted')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['dtr'].sel(time=str(max_pre)).mean(['lat','lon']), label = 'Highest precipitation')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['dtr'].sel(time=str(min_pre)).mean(['lat','lon']), label = 'Lowest precipitation')
plt.ylabel('Diurnal temperature range (°C)')
plt.legend(loc="upper left",  frameon=False)
plt.show()

plt.plot(df_3C_test.index.unique(), DS_cli_us['precip'].sel(time='2012').mean(['lat','lon']), label = '2012 season', color = 'black')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['pre'].sel(time=str(max_tmx)).mean(['lat','lon']), label = 'Highest temperature')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['pre'].sel(time=str(min_tmx)).mean(['lat','lon']), label = 'Lowest temperature')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['pre'].sel(time=str(max_dtr)).mean(['lat','lon']), label = 'Highest DTR')
# plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['pre'].sel(time=str(min_dtr)).mean(['lat','lon']), label = 'Lowest DTR')
plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['pre'].sel(time=str(max_pre)).mean(['lat','lon']), label = 'Highest precipitation')
plt.plot(df_3C_test.index.unique(), DS_ec_earth_2C_us['pre'].sel(time=str(min_pre)).mean(['lat','lon']), label = 'Lowest precipitation')
plt.ylabel('Precipitation (mm/month)')
plt.legend(loc="upper left",  frameon=False)
plt.xticks(df_3C_test.index.unique(), ticks)
plt.show()

brf_model_us.predict_proba(df_features_ec_season_2C_us_test[df_features_ec_season_2C_us_test.index == max_pre])

