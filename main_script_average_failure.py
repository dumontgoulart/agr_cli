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

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})
#%% Crop space to US and soy states 
# Function
def states_mask(input_gdp_shp, state_names) :
    country = gpd.read_file(input_gdp_shp, crs="epsg:4326") 
    country_shapes = list(shpreader.Reader(input_gdp_shp).geometries())
    soy_states = country[country['NAME_1'].isin(state_names)]
    states_area = soy_states['geometry'].to_crs({'proj':'cea'}) 
    states_area_sum = (sum(states_area.area / 10**6))
    return soy_states, country_shapes, states_area_sum

state_names = ['Illinois','Iowa','Minnesota','Indiana','Nebraska','Ohio', 
               'South Dakota','North Dakota', 'Missouri','Arkansas']
soy_us_states, us1_shapes, us_states_area_sum = states_mask('gadm36_USA_1.shp', state_names)

state_names = ['Rio Grande do Sul','Paraná']
soy_brs_states, br1_shapes, brs_states_area_sum = states_mask('gadm36_BRA_1.shp', state_names)

state_names = ['Buenos Aires','Santa Fe', 'Córdoba'] 
soy_ar_states, ar1_shapes, ar_states_area_sum = states_mask('gadm36_ARG_1.shp', state_names)

state_names = ['Mato Grosso','Goiás']
soy_brc_states, br1_shapes, brc_states_area_sum = states_mask('gadm36_BRA_1.shp', state_names)

#%% Functions
# detrend feature
def detrend_feature_iso(dataarray_reference, dataarray_in, reference_value, NA_value, months_selected):
    dataarray_iso = dataarray_in.where(dataarray_reference > reference_value, NA_value)
    mean_cli = dataarray_iso.mean(axis=0)
    dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=dataarray_in.attrs, name = dataarray_in.name ) + mean_cli
    dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin(months_selected)) 
    return dataarray_iso_2

#convert to dataframe, reshape so every month is in a separate colum:
def reshape_data(dataarray):  #converts and reshape data
    if isinstance(dataarray, pd.DataFrame): #If already dataframe, skip the convertsion
        dataframe = dataarray.dropna(how='all')
    else:    
        dataframe = dataarray.to_dataframe().dropna(how='all')
        
    dataframe['month'] = dataframe.index.get_level_values('time').month
    dataframe['year'] = dataframe.index.get_level_values('time').year
    dataframe.set_index('month', append=True, inplace=True)
    dataframe.set_index('year', append=True, inplace=True)
    dataframe = dataframe.reorder_levels(['time', 'year','month'])
    dataframe.index = dataframe.index.droplevel('time')
    dataframe = dataframe.unstack('month')
    dataframe.columns = dataframe.columns.droplevel()
    
    return dataframe

# Comparison between true ratio of failures and predicted ratio of failures by the ML model
def failure_test(df_clim, df_yield, brf_model, df_clim_2012):
    
    from sklearn.model_selection import train_test_split
    df_severe =pd.DataFrame( np.where(df_yield < df_yield.mean() - df_yield.std(),True, False), index = df_yield.index, columns = ['severe_loss'] ).astype(int)
    
    #divide data train and test
    X_train, X_test, y_train, y_test = train_test_split(df_clim, df_severe, test_size=0.3, random_state=0)
    
    # predictions
    y_pred_test = brf_model.predict(X_test)
    y_pred_train = brf_model.predict(X_train)
    y_pred_total = brf_model.predict(df_clim)
    ratio_train = sum(y_pred_train)/len(y_pred_train) 
    ratio_test = sum(y_pred_test)/len(y_pred_test) 
    ratio_total = round(sum(y_pred_total)/len(y_pred_total) ,4)
    proof_train = sum(y_train.values)/len(y_train) 
    proof_test = sum(y_test.values)/len(y_test) 
    proof_total = round(np.sum(df_severe.values)/len(df_severe),4)
        
    # Expected conditions of failures: Mean +- STD
    exp_cond = np.mean(df_clim[y_pred_total == 1] , axis=0) - np.std( df_clim[y_pred_total == 1], axis=0)
    print(exp_cond)
    # Invert for precipitation
    exp_cond[2] = exp_cond[2] + 2* np.std( df_clim.iloc[:,2][y_pred_total == 1], axis=0)
    
    # Count Joint occurrence for AND and OR conditions
    fail_joint_and = pd.DataFrame( np.where( 
        (df_clim.iloc[:,0] >  exp_cond.iloc[0]) & 
        (df_clim.iloc[:,1] >  exp_cond.iloc[1]) &
        (df_clim.iloc[:,2] < exp_cond.iloc[2]), 1, 0), index = df_clim.index ) 

    fail_joint_or = pd.DataFrame( np.where( 
        (df_clim.iloc[:,0] >  exp_cond.iloc[0]) | 
        (df_clim.iloc[:,1] >  exp_cond.iloc[1]) |
        (df_clim.iloc[:,2] < exp_cond.iloc[2]), 1, 0) , index = df_clim.index ) 
    
    from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, matthews_corrcoef
    
    score_acc_and = accuracy_score(df_severe, fail_joint_and.iloc[:,0])
    score_acc_or = accuracy_score(df_severe, fail_joint_or.iloc[:,0])
    score_acc_RF = accuracy_score(df_severe, y_pred_total)
    
    score_pcc_and = precision_score(df_severe, fail_joint_and.iloc[:,0])
    score_pcc_or = precision_score(df_severe, fail_joint_or.iloc[:,0])
    score_pcc_RF = precision_score(df_severe, y_pred_total)
    
    score_rec_and = recall_score(df_severe, fail_joint_and.iloc[:,0])
    score_rec_or = recall_score(df_severe, fail_joint_or.iloc[:,0])
    score_rec_RF = recall_score(df_severe, y_pred_total)
    
    score_f1_and = f1_score(df_severe, fail_joint_and.iloc[:,0])
    score_f1_or = f1_score(df_severe, fail_joint_or.iloc[:,0])
    score_f1_RF = f1_score(df_severe, y_pred_total)
    
    score_mcc_and = matthews_corrcoef(df_severe, fail_joint_and.iloc[:,0])
    score_mcc_or = matthews_corrcoef(df_severe, fail_joint_or.iloc[:,0])
    score_mcc_RF = matthews_corrcoef(df_severe, y_pred_total)
   
    print("JO AND acc, pcc, rec, f1, mcc",score_acc_and,score_pcc_and,score_rec_and, score_f1_and, score_mcc_and,
          "\n JO OR acc, pcc, rec, f1, mcc",score_acc_or,score_pcc_or,score_rec_or, score_f1_or, score_mcc_or,
          "\n RF acc, pcc, rec, f1, mcc ",score_acc_RF,score_pcc_RF,score_rec_RF, score_f1_RF, score_mcc_RF)
    
    print("True ratio:",proof_total," RF predicted total ratio:",ratio_total, "RF predicted test ratio:", ratio_test, 
          "JO AND ratio:",fail_joint_and.iloc[:,0].sum()/len(df_clim), 
          "JO OR ratio:",fail_joint_or.iloc[:,0].sum()/len(df_clim), 
          "\n Difference is:",(ratio_total-proof_total)*100,"%","ratio is:",ratio_total/proof_total)

    df_performance_train = pd.DataFrame([[score_acc_and,score_pcc_and,score_rec_and, score_f1_and, score_mcc_and],
                                         [score_acc_or,score_pcc_or,score_rec_or, score_f1_or, score_mcc_or],
                                         [score_acc_RF,score_pcc_RF,score_rec_RF, score_f1_RF,score_mcc_RF]], 
                                             columns = ['acc', 'pcc', "rec", 'f1', 'mcc'], index = ['AND','OR','RF'])
    
    y_pred_2012 = brf_model.predict_proba(df_clim_2012.values.reshape(1, -1))[0][1]
    print("The confidence level for 2012 of failure is:", y_pred_2012)
    
    # Shuffle data and check failure prediction
    df_clim_permuted = df_clim.apply(np.random.RandomState(seed=1).permutation, axis=0)    
    
    # Predictions for shuffled data
    y_pred_ord = brf_model.predict(df_clim)
    y_pred_perm = brf_model.predict(df_clim_permuted)
    compoundness_obs = sum(y_pred_ord)/sum(y_pred_perm)
    print("Compoundness ratio for observed data is:", compoundness_obs)
    
    return df_performance_train, y_pred_2012, proof_total




#%% average values for time
def conversion_clim_yield(DS_y, DS_cli_us, df_co2, months_to_be_used=[7,8], water_year = False, detrend = False):
    
    # option 1 detrending yield
    df_epic = DS_y.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
    plt.plot(df_epic)
    ax = sns.regplot(x = df_epic.index, y=df_epic['yield'] ,color="g")
    plt.show()
    
    # removal with a 2nd order based on the CO2 levels
    coeff = np.polyfit(df_co2.values.ravel(), df_epic.values, 1)
    trend = np.polyval(coeff, df_co2.values.ravel())
    
    plt.plot(df_epic.index, df_co2, 'k')
    plt.plot(df_epic.index, trend, 'r')
    plt.show()
    
    df_epic_det =  pd.DataFrame( df_epic['yield'] - trend, index = df_epic.index, columns = df_epic.columns) + df_epic.mean() 
    plt.plot(df_epic.index, df_epic_det, 'r-')
    plt.plot(df_epic.index, trend, 'k')
    plt.scatter(df_epic_det.loc[2012].name, df_epic_det.loc[2012].values, label = '2012 season', color = 'red')
    plt.show()

    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax1.plot(df_epic, 'k--', label = 'Original (Not detrended)')
    ax1.plot(df_epic_det, label = 'Detrended')
    ax1.scatter(df_epic_det.loc[2012].name, df_epic_det.loc[2012].values, label = '2012 season', color = 'red')
    ax1.plot(df_epic.index, trend, 'g', label = 'Regression trend (based on CO2 levels)')
    
    ax1.set_ylabel('Yield (ton/ha)')
    # ax2.set_ylabel('CO2 concentration (ppm)') # Add if you want to set another axis
    plt.legend(loc="upper left",  frameon=False)
    plt.tight_layout()
    plt.savefig('paper_figures/soy_trends.png', format='png', dpi=500)

    plt.show()
    plt.plot(df_epic_det, label = 'Detrended method 1')

    # option 2 detrending yield
    da_epic_det = DS_y['yield'].where(DS_y['yield'] > -2, -400)
    mean_cli = da_epic_det.mean(axis=0)
    da_iso_1 =  xr.DataArray(signal.detrend(da_epic_det, axis=0), dims=da_epic_det.dims, coords=da_epic_det.coords, attrs=DS_y['yield'].attrs, name = DS_y['yield'].name ) + mean_cli
    da_iso_2 = da_iso_1.where(da_iso_1 > -100, np.nan ) 
    df_y_total_det = da_iso_2.to_dataframe().dropna(how='all')
    df_y_total_det.groupby(['time']).mean().plot()
    # Check if both methods work the same
    
    # Meteorological features considered for this case, transformation to dataframe for the select months of the season
    column_names = [i+str(j) for i in list(DS_cli_us.keys()) for j in months_to_be_used]
    
    df_features_avg_list = []
    for feature in list(DS_cli_us.keys()):        
        df_feature = DS_cli_us[feature].to_dataframe().groupby(['time']).mean()
        
        # Water year or not
        if water_year == True:
            df_feature['year'] = df_feature.index.year.where(df_feature.index.month < 10, df_feature.index.year + 1)
            df_feature['month'] = pd.DatetimeIndex(df_feature.index).month
            df_feature['day'] = pd.DatetimeIndex(df_feature.index).day
            df_feature['time'] = pd.to_datetime(df_feature.iloc[:,1:4])
            df_feature.index = df_feature['time']
            df_feature.drop(['month','day','time','year'], axis=1, inplace = True)
            df_feature_reshape = reshape_data(df_feature).loc[:,months_to_be_used]  
        elif water_year == False:
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
            
    # Remove overlapping speis
    if 'spei2_7' in df_clim_avg_features:
        df_clim_avg_features = df_clim_avg_features.drop(columns=['spei2_7'])
    if 'spei2_9' in df_clim_avg_features:
        df_clim_avg_features = df_clim_avg_features.drop(columns=['spei2_9'])
    
    # generate pdf with distribution for each region 
    sns.displot(df_epic_det, x="yield", kind="kde")
    
    return df_clim_avg_features, df_epic_det
                                                       
#%% # Violin plot showing difference between failure and non failures
def violin_plots_fail_nofail(DS_cli, df_clim_avg_features, df_epic_det):
    df_severe_test = pd.DataFrame( np.where(df_epic_det < df_epic_det.mean()-df_epic_det.std(),True, False), index = df_epic_det.index,columns = ['severe_loss'] ).astype(int)
    df_clim_avg_features_fail = df_clim_avg_features.loc[df_severe_test['severe_loss'] == 1]
    df_clim_avg_features_no_fail = df_clim_avg_features.loc[df_severe_test['severe_loss'] == 0]
    dict_attrs= dict(zip(list(DS_cli.keys()), [DS_cli[x].attrs['units'] for x in list(DS_cli.keys())]))
    
    fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_avg_features_fail.columns)/2)), figsize=(10, 6), dpi=200)
    for i, ax in enumerate(axes.flatten()):
        if i < len(df_clim_avg_features_no_fail.columns):
            mean_cli_no_fail = df_clim_avg_features_no_fail.iloc[:,i]
            sns.violinplot(y = mean_cli_no_fail.values , color="darkblue",ax=ax)
            #failures
            mean_cli_fail = df_clim_avg_features_fail.iloc[:,i]
            ax = sns.violinplot(y=mean_cli_fail.values, color="red",ax=ax)
            ax.set_ylabel(dict_attrs[mean_cli_no_fail.name[:-1] if len([x for x in mean_cli_no_fail.name[-4:] if x.isdigit()]) == 1 else mean_cli_no_fail.name[:-4]])
            plt.setp(ax.collections, alpha=.5)
            ax.set_title(f"{df_clim_avg_features_fail.columns[i]}")
        else:
            fig.delaxes(axes[-1,-1])     
    fig.suptitle('a)', y = 0.93, x = 0.05, fontsize=20)
    fig.tight_layout()
    plt.show()
    return fig


# Figure columns performance
def bar_perf_train(df_perf_train_us, rf_values):
    labels = ['Accuracy','Precision','Recall','F1', 'MCC']
    and_values =df_perf_train_us.iloc[0,:]
    or_values = df_perf_train_us.iloc[1,:]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.75  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=500)
    rects1 = ax.bar(x - width/3, and_values, width/3, label='AND', color ='tab:blue')
    rects2 = ax.bar(x, or_values, width/3, label='OR', color ='#fdbb84')
    rects3 = ax.bar(x + width/3, rf_values, width/3, label='RF', color ='tab:green')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    
    
    fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=500)
    labels = ['AND', 'OR', 'RF']
    students = [df_perf_train_us.iloc[0,4], df_perf_train_us.iloc[1,4], rf_values[4]]
    ax2.bar(labels, students, color = ['tab:blue','#fdbb84','tab:green'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_ylabel('Score')
    # ax2.set_xticks(x2)
    # ax2.set_xticklabels(['MCC'])
    # ax2.legend()
    fig2.tight_layout()
    
    return fig, fig2

# Create plots comparing climatic conditions of season and historical dataset
def deviation_2012(df_clim_agg_chosen, df_clim_2012_us):
    
    df_clim_agg_chosen.columns = ['Temperature','DTR','Precipitation']
    
    dev_2012 = (df_clim_2012_us - df_clim_agg_chosen.mean(axis=0)) / df_clim_agg_chosen.std(axis=0, ddof=0)
    print(dev_2012)
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(df_clim_agg_chosen)
    ar_scaled = scaler.transform(df_clim_agg_chosen)
    df_scaled = pd.DataFrame(ar_scaled, index=df_clim_agg_chosen.index, columns = df_clim_agg_chosen.columns)
    
    plt.figure(figsize=(7,5), dpi=500)
    fig = sns.boxplot(x='variable', y='value', data=pd.melt(df_scaled))
    fig = sns.scatterplot(x = df_scaled.columns, y = df_scaled.loc[2012,:], data =df_scaled.loc[2012,:], 
                    color = 'red', alpha = 1, s = 80, zorder=100, label = '2012 season')
    plt.xlabel('')
    plt.legend(bbox_to_anchor=(0.95, 1.15), borderaxespad=0.)
    fig.set_ylabel('STD units')
    plt.tight_layout()
    return fig

#%% CROP MODEL DATASET (yield - ton/year)
'''
Start loading and formating data
'''

start_date, end_date = '1916-12-31','2016-12-31'

# First mask for US main states producers 
## Add this to the coordinates in case they are generic: ncatted -a units,lon,o,c,"degreeE" -a units,lat,o,c,"degreeN" yield_isimip_epic_3A_2.nc
DS_ref = xr.open_dataset("ACY_gswp3-w5e5_obsclim_2015soc_default_soy_noirr.nc", decode_times=True).sel(time=slice('1900-01-01','2016-12-12'), lon=slice(-160,-10))

DS_y_base_us = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc", decode_times=False).sel( lon=slice(-160,-10))
DS_y_base_us['yield'] = DS_y_base_us['yield-soy-noirr']
DS_y_base_us = DS_y_base_us.drop(['yield-soy-noirr'])
DS_y_base_us['time'] = DS_ref.time
new_yield = DS_y_base_us['yield'].sel(time=slice(start_date, end_date)).to_dataframe().groupby(['time']).mean()
old_yield = DS_ref['yield'].sel(time=slice(start_date, end_date)).to_dataframe().groupby(['time']).mean()

plt.plot(old_yield)
plt.plot(new_yield)

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y_base_us['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

DS_y = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016_lr.nc",decode_times=False)
DS_y['time'] = DS_y_base_us.time
DS_y['yield'] = DS_y['yield-soy-noirr']
DS_y = DS_y.drop(['yield-soy-noirr'])

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

# conversion to standard form
def conversion_lat_lon(DS):
    DS.coords['lon'] = (DS.coords['lon'] + 180) % 360 - 180
    DS = DS.sortby(DS.lon)
    DS = DS.reindex(lat=DS.lat[::-1])
    DS = DS.sel(lat=slice(0,50), lon=slice(-160,-10))
    return DS
DS_y = conversion_lat_lon(DS_y)

# Mask US states
DS_y = mask_shape_border(DS_y,soy_us_states) #US_shape
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
DS_y=DS_y.sel(time=slice(start_date, end_date))

# Mask to guarantee minimum 0.5 ton/ha yield
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

DS_y_full = DS_y # Full 10 states

# Mask for MIRCA 2000 each tile >0.9 rainfed
ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed_lr.nc")
ds_mask = conversion_lat_lon(ds_mask)
ds_mask_full = ds_mask.where( DS_y_full['yield'].mean('time') > 0 )

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
ds_mask_full['soybean_rainfed'].plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

# Compare yields full states and rainfed
np.sum(DS_y_full.mean('time')>0)
np.sum(ds_mask_full['soybean_rainfed']>0.95)

DS_y = DS_y.where(ds_mask['soybean_rainfed'] > 0.9 )
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')
    
# DS_y_360format = DS_y
# DS_y_360format.coords['lon']= DS_y.coords['lon'] % 360
# DS_y_360format = DS_y_360format.reindex(lat=DS_y_360format.lat[::-1])
# DS_y.to_netcdf('epic_soy_yield_us_lr.nc')

# Test if bias adjusted data is the same
# DS_y_lr = xr.open_dataset("epic_soy_yield_us_lr.nc",decode_times=True) # mask

plt.figure(figsize=(10,5), dpi=250) #plot clusters
ax=plt.axes(projection=ccrs.LambertConformal())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-115,-67,25,50], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.6)

plt.savefig('paper_figures/us_map_crop_yield.png', format='png', dpi=250)
plt.show()


### Import CO2 levels globally
DS_co2 = xr.open_dataset("ico2_annual_1901 2016.nc",decode_times=False)

DS_co2['time'] = DS_ref.time
DS_co2 = DS_co2.sel(time=slice(start_date, end_date))
df_co2 = DS_co2.to_dataframe()


#%% load data - climate CRU
DS_tmx = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.tmx.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_tmp = xr.open_dataset("cru_ts4.04.1901.2019.tmp.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_tmn = xr.open_dataset("cru_ts4.04.1901.2019.tmn.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_dtr=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.dtr.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_frs=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.frs.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_cld=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.cld.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pre.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec = DS_prec.rename_vars({'pre': 'precip'})

DS_wet = xr.open_dataset("cru_ts4.04.1901.2019.wet.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_frs = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.frs.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_vap = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.vap.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_pet = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pet.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_spei = xr.open_dataset("spei01.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_spei2 = xr.open_dataset("spei02.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_spei2 = DS_spei2.rename_vars({'spei': 'spei2_'})
# DS_rad = xr.open_dataset("surface_radiation_1980_2012_grid.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_rad.coords['time'] = DS_spei.coords['time']

DS_wet_days = DS_wet['wet'].dt.days
DS_wet_days = DS_wet_days.rename('wet')
DS_wet_days.attrs["units"] = 'days'

DS_frs_days = DS_frs['frs'].dt.days
DS_frs_days = DS_frs_days.rename('frs')
DS_frs_days.attrs["units"] = 'days'

# Merge and mask - when using wet or frost days, add dt.days after DS['days'] ;;;;;;; DS_frs['frs'].dt.days,  
DS_cli_all = xr.merge([DS_tmx.tmx, DS_tmp.tmp, DS_tmn.tmn, DS_dtr.dtr, DS_prec.precip, DS_wet_days, DS_vap.vap, DS_pet.pet, DS_cld.cld]) 
DS_cli_all = conversion_lat_lon(DS_cli_all)
# Masked after yield data
DS_cli_all = DS_cli_all.where(DS_y['yield'].mean('time') > -0.1 )

if len(DS_cli_all.coords) >3 :
    DS_cli_all=DS_cli_all.drop('spatial_ref')

# Selected features   ------------------------------------------------------------------------
DS_cli = xr.merge([DS_tmx.tmx, DS_prec.precip, DS_dtr.dtr]) 
DS_cli = conversion_lat_lon(DS_cli)

# Masked after yield data
DS_cli_us = DS_cli.where(DS_y['yield'].mean('time') > -0.1 )

if len(DS_cli_us.coords) >3 :
    DS_cli_us=DS_cli_us.drop('spatial_ref')

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_cli_us['tmx'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()
#%% Alternative version with IIzumi data
# ds_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc")
# ds_iizumi = ds_iizumi.rename({'latitude': 'lat', 'longitude': 'lon'})
# ds_iizumi = mask_shape_border(ds_iizumi,soy_us_states) #US_shape
# ds_iizumi = ds_iizumi.dropna(dim = 'lon', how='all')
# ds_iizumi = ds_iizumi.dropna(dim = 'lat', how='all')
# DS_y=ds_iizumi
# DS_y=DS_y.sel(time=slice(start_date, end_date))

# # second mask
# DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

# # Third mask
# ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")
# DS_y = DS_y.where(ds_mask['soybean_rainfed'] >= 0.9 )
# DS_y = DS_y.dropna(dim = 'lon', how='all')
# DS_y = DS_y.dropna(dim = 'lat', how='all')
# if len(DS_y.coords) >3 :
#     DS_y=DS_y.drop('spatial_ref')
#%% WP3 climate - 2012
tmx_2012_8 = DS_cli_us['tmx'].sel(time='08-2012')
pre_2012_8 = DS_cli_us['precip'].sel(time='08-2012')

monthly_avr = DS_cli_us.groupby('time.month').mean('time').sel(month=8)
monthly_avr= monthly_avr.rename({'month': 'time'})
monthly_avr['time'] = tmx_2012_8.time
tmx_monthly_avr = monthly_avr['tmx']
pre_monthly_avr = monthly_avr['precip']

tmx_2012_8.to_netcdf('US_tmx_2012_8.nc')
pre_2012_8.to_netcdf('US_pre_2012_8.nc')
tmx_monthly_avr.to_netcdf('US_tmx_clim_8.nc')
pre_monthly_avr.to_netcdf('US_pre_clim_8.nc')
#%% MAIN BLOCK 

"""
Feature selection - time and meterological variables
"""

# Feature selection
df_clim_avg_features_all_season_us, df_epic_det_all_us = conversion_clim_yield(
    DS_y, DS_cli_all,df_co2, months_to_be_used=[5,6,7,8,9,10], detrend = True)


df_clim_features_selected_us, df_epic_det_all_us = conversion_clim_yield(
    DS_y, DS_cli_all[['tmx','precip','dtr', 'pet']],df_co2, months_to_be_used=[5,6,7,8,9,10], detrend = True)


df_clim_avg_features_all_us, df_epic_det_all_us = conversion_clim_yield(
    DS_y, DS_cli_all,df_co2, months_to_be_used=[7,8], detrend = True)

# ML feature exploration, importance and selection if needed
pearsons_cor = feature_importance_selection(df_clim_avg_features_all_season_us, df_epic_det_all_us)
import calendar
pearsons_cor['month'] =pearsons_cor.index
pearsons_cor = pearsons_cor.assign(month = lambda x: x['month'].str.extract('(\d+)'))
pearsons_cor['month'] = pd.to_numeric(pearsons_cor['month'], errors='coerce')
pearsons_cor['month']=pearsons_cor['month'].apply(lambda x: calendar.month_abbr[x])

sns.boxplot(data=pearsons_cor, x="month", y = "R2_score")
plt.ylabel("R² Score")
plt.xlabel("Months in growing season")
plt.show()

pearsons_cor_78 = feature_importance_selection(df_clim_avg_features_all_us, df_epic_det_all_us)
pearsons_cor_red = feature_importance_selection(df_clim_features_selected_us, df_epic_det_all_us)

# Main algorithm for training the ML model
brf_model_month_us = failure_probability(df_clim_avg_features_all_us, df_epic_det_all_us, show_partial_plots= True, model_choice = 'conservative')

# alternative formation to aggregate values and improve performance
df_clim_agg_features_all_us =  pd.concat([df_clim_avg_features_all_us.iloc[:,0:2].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,2:4].mean(axis=1), 
                                          df_clim_avg_features_all_us.iloc[:,4:6].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,6:8].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,8:10].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,10:12].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,12:14].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,14:16].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,16:18].mean(axis=1),
                                          df_clim_avg_features_all_us.iloc[:,18:20].mean(axis=1)],axis=1)
df_clim_agg_features_all_us.columns=['tmx_7_8', 'tmp_7_8', 'tmn_7_8', 'dtr_7_8', 'precip_7_8', 'wet_7_8', 'frs_7_8','vap_7_8', 'pet_7_8', 'cld_7_8']

# violin plots for all features
violin_us = violin_plots_fail_nofail(DS_cli_all, df_clim_agg_features_all_us, df_epic_det_all_us)
violin_us.savefig('paper_figures/violin_plots_all_us.png', format='png', dpi=300)

# ML feature exploration, importance and selection if needed
df_clim_agg_features_all_us =  pd.concat([df_clim_agg_features_all_us.iloc[:,0:6],df_clim_agg_features_all_us.iloc[:,8:10]],axis=1)
                                              
plot_corr, plot_feat_imp_all = feature_importance_selection(df_clim_agg_features_all_us, df_epic_det_all_us)
plot_corr.figure.savefig('paper_figures/plot_cor_all_us.png', format='png', dpi=300)
plot_feat_imp_all.savefig('paper_figures/feat_imp_all_us.png', format='png', dpi=300)

# Main algorithm for training the ML model
brf_model_all_us = failure_probability(df_clim_agg_features_all_us, df_epic_det_all_us, show_partial_plots= True, model_choice = 'conservative')

#%% TRAIN MODEL
"""
Model training and validation
"""
# First convert datasets to dataframes divided by month of season and show timeseries
df_clim_avg_features_us, df_epic_det_us = conversion_clim_yield(
    DS_y, DS_cli_us, df_co2, months_to_be_used=[7,8], detrend = True)

# Plot violinplots divided between failure and non-failure to illustrate association of climatic variables
violin_us = violin_plots_fail_nofail(DS_cli_us, df_clim_avg_features_us, df_epic_det_us)
violin_us.savefig('paper_figures/violin_plots_us.png', format='png', dpi=500)

# alternative formation to aggregate values and improve performance
# if len(df_clim_avg_features_us.columns) == 6:
df_clim_agg_features_us =  pd.concat([df_clim_avg_features_us.iloc[:,0:2].mean(axis=1),df_clim_avg_features_us.iloc[:,4:6].mean(axis=1),df_clim_avg_features_us.iloc[:,2:4].mean(axis=1) ], axis=1)
df_clim_agg_features_us.columns=['tmx_7_8','dtr_7_8', 'precip_7_8']

# elif len(df_clim_avg_features_us.columns) == 8:
#     df_clim_agg_features_us =  pd.concat([df_clim_avg_features_us.iloc[:,0:2].mean(axis=1),df_clim_avg_features_us.iloc[:,4:6].mean(axis=1),df_clim_avg_features_us.iloc[:,2:4].mean(axis=1), df_clim_avg_features_us.iloc[:,6:8].mean(axis=1) ], axis=1)
#     df_clim_agg_features_us.columns=['tmx_7_8','dtr_7_8', 'precip_7_8', 'pet_7_8']

# select which set of clim. variables to use & 2012 climatic conditions
df_clim_agg_chosen = df_clim_agg_features_us
df_clim_2012_us = df_clim_agg_chosen.loc[2012]

#2012 season comparing climatic conditions to historical dataset
fig_2012_deviation = deviation_2012(df_clim_agg_chosen, df_clim_2012_us)
fig_2012_deviation.figure.savefig('paper_figures/fig_2012_deviation.png', format='png', dpi=500)

# ML feature exploration, importance and selection if needed
plot_feat_imp = feature_importance_selection(df_clim_agg_chosen, df_epic_det_us)

# Save input data
df_clim_agg_chosen.to_csv('rf_tuning/df_clim.csv')
df_epic_det_us.to_csv('rf_tuning/df_yield.csv')

# Main algorithm for training the ML model
df_clim_agg_chosen.columns = ['Temperature (°C)','DTR (°C)', 'Precipitation (mm/month)']
brf_model_us, fig_dep_us, rf_scores_us = failure_probability(df_clim_agg_chosen, df_epic_det_us, show_partial_plots= True, model_choice = 'conservative')

df_clim_agg_chosen.columns = ['tmx_7_8','dtr_7_8', 'precip_7_8']

# Save model RF
with open('rf_model_us.pickle', 'wb') as f:
    pickle.dump(brf_model_us, f)

# Use SHAP algorithms to explain results
fig_shap_us, fig_shap_2012 = shap_prop(df_clim_agg_chosen, df_epic_det_us, brf_model_us)

fig_shap_us.suptitle('Shap values for general model',fontsize=14)
fig_shap_us.tight_layout()
fig_shap_us.savefig('paper_figures/shap_us.png', format='png', dpi=500)

# fig_shap_2012.suptitle('c) Shap values for general model')
fig_shap_2012.tight_layout()
fig_shap_2012.savefig('paper_figures/shap_2012_us.png', format='png', dpi=500)


# Failure ratio with true and RF predictions
df_perf_train_us, y_pred_2012_us, proof_total_us = failure_test(df_clim_agg_chosen, df_epic_det_us, brf_model_us, df_clim_2012_us)

# Figure 1 - bar performance - attention graph 
bar_perf_train_us, single_plot_us = bar_perf_train(df_perf_train_us, rf_scores_us)
bar_perf_train_us.savefig('paper_figures/bar_perf_train_us.png', format='png', dpi=300)

single_plot_us.suptitle('a) MCC score',fontsize=14)
single_plot_us.savefig('paper_figures/single_perf_train_us.png', format='png', dpi=300)

# fig_dep_us.subtitle('b) Partial dependence plot',fontsize=14)
# COMBINE FIG_DEP_US WITH GIF_SHAP_US
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), dpi=500)
# single_plot_us.plot(ax = ax1)
# ax1.set_title("a) MCC score")
# fig_dep_us.plot(ax = ax2)
# ax2.set_title("b) Partial depndece plots")
# fig.tight_layout()

#%% #####################################################################################################
##### BIAS ADJUST AND SAVE TO NEW FILES - takes a lot of time, leave it off
DS_y_us_lr = xr.open_dataset("epic_soy_yield_us_lr.nc", decode_times=True) # mask

DS_ec_earth_PD_test, DS_ec_earth_2C_test, DS_ec_earth_3C_test = bias_correction_masked(
    DS_y_us_lr, start_date = '31-12-1916', end_date = '31-12-2016', cru_detrend = True, df_features_ec_3C_season = True, save_figs = True )

DS_ec_earth_PD_test.to_netcdf("ds_ec_earth_PD_us_lr.nc")
DS_ec_earth_2C_test.to_netcdf("ds_ec_earth_2C_us_lr.nc")
DS_ec_earth_3C_test.to_netcdf("ds_ec_earth_3C_us_lr.nc")
# #####################################################################
df_features_ec_season_us, df_features_ec_season_2C_us = function_conversion(DS_ec_earth_PD_test, DS_ec_earth_2C_test, months_to_be_used=[7,8])

# #####################################################################










#%% yield model - WOFOST - South Brazil
start_date, end_date = '1980-09-30','2016-03-31' #- 1959 because data before is poor

#%% CROP MODEL DATASET (yield - ton/year)
lat_south_america = slice(-50,50)
DS_y_base_brs = xr.open_dataset("soy_ibge_yield_1981_2019_lr_2.nc",decode_times=True).sel(lat=lat_south_america, lon=slice(-160,-10))
DS_y = xr.open_dataset("soy_ibge_yield_1981_2019_lr_2.nc",decode_times=True)
DS_y['time'] = DS_y_base_brs.time

DS_y['yield'] = DS_y['Yield']
DS_y = DS_y.drop(['Yield'])
    
    
# conversion to standard form
def conversion_lat_lon(DS):
    DS.coords['lon'] = (DS.coords['lon'] + 180) % 360 - 180
    DS = DS.sortby(DS.lon)
    DS = DS.reindex(lat=DS.lat[::-1])
    DS = DS.sel(lat=lat_south_america, lon=slice(-160,-10))
    return DS
DS_y = conversion_lat_lon(DS_y)

# Mask US states
DS_y = mask_shape_border(DS_y,soy_brs_states) #US_shape
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
DS_y = DS_y.sel(time=slice(start_date, end_date))

# Mask to guarantee minimum 0.5 ton/ha yield
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-62,-40,-5,-35], ccrs.PlateCarree())
plt.show()

# Mask for MIRCA 2000 each tile >0.9 rainfed
ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed_lr.nc")
ds_mask = conversion_lat_lon(ds_mask)

DS_y = DS_y.where(ds_mask['soybean_rainfed'] > 0.9 )
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')
    
# DS_y_360format = DS_y
# DS_y_360format.coords['lon']= DS_y.coords['lon'] % 360
# DS_y_360format = DS_y_360format.reindex(lat=DS_y_360format.lat[::-1])

DS_y.to_netcdf('epic_soy_yield_brs_lr.nc')
DS_y_brs = DS_y

plt.figure(figsize=(10,5), dpi=500) #plot clusters
ax=plt.axes(projection=ccrs.LambertConformal())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-62,-40,-5,-35], ccrs.Geodetic())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.6)

plt.savefig('paper_figures/br_map_crop_yield.png', format='png', dpi=150)
plt.show()


#%% load data - climate CRU
DS_tmx = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.tmx.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_t_mn=xr.open_dataset("cru_ts4.04.1901.2019.tmn.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_dtr=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.dtr.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_frs=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.frs.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_cld=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.cld.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pre.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec = DS_prec.rename_vars({'pre': 'precip'})

DS_vap = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.vap.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_pet=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pet.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_spei = xr.open_dataset("spei01.nc",decode_times=True).sel(time=slice(start_date, end_date))

# Merge and mask - when using wet or frost days, add dt.days after DS['days'] ;;;;;;; DS_frs['frs'].dt.days,
DS_cli_all = xr.merge([DS_tmx.tmx, DS_prec.precip, DS_dtr.dtr, DS_vap.vap, DS_pet.pet, DS_cld.cld]) 
DS_cli_all = conversion_lat_lon(DS_cli_all)
# Masked after yield data
DS_cli_all_brs = DS_cli_all.where(DS_y_brs['yield'].mean('time') > -0.1 )

if len(DS_cli_all_brs.coords) >3 :
    DS_cli_all_brs=DS_cli_all_brs.drop('spatial_ref')

# Selected features    #############################################################################################
DS_cli = xr.merge([DS_tmx.tmx, DS_prec.precip, DS_dtr.dtr]) 
DS_cli = conversion_lat_lon(DS_cli)

# Masked after yield data
DS_cli_brs = DS_cli.where(DS_y_brs['yield'].mean('time') > -0.1 )

if len(DS_cli_brs.coords) >3 :
    DS_cli_brs=DS_cli_brs.drop('spatial_ref')

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_cli_brs['tmx'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

# WP3 climate
tmx_2012_8 = DS_cli_brs['tmx'].sel(time='01-2012')
pre_2012_8 = DS_cli_brs['precip'].sel(time='01-2012')

monthly_avr = DS_cli_brs.groupby('time.month').mean('time').sel(month=1)
monthly_avr= monthly_avr.rename({'month': 'time'})
monthly_avr['time'] = tmx_2012_8.time
tmx_monthly_avr = monthly_avr['tmx']
pre_monthly_avr = monthly_avr['precip']

tmx_2012_8.to_netcdf('BR_tmx_2012_1.nc')
pre_2012_8.to_netcdf('BR_pre_2012_1.nc')
tmx_monthly_avr.to_netcdf('BR_tmx_clim_1.nc')
pre_monthly_avr.to_netcdf('BR_pre_clim_1.nc')

# ####### boxplot for wp3
# plt.figure(figsize = (6,6), dpi=144)
# mean_graph = np.mean(df_epic_det.loc[df_epic_det.index != 2012 ])
# deficit = (df_epic_det - mean_graph)
# plt.bar(x=deficit.index, height = deficit.values.ravel() )
# plt.bar(x=df_epic_det.loc[df_epic_det.index == 2012 ].index, height = deficit.loc[df_epic_det.index == 2012 ].values.ravel(), color = 'salmon' )
# plt.ylabel('Yield Anomaly (ton/ha)')
# plt.title(f'Yield anomaly per year')   
# plt.tight_layout()
# plt.show()


# Import CO2 levels globally
DS_co2 = xr.open_dataset("ico2_annual_1901 2016.nc",decode_times=False)

DS_co2['time'] = DS_ref.time
DS_co2 = DS_co2.sel(time=slice(start_date, end_date))
df_co2 = DS_co2.to_dataframe()

#%% Run model -  

# First convert datasets to dataframes divided by month of season and show timeseries
df_clim_avg_features_brs, df_epic_det_brs = conversion_clim_yield(DS_y_brs, DS_cli_brs, df_co2,
                                                                  months_to_be_used=[1,2], 
                                                                  water_year = True, 
                                                                  detrend = True)


# Plot violinplots divided between failure and non-failure to illustrate association of climatic variables
violin_plots_fail_nofail(DS_cli_brs, df_clim_avg_features_brs, df_epic_det_brs)

df_clim_agg_brs = pd.concat( [df_clim_avg_features_brs.iloc[:,0:2].mean(axis=1),
                                         df_clim_avg_features_brs.iloc[:,4:6].mean(axis=1),
                                         df_clim_avg_features_brs.iloc[:,2:4].mean(axis=1) ], axis=1)
df_clim_agg_brs.columns=['tmx_1_2','dtr_1_2', 'precip_1_2']

# main function call
input_features_brs = df_clim_agg_brs
df_clim_2012_brs = input_features_brs.loc[2005]

# ML feature exploration, importance and selection if needed
feature_importance_selection(input_features_brs, df_epic_det_brs)

# Main algorithm for training the ML model
brf_model_brs, fig_dep_brs, rf_scores_brs  = failure_probability(input_features_brs, df_epic_det_brs, show_partial_plots= True, model_choice = 'conservative')

y_test_br = brf_model_brs.predict(input_features_brs)

# Shap functions to explain model
# shap_prop(input_features_brs, df_epic_det_brs, brf_model_brs)

# Test for failures wrt to real data - validation
df_perf_train_brs, y_pred_2012_brs, proof_total_brs = failure_test(input_features_brs, df_epic_det_brs, brf_model_brs, df_clim_2012_brs)

# Figure 1 - bar performance - attention graph 
bar_perf_train_brs, single_plot_brs = bar_perf_train(df_perf_train_brs, rf_scores_brs)


#%% #####################################################################################################
# BIAS ADJUST AND SAVE TO NEW FILES - takes a lot of time, leave it off
DS_y_brs_lr = xr.open_dataset("EC_earth_PD/epic_soy_yield_brs_lr.nc", decode_times=True) 

DS_ec_earth_PD_brs, DS_ec_earth_2C_brs, DS_ec_earth_3C_brs = bias_correction_masked(
    DS_y_brs_lr, start_date = '31-12-1959', end_date = '31-12-2020', cru_detrend = True, df_features_ec_3C_season = True  )
DS_ec_earth_PD_brs.to_netcdf("ds_ec_earth_pd_brs.nc")
DS_ec_earth_2C_brs.to_netcdf("ds_ec_earth_2C_brs.nc")
DS_ec_earth_3C_brs.to_netcdf("ds_ec_earth_3C_brs.nc")
###########################################################################################################








#%% yield model - WOFOST - North Argentina
start_date, end_date = '1959-09-30','2016-03-31' #- 1959 because data before is poor


#%% Yield data and masking
lat_south_america = slice(-60,50)
DS_y_base_ar = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016.nc",decode_times=False).sel(lat=lat_south_america, lon=slice(-160,-10))
DS_y_base_ar['yield'] = DS_y_base_ar['yield-soy-noirr']
DS_y_base_ar = DS_y_base_ar.drop(['yield-soy-noirr'])
DS_y_base_ar['time'] = DS_ref.time

DS_y = xr.open_dataset("epic-iiasa_gswp3-w5e5_obsclim_2015soc_default_yield-soy-noirr_global_annual_1901_2016_lr.nc",decode_times=False)
DS_y['time'] = DS_y_base_us.time
DS_y['yield'] = DS_y['yield-soy-noirr']
DS_y = DS_y.drop(['yield-soy-noirr'])
# conversion to standard form
def conversion_lat_lon(DS):
    DS.coords['lon'] = (DS.coords['lon'] + 180) % 360 - 180
    DS = DS.sortby(DS.lon)
    DS = DS.reindex(lat=DS.lat[::-1])
    DS = DS.sel(lat=lat_south_america, lon=slice(-160,-10))
    return DS

DS_y = conversion_lat_lon(DS_y)


# Mask US states
DS_y = mask_shape_border(DS_y,soy_ar_states) #US_shape
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
DS_y = DS_y.sel(time=slice(start_date, end_date))

# Mask to guarantee minimum 0.5 ton/ha yield
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(ar1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-72,-40,-5,-45], ccrs.PlateCarree())
plt.show()

# Mask for MIRCA 2000 each tile >0.9 rainfed
ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed_lr.nc")
ds_mask = conversion_lat_lon(ds_mask)

DS_y = DS_y.where(ds_mask['soybean_rainfed'] > 0.9 )
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')
    
# DS_y_360format = DS_y
# DS_y_360format.coords['lon']= DS_y.coords['lon'] % 360
# DS_y_360format = DS_y_360format.reindex(lat=DS_y_360format.lat[::-1])

DS_y.to_netcdf('epic_soy_yield_ar_lr.nc')
DS_y_ar = DS_y

#%% load data - climate CRU
DS_tmx = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.tmx.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_t_mn=xr.open_dataset("cru_ts4.04.1901.2019.tmn.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_dtr=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.dtr.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_frs=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.frs.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_cld=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.cld.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pre.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec = DS_prec.rename_vars({'pre': 'precip'})

DS_vap = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.vap.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_pet=xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pet.dat_lr.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_spei = xr.open_dataset("spei01.nc",decode_times=True).sel(time=slice(start_date, end_date))

# Merge and mask - when using wet or frost days, add dt.days after DS['days'] ;;;;;;; DS_frs['frs'].dt.days,
DS_cli_all = xr.merge([DS_tmx.tmx, DS_prec.precip, DS_dtr.dtr, DS_vap.vap, DS_pet.pet, DS_cld.cld]) 
DS_cli_all = conversion_lat_lon(DS_cli_all)
# Masked after yield data
DS_cli_all_ar = DS_cli_all.where(DS_y_ar['yield'].mean('time') > -0.1 )

if len(DS_cli_all_ar.coords) >3 :
    DS_cli_all_ar=DS_cli_all_ar.drop('spatial_ref')
    
 # Selected features    #############################################################################################
DS_cli = xr.merge([DS_tmx.tmx, DS_prec.precip, DS_dtr.dtr]) 
DS_cli = conversion_lat_lon(DS_cli)

# Masked after yield data
DS_cli_ar = DS_cli.where(DS_y_ar['yield'].mean('time') > -0.1 )

if len(DS_cli_ar.coords) >3 :
    DS_cli_ar=DS_cli_ar.drop('spatial_ref')

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_cli_ar['tmx'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(ar1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-72,-40,-5,-45], ccrs.PlateCarree())
plt.show()   
#WP3 climate

tmx_2012_8 = DS_cli_ar['tmx'].sel(time='01-2012')
pre_2012_8 = DS_cli_ar['precip'].sel(time='01-2012')

monthly_avr = DS_cli_ar.groupby('time.month').mean('time').sel(month=1)
monthly_avr= monthly_avr.rename({'month': 'time'})
monthly_avr['time'] = tmx_2012_8.time
tmx_monthly_avr = monthly_avr['tmx']
pre_monthly_avr = monthly_avr['precip']

tmx_2012_8.to_netcdf('AR_tmx_2012_1.nc')
pre_2012_8.to_netcdf('AR_pre_2012_1.nc')
tmx_monthly_avr.to_netcdf('AR_tmx_clim_1.nc')
pre_monthly_avr.to_netcdf('AR_pre_clim_1.nc')
    
#%% Run model 

# First convert datasets to dataframes divided by month of season and show timeseries
df_clim_avg_features_ar, df_epic_det_ar = conversion_clim_yield(DS_y_ar, DS_cli_ar, 
                                                                  months_to_be_used=[1,2,3], 
                                                                  water_year = True, 
                                                                  detrend = True)

# Plot violinplots divided between failure and non-failure to illustrate association of climatic variables
violin_plots_fail_nofail(DS_cli_ar, df_clim_avg_features_ar, df_epic_det_ar)

# # Alternative for better performance and aggergation
# df_clim_avg_features_alt = pd.concat([df_clim_avg_features.iloc[:,0:3].mean(axis=1),df_clim_avg_features.iloc[:,6:9].mean(axis=1),df_clim_avg_features.iloc[:,9:10] ], axis=1)
# df_clim_avg_features_alt.columns=['tmx1_2_3','dtr_1_2_3', 'spei2_1_2']

# No spei -> precip
df_clim_avg_features_alt_2 = pd.concat([df_clim_avg_features_ar.iloc[:,0:3].mean(axis=1),
                                        df_clim_avg_features_ar.iloc[:,6:9].mean(axis=1),
                                        df_clim_avg_features_ar.iloc[:,3:6].mean(axis=1) ], axis=1)
df_clim_avg_features_alt_2.columns=['tmx_1_2_3','dtr_1_2_3', 'precip_1_2_3']

# No spei -> precip
if len(df_clim_avg_features_ar.columns) == 6:
    df_clim_avg_features_alt_2 = pd.concat([df_clim_avg_features_ar.iloc[:,0:2].mean(axis=1),
                                            df_clim_avg_features_ar.iloc[:,4:6].mean(axis=1),
                                            df_clim_avg_features_ar.iloc[:,2:4].mean(axis=1) ], axis=1)
    df_clim_avg_features_alt_2.columns=['tmx_1_2','dtr_1_2', 'precip_1_2']
    

# main function call
input_features_ar = df_clim_avg_features_alt_2
df_clim_2012_ar = input_features_ar.loc[2009]

# ML feature exploration, importance and selection if needed
feature_importance_selection(input_features_ar, df_epic_det_ar)

# Main algorithm for training the ML model
brf_model_ar, fig_ar = failure_probability(input_features_ar, df_epic_det_ar, show_partial_plots= True, model_choice = "conservative")

# Use SHAP algorithms to explain results
shap_prop(input_features_ar, df_epic_det_ar, brf_model_ar)

# Test for failures wrt to real data
df_perf_train_ar, y_pred_2012_ar, compoundness_obs_ar = failure_test(input_features_ar, df_epic_det_ar, brf_model_ar, df_clim_2012_ar)


#%% #####################################################################################################
# BIAS ADJUST AND SAVE TO NEW FILES - takes a lot of time, leave it off
DS_y_ar_lr = xr.open_dataset("EC_earth_PD/epic_soy_yield_ar_lr.nc",decode_times=True) # mask

DS_ec_earth_PD_ar, DS_ec_earth_2C_ar, DS_ec_earth_3C_ar = bias_correction_masked(DS_y_ar_lr, start_date = '31-12-1959', end_date = '31-12-2020', cru_detrend = True, df_features_ec_3C_season = True  )
DS_ec_earth_PD_ar.to_netcdf("ds_ec_earth_pd_ar.nc")
DS_ec_earth_2C_ar.to_netcdf("ds_ec_earth_2C_ar.nc")
DS_ec_earth_3C_ar.to_netcdf("ds_ec_earth_3C_ar.nc")
#################################################################################






#%% yield model - WOFOST - Brazil - central
start_date, end_date = '1959-09-30','2016-04-30'

DS_y_base_brc = xr.open_dataset("soy_ibge_yield_1981_2019_lr.nc",decode_times=True).sel(lat=lat_south_america, lon=slice(-160,-10))
DS_y = xr.open_dataset("soy_ibge_yield_1981_2019_lr.nc",decode_times=True)
DS_y['time'] = DS_y_base_brc.time
# conversion to standard form
def conversion_lat_lon(DS):
    DS.coords['lon'] = (DS.coords['lon'] + 180) % 360 - 180
    DS = DS.sortby(DS.lon)
    DS = DS.reindex(lat=DS.lat[::-1])
    DS = DS.sel(lat=lat_south_america, lon=slice(-160,-10))
    return DS

DS_y = conversion_lat_lon(DS_y)


# Mask US states
DS_y = mask_shape_border(DS_y,soy_brc_states) #US_shape
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
DS_y = DS_y.sel(time=slice(start_date, end_date))

# Mask to guarantee minimum 0.5 ton/ha yield
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

plt.figure(figsize=(12,5)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-72,-40,-5,-45], ccrs.PlateCarree())
plt.show()

# Mask for MIRCA 2000 each tile >0.9 rainfed
ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed_lr.nc")
ds_mask = conversion_lat_lon(ds_mask)

DS_y = DS_y.where(ds_mask['soybean_rainfed'] > 0.9 )
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')
    
# DS_y_360format = DS_y
# DS_y_360format.coords['lon']= DS_y.coords['lon'] % 360
# DS_y_360format = DS_y_360format.reindex(lat=DS_y_360format.lat[::-1])

DS_y.to_netcdf('epic_soy_yield_brc_lr.nc')
DS_y_brc = DS_y
#%% load data - climate CRU
DS_t_max = xr.open_dataset("cru_ts4.04.1901.2019.tmx.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_t_mn=xr.open_dataset("cru_ts4.04.1901.2019.tmn.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_dtr=xr.open_dataset("cru_ts4.04.1901.2019.dtr.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_frs=xr.open_dataset("cru_ts4.04.1901.2019.frs.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_cld=xr.open_dataset("cru_ts4.04.1901.2019.cld.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_prec=xr.open_dataset("cru_ts4.04.1901.2019.pre.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_vap=xr.open_dataset("cru_ts4.04.1901.2019.vap.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_pet=xr.open_dataset("cru_ts4.04.1901.2019.pet.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_spei = xr.open_dataset("spei01.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_spei2 = xr.open_dataset("spei02.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_spei2 = DS_spei2.rename_vars({'spei': 'spei2_'})
# DS_rad = xr.open_dataset("surface_radiation_1980_2012_grid.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_rad.coords['time'] = DS_spei.coords['time']

#%% Merge and mask - when using wet or frost days, add dt.days after DS['days'] ;;;;;;; DS_frs['frs'].dt.days,
DS_cli = xr.merge([DS_t_max.tmx, DS_prec.precip, DS_dtr.dtr]) 
# #mask
DS_cli_brc = DS_cli.where(DS_y_brc['yield'].mean('time') > -0.1 )
if len(DS_cli_brc.coords) >3 :
    DS_cli_brc=DS_cli_brc.drop('spatial_ref')
       
#WP3 climate

tmx_2012_8 = DS_cli_brc['tmx'].sel(time='01-2012')
pre_2012_8 = DS_cli_brc['pre'].sel(time='01-2012')

monthly_avr = DS_cli_brc.groupby('time.month').mean('time').sel(month=1)
monthly_avr= monthly_avr.rename({'month': 'time'})
monthly_avr['time'] = tmx_2012_8.time
tmx_monthly_avr = monthly_avr['tmx']
pre_monthly_avr = monthly_avr['pre']

tmx_2012_8.to_netcdf('BRC_tmx_2012_1.nc')
pre_2012_8.to_netcdf('BRC_pre_2012_1.nc')
tmx_monthly_avr.to_netcdf('BRC_tmx_clim_1.nc')
pre_monthly_avr.to_netcdf('BRC_pre_clim_1.nc')

#%% Run model
# First convert datasets to dataframes divided by month of season and show timeseries
df_clim_avg_features_brc, df_epic_det_brc = conversion_clim_yield(DS_y_brc, DS_cli_brc, 
                                                                  months_to_be_used=[12,1,2], 
                                                                  water_year = True, 
                                                                  detrend = False)

# Plot violinplots divided between failure and non-failure to illustrate association of climatic variables
violin_plots_fail_nofail(DS_cli_brc, df_clim_avg_features_brc, df_epic_det_brc)

# No spei -> precip
if len(df_clim_avg_features.columns) == 7:
    df_clim_avg_features_alt_2 = pd.concat([df_clim_avg_features.iloc[:,0:2].mean(axis=1),df_clim_avg_features.iloc[:,4:6].mean(axis=1),df_clim_avg_features.iloc[:,2:4].mean(axis=1) ], axis=1)
    df_clim_avg_features_alt_2.columns=['tmx_12_1','dtr_12_1', 'precip_12_1']
    
    # # main function call
    input_features_brc = df_clim_avg_features_alt_2
    
# No spei -> precip
elif len(df_clim_avg_features.columns) == 10:
    df_clim_avg_features_alt_3 = pd.concat([df_clim_avg_features.iloc[:,0:3].mean(axis=1),df_clim_avg_features.iloc[:,6:9].mean(axis=1),df_clim_avg_features.iloc[:,3:6].mean(axis=1) ], axis=1)
    df_clim_avg_features_alt_3.columns=['tmx_12_2','dtr_12_2', 'precip_12_2']

    # # main function call
    input_features_brc = df_clim_avg_features_alt_3
print(input_features.columns)    

# Main algorithm for training the ML model
brf_model_brc = failure_probability(input_features_brc, df_epic_det_brc, show_partial_plots= True, model_choice = 'conservative') # 


# Use SHAP algorithms to explain results
shap_prop(input_features_brc, df_epic_det_brc, brf_model_brc)

# Test for failures wrt to real data
proof_total_brc, ratio_total_brc, y_pred_2012_brc = failure_test(input_features, df_epic_det_brc, brf_model_brc)

#####################################################################################################
#%%


#%% #####################################################################################################
# # BIAS ADJUST AND SAVE TO NEW FILES - takes a lot of time, leave it off
# DS_y_lr = xr.open_dataset("EC_earth_PD/epic_soy_yield_brc_lr.nc",decode_times=True) # mask

# DS_ec_earth_PD_brc, DS_ec_earth_2C_brc = bias_correction_masked(DS_y_lr)
# DS_ec_earth_PD_brc.to_netcdf("ds_ec_earth_pd_brc1.nc")
# DS_ec_earth_2C_brc.to_netcdf("ds_ec_earth_2C_brc.nc")



# DEal with correlations and changing conditions
# df_scaled = pd.concat([df_clim_avg_features_2_alt,df_epic_det], axis=1, sort=False)
# # heatmap with the correlation of each feature + yield

# for scale in [-2,-1,0,1,1.5]:
    
#     df_severe_test = pd.DataFrame( np.where(df_epic_det < df_epic_det.mean()+scale*df_epic_det.std(),True, False), index = df_epic_det.index,columns = ['severe_loss'] ).astype(int)
#     df_clim_avg_fail = df_avg_total.loc[df_severe_test['severe_loss'] == 1]
#     df_clim_avg_no_fail = df_avg_total.loc[df_severe_test['severe_loss'] == 0]
    
#     corrmat = df_clim_avg_fail.corr()
#     top_corr_features = corrmat.index
#     plt.figure(figsize = (10,6))
#     plt.subplot(1, 2, 1)

#     g = sns.heatmap(df_clim_avg_fail[top_corr_features].corr(),annot=True, cmap="RdYlGn",vmin=-1, vmax=1)
#     plt.title(f"Failure = Mean + {scale}*STD")
    
#     plt.subplot(1, 2, 2)
#     corrmat = df_clim_avg_no_fail.corr()
#     g = sns.heatmap(df_clim_avg_no_fail[top_corr_features].corr(),annot=True, cmap="RdYlGn",vmin=-1, vmax=1)
#     plt.title("No fail correlation")
#     plt.tight_layout()
#     plt.show()

#%% Plot figures together

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_y_brs['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(br1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-70,-30,0,-35], ccrs.PlateCarree())
plt.show()    


DS_y_base_brs = mask_shape_border(DS_y_base_brs,soy_brs_states) #US_shape
DS_y_base_brc = mask_shape_border(DS_y_base_brc,soy_brc_states) #US_shape
DS_y_base_ar = mask_shape_border(DS_y_base_ar,soy_ar_states) #US_shape

DS_south_america = xr.merge([DS_y_base_brs,DS_y_base_brc, DS_y_base_ar])


plt.figure(figsize=(10,5), dpi=500) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_south_america['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-73,-33,-45,-5], ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.6)

plt.savefig('paper_figures/south_america_map.png', format='png', dpi=150)
plt.show()


#### ADDD THIS TO THE MAIN CODE

# VVVVVV 

#%% Start function

def feature_importance_selection(df_clim_input, df_yield_output):
    
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, ShuffleSplit
    from sklearn.metrics import accuracy_score
    
    # Define dataset
    X, y.values.ravel() = df_clim_input, df_yield_output
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Define models and parameters
    model = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced_subsample')
    n_estimators = [100, 500, 600, 1000]
    max_features = [3,4,5,6]
    max_depth = [6, 7, 8, 12]
    
    # Define grid search
    grid = dict(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth)
    
    # Define how it will be seaarched, being stratified to preserve the two calsses, splitting in 10 and repeating 3 random times
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)
    
    # Create model with grid and CV structure
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, error_score=0, verbose = 2)
    
    # Find hyperparameters
    grid_result = grid_search.fit(X_train, y_train)
    
    # Summarize results
    print("!! Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    
    # Test model
    grid_result.best_params_
    
    
    pred = rfc1.predict(x_test)
        
    print("Accuracy for Random Forest on CV data: ", accuracy_score(y_test, pred))
