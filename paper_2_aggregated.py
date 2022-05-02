# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:51:46 2021

@author: morenodu
"""
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')

#%% Dataframe of gridded etrended values
if len(DS_y_obs_up_clip_det.coords) >3 :
    DS_y_obs_up_clip_det=DS_y_obs_up_clip_det.drop('spatial_ref')
df_obs_det = DS_y_obs_up_clip_det.to_dataframe().dropna()
df_epic_det = DS_y_epic_br_clip_det.to_dataframe().dropna()

df_epic_grouped = df_epic.groupby('time').mean(...)

# Detrend
# Import CO2 levels globally
DS_co2 = xr.open_dataset("ico2_annual_1901 2016.nc",decode_times=False)
DS_co2['time'] = pd.date_range(start='1901', periods=DS_co2.sizes['time'], freq='YS').year

DS_co2 = DS_co2.sel(time=slice(df_epic_grouped.index.get_level_values('time')[0], df_epic_grouped.index.get_level_values('time')[-1]))
df_co2 = DS_co2.to_dataframe()

# removal with a 2nd order based on the CO2 levels
coeff = np.polyfit(df_co2.values.ravel(), df_epic_grouped['yield-soy-noirr'].values, 1)
trend = np.polyval(coeff, df_co2.values.ravel())
df_epic_grouped_det =  pd.DataFrame( df_epic_grouped['yield-soy-noirr'] - trend, index = df_epic_grouped.index, columns = df_epic_grouped.columns) + df_epic_grouped.mean() 
df_epic_grouped_det = df_epic_grouped_det['yield-soy-noirr']
plt.plot(df_epic_grouped)
plt.plot(df_epic_grouped_det)
plt.show()    

df_obs_mean_det = detrending(df_obs.groupby('time').mean(...))

plt.plot(df_obs_mean_det, label = 'Observed')
plt.plot(df_epic_grouped_det, label = 'EPIC')
plt.vlines(df_epic_grouped_det.index, 1,3.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()
# Pearson's correlation
from scipy.stats import pearsonr
# corr_res, _ = pearsonr(DS_y_obs_clip.Yield.groupby('time').mean(...).values.flatten(), df_obs.groupby('time').mean(...).values.flatten())
# print('Pearsons correlation: %.3f' % corr_res)

corr_grouped, _ = pearsonr(df_epic_grouped_det.values.flatten(), df_obs_mean_det.values.flatten())
print('Pearsons correlation: %.3f' % corr_grouped)

corr_batch, _ = pearsonr(df_epic_det.values.flatten(), df_obs_det.values.flatten())
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

DS_y_iizumi_test = DS_y_iizumi.where(DS_y_obs_up_clip['Yield'] > 0.0 )
df_iizumi = DS_y_iizumi_test.to_dataframe().dropna()
df_iizumi_mean = df_iizumi.groupby('time').mean(...)
df_iizumi_mean_det = detrending(df_iizumi_mean)

# Plot time series
plt.figure(figsize=(10,6))
plt.plot(df_epic_grouped_det, label = 'EPIC')
plt.plot(df_obs_mean_det, label = 'Obs (subset)')
plt.plot(df_fao_det, label = 'FAO')
plt.plot(df_iizumi_mean_det, label = 'Iizumi')
plt.vlines(df_epic_grouped_det.index, 1,3.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()

#%% weighted analysis

### WEIGHTED -----------------------------------------------
DS_harvest_area_globiom = xr.open_dataset('../../paper_hybrid_agri/data/soy_harvest_area_globiom_05x05_2b.nc').mean('time')
DS_harvest_area_globiom['harvest_area'] = DS_harvest_area_globiom['harvest_area'].where(DS_y_obs_up_clip_det>0)

total_area = DS_harvest_area_globiom['harvest_area'].sum(['lat','lon'])
DS_obs_weighted = ((DS_y_obs_up_clip['Yield'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'Yield')
DS_epic_orig_weighted =((DS_y_epic_br_clip['yield-soy-noirr'] * DS_harvest_area_globiom['harvest_area'] ) / total_area).to_dataset(name = 'yield-soy-noirr') 

# SUM IT ALL
df_obs_weighted_br = DS_obs_weighted['Yield'].sum(['lat','lon']).to_dataframe()
df_epic_weighted_br = DS_epic_orig_weighted['yield-soy-noirr'].sum(['lat','lon']).to_dataframe()

### -----------------------------------------------------

# Detrend

# removal with a 2nd order based on the CO2 levels
coeff = np.polyfit(df_co2.values.ravel(), df_epic_weighted_br['yield-soy-noirr'].values, 1)
trend = np.polyval(coeff, df_co2.values.ravel())
df_epic_weighted_det_br =  pd.DataFrame( df_epic_weighted_br['yield-soy-noirr'] - trend, index = df_epic_weighted_br.index, columns = df_epic_weighted_br.columns) + df_epic_weighted_br.mean() 
df_epic_weighted_br = df_epic_weighted_br['yield-soy-noirr']
plt.plot(df_epic_weighted_br)
plt.plot(df_epic_weighted_det_br)
plt.show()    

##### Convert values into weighted means
df_epic_grouped_det = df_epic_weighted_det_br
df_obs_mean_det = detrending(df_obs_weighted_br.groupby('time').mean(...))

plt.plot(df_obs_mean_det, label = 'Observed')
plt.plot(df_epic_weighted_br, label = 'EPIC')
plt.vlines(df_obs_weighted_br.index, 1,3.5, linestyles ='dashed', colors = 'k')
plt.legend()
plt.show()

# Pearson's correlation
from scipy.stats import pearsonr
# corr_res, _ = pearsonr(DS_y_obs_clip.Yield.groupby('time').mean(...).values.flatten(), df_obs.groupby('time').mean(...).values.flatten())
# print('Pearsons correlation: %.3f' % corr_res)

corr_grouped, _ = pearsonr(df_epic_grouped_det.values.flatten(), df_obs_mean_det.values.flatten())
print('Pearsons correlation: %.3f' % corr_grouped)

corr_batch, _ = pearsonr(df_epic_det.values.flatten(), df_obs_det.values.flatten())
print('Pearsons correlation: %.3f' % corr_batch)


#%% Machine learning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,  explained_variance_score

df_obs_train, df_obs_test = train_test_split(df_obs_mean_det, test_size=0.2, random_state=0)

def calibration(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # define the model
    model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)
    model.fit(X_train, y_train)
    # evaluate the model
    
      
    # Test performance
    y_pred = model.predict(X_test)
    
     # report performance
    print("R2 on test set:",round(r2_score(y_test, y_pred),2))
    print("Var score on test set:",round(explained_variance_score(y_test, y_pred),2))
    print("MAE on test set:",round(mean_absolute_error(y_test, y_pred),5))
    print("RMSE  on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
    print("______")
    y_pred_total = model.predict(X)
    
    return y_pred, y_pred_total

X, y = pd.DataFrame(df_epic_grouped_det), df_obs_mean_det.values.ravel()
print('Model 1: EPIC results:')
y_pred_epic, y_pred_total_epic = calibration(X,y)

df_pred_epic = pd.DataFrame(y_pred_epic, index = df_obs_test.index)
df_pred_epic_total = pd.DataFrame(y_pred_total_epic, index = df_epic_grouped_det.index)


plt.figure(figsize=(5,5), dpi=250) #plot clusters
plt.scatter(df_obs_mean_det['Yield'],df_pred_epic_total[0])
plt.plot(df_obs_mean_det['Yield'], df_obs_mean_det['Yield'], linestyle = '--' , color = 'black', label = '1:1 line')
plt.ylabel('RF clim predicted yield')
plt.xlabel('Observed yield')
plt.legend()
# plt.savefig('paper_figures/epic_usda_validation.png', format='png', dpi=500)
plt.show()



for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)
    regr_rf.fit(X_train, y_train)
    
    y_pred = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_pred),2))
    
    
#%% HADEX V3 - NEW TEST
# start_date, end_date = '1979-09-30','2016-03-31'

# # DS_hadex = xr.open_mfdataset('HadEX3_ref1961-1990_mon/*.nc').sel(time=slice(start_date, end_date))
# # DS_hadex = DS_hadex.drop(['latitude_bnds','longitude_bnds'])

# # # New dataset
# DS_hadex = xr.open_dataset('DS_hadex_all_hr.nc').sel(time=slice(start_date, end_date)).sel(lon=slice(-79,-30), lat=slice(0,-39))
# DS_hadex = DS_hadex.drop_vars('FD') # Always zero
# DS_hadex = DS_hadex.drop_vars('ID') # Always zero
# plot_2d_map(DS_hadex['TX90p'].mean('time'))
# plot_2d_map(DS_hadex['TX90p'].sel(time='1989-10-15'))


# def timedelta_to_int(DS, var):
#     da_timedelta = DS[var].dt.days
#     da_timedelta = da_timedelta.rename(var)
#     da_timedelta.attrs["units"] = 'days'
    
#     return da_timedelta
    

# da_list = []
# for feature in list(DS_hadex.keys()):
#     if (type(DS_hadex[feature].values[0,0,0]) == type(DS_hadex.TR.values[0,0,0])):
#         print('Time')
#         DS = timedelta_to_int(DS_hadex, feature)
#     else:
#         print('Integer')
#         DS = DS_hadex[feature]
    
#     da_list.append(DS)

# DS_hadex_combined = xr.merge(da_list)    

# # DS_hadex_combined = DS_hadex_combined.rename({'latitude': 'lat', 'longitude': 'lon'})
# DS_hadex_combined.coords['lon'] = (DS_hadex_combined.coords['lon'] + 180) % 360 - 180
# DS_hadex_combined = DS_hadex_combined.sortby(DS_hadex_combined.lon)
# DS_hadex_combined = DS_hadex_combined.reindex(lat=DS_hadex_combined.lat[::-1])
# if len(DS_hadex_combined.coords) >3 :
#     DS_hadex_combined=DS_hadex_combined.drop('spatial_ref')
    
# DS_hadex_combined_br = mask_shape_border(DS_hadex_combined, soy_brs_states)
# plot_2d_map(DS_hadex_combined_br['TX90p'].mean('time'))


# # Detrend Dataset
# def detrend_dataset(DS):
#     px= DS.polyfit(dim='time', deg=1)
#     fitx = xr.polyval(DS['time'], px)
#     dict_name = dict(zip(list(fitx.keys()), list(DS.keys())))
#     fitx = fitx.rename(dict_name)
#     DS_det  = (DS - fitx) + DS.mean('time')
#     return DS_det

# # Select data according to months
# def is_month(month, ref_in, ref_out):
#     return (month >= ref_in) & (month <= ref_out)

# DS_hadex_combined_br_det = detrend_dataset(DS_hadex_combined_br)

# # Check if it works:
# DS_hadex_combined_br.TXx.groupby('time').mean(...).plot()
# DS_hadex_combined_br_det.TXx.groupby('time').mean(...).plot()

# # Select months
# DS_hadex_combined_br_season = DS_hadex_combined_br_det.sel(time=is_month(DS_hadex_combined_br_det['time.month'], 1, 2))
# DS_hadex_combined_br_season = DS_hadex_combined_br_season.transpose("time", "lat", "lon")

# # Average across season
# DS_hadex_combined_br_season = DS_hadex_combined_br_season.groupby('time.year').mean('time')
# DS_hadex_combined_br_season = DS_hadex_combined_br_season.rename({'year': 'time'})



# DS_hadex_combined_br_season = DS_hadex_combined_br_season.where(DS_y_obs_up_clip_det > 0.0 )
# DS_y_obs_up_clip_det_test = DS_y_obs_up_clip_det.where(DS_hadex_combined_br_season['TX90p'] >= 0.0 )
# DS_y_obs_up_clip_det_test = DS_y_obs_up_clip_det_test.where(DS_hadex_combined_br_season['TX90p'].mean('time') >= 0.0 )

# test = DS_y_obs_up_clip_det_test.to_dataframe().dropna()

# df_hadex_combined_br_season = DS_hadex_combined_br_season.to_dataframe().dropna()

# X, y = df_hadex_combined_br_season.values, df_obs_det.values.ravel()

# plot_2d_map(DS_hadex_combined_br_season['DTR'].mean('time'))
# plot_2d_map(DS_y_obs_up_clip_det.mean('time'))



# DS_y_test = DS_y_obs_up_clip_det.to_dataset()
# DS_y_test = DS_y_test.rename({'Yield':'yield'})
# test = DS_hadex_combined_br['DTR'].dropna(how='all', dim='time')

# months_to_be_used=[1,2,3]
# df_clim_mon_brs, df_epic_det_brs = conversion_clim_yield(DS_y_test, DS_hadex_combined_br.dropna(how='all', dim='time'), df_co2,
#                                                                   months_to_be_used=months_to_be_used, 
#                                                                   water_year = True, 
#                                                                   detrend = True)
# df_obs_mean_det = df_obs_mean_det.where(df_clim_mon_brs['DTR1']>=-100).dropna()
# feature_importance_selection(df_clim_mon_brs, df_obs_mean_det)

# X, y = df_clim_mon_brs.values, df_obs_mean_det.values.ravel()

# y_pred_ece, y_pred_total_ece = calibration(X,y)
# df_pred_ece = pd.DataFrame(y_pred_ece, index = df_obs_test.index)
# df_pred_ece_total = pd.DataFrame(y_pred_total_ece, index = df_clim_mon_brs.index)


# df_clim_mon_brs_agg = df_clim_mon_brs.groupby(np.arange(len(df_clim_mon_brs.columns))// len(months_to_be_used), axis=1).mean()
# df_clim_mon_brs_agg.columns=['DTR','ETR', 'PRCPTOT', 'R10mm', 'R20mm','Rx1day', 'Rx5day', 'SU','TN10p', 'TN90p', 'TNn', 'TNx', 'TR', 'TX10p', 'TX90p', 'TXn', 'TXx']

# feature_importance_selection(df_clim_mon_brs_agg, df_obs_mean_det)

# X, y = df_clim_mon_brs_agg.values, df_obs_mean_det.values.ravel()

# y_pred_ece, y_pred_total_ece = calibration(X,y)
# df_pred_ece = pd.DataFrame(y_pred_ece, index = df_obs_test.index)
# df_pred_ece_total = pd.DataFrame(y_pred_total_ece, index = df_clim_mon_brs.index)



# df_clim_mon_brs_sub = df_clim_mon_brs.loc[:,['ETR1', 'ETR2', 'DTR1', 'DTR2', 'R10mm1', 'R10mm2', 'TN10p1','TN10p2']]
# df_clim_mon_brs_sub_agg = pd.concat( [df_clim_mon_brs_sub.iloc[:,0:2].mean(axis=1),
#                                           df_clim_mon_brs_sub.iloc[:,2:4].mean(axis=1),
#                                           df_clim_mon_brs_sub.iloc[:,4:6].mean(axis=1),
#                                           df_clim_mon_brs_sub.iloc[:,6:8].mean(axis=1) ], axis=1)

# df_clim_mon_brs_sub_agg.columns=['ETR','DTR', 'R10mm', ' TN10p']

# feature_importance_selection(df_clim_mon_brs_sub_agg, df_obs_mean_det)

# X, y = df_clim_mon_brs_sub_agg.values, df_obs_mean_det.values.ravel()

# y_pred_ece, y_pred_total_ece = calibration(X,y)
# df_pred_ece = pd.DataFrame(y_pred_ece, index = df_obs_test.index)
# df_pred_ece_total = pd.DataFrame(y_pred_total_ece, index = df_clim_mon_brs.index)

#%% EXTREME CLIMATE

df_clim_agg_br = df_feature_season_6mon_test.groupby(level='time').mean()


# DS_clim_weighted_br =((df_feature_season_6mon_test.to_xarray() * DS_harvest_area_globiom['harvest_area'] ) / total_area)

# SUM IT ALL
# df_clim_weighted_br = DS_clim_weighted_br.sum(['lat','lon']).to_dataframe()

#plot
df_clim_agg_br['prcptot_1'].plot()
# df_clim_weighted['prcptot_1'].plot()

df_obs_mean_det = df_obs_mean_det.where(df_clim_agg_br['prcptot_1']>-1000).dropna()

feature_importance_selection(df_clim_agg_br, df_obs_mean_det)

X, y = df_clim_agg_br, df_obs_mean_det.values.flatten().ravel()

for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)
    regr_rf.fit(X_train, y_train)
    
    y_pred = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_pred),2))
    
print('Model 1: ECE results:')

y_pred_ece, y_pred_total_ece = calibration(X,y)
df_pred_ece = pd.DataFrame(y_pred_ece, index = df_obs_test.index)
df_pred_ece_total = pd.DataFrame(y_pred_total_ece, index = df_clim_agg_br.index)


#%% HYBRID
# Define hybrid as:
df_epic_grouped_det = df_epic_grouped_det.where(df_clim_agg_br['prcptot_1']>-1000).dropna()
df_hybrid = pd.concat([df_epic_grouped_det, df_clim_agg_br], axis =1 )
X, y = df_hybrid, df_obs_mean_det.values.ravel()

for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)
    regr_rf.fit(X_train, y_train)
    
    y_pred = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_pred),2))
    
# evaluate the model
feature_importance_selection(df_hybrid, df_obs_mean_det)
print('Model 1: hybrid results:')

y_pred_hyb, y_pred_total_hyb = calibration(X,y)
df_pred_hybrid = pd.DataFrame(y_pred_hyb, index = df_obs_test.index)
df_pred_hybrid_total = pd.DataFrame(y_pred_total_hyb, index = df_hybrid.index)

# Plot time series
plt.figure(figsize=(10,6))
plt.plot(df_obs_mean_det, '--',label = 'Observed', c = 'k')
plt.plot(df_epic_grouped_det, label = 'EPIC')
plt.plot(df_pred_epic_total, label = 'RF: EPIC')
# plt.plot(df_pred_ece_total, label = 'extreme indices')
plt.plot(df_pred_hybrid_total, label = 'Hybrid', c = 'r')

plt.legend()
plt.show()


def score_list(metric, name):
    print('ALL YEARS (mixed)')
    # print(f"{name} EPIC:",round(metric(df_obs_mean_det, df_pred_epic_total),3))
    # print(f"{name} climatic Indices:",round(metric(df_obs_mean_det, df_pred_clim_total),3))
    print(f"{name} Extreme Indices:",round(metric(df_obs_mean_det, df_pred_ece_total),3))
    print(f"{name} Hybrid:",round(metric(df_obs_mean_det, df_pred_hybrid_total),3), '\n')
    
def score_list_oos(metric, name):
    print('TEST YEARS (Out of sample)')
    print(f"{name} EPIC :",round(metric(df_obs_test, df_pred_epic),3))
    # print(f"{name} climatic Indices:",round(metric(df_obs_test, df_pred_clim),3))
    print(f"{name} Extreme Indices:",round(metric(df_obs_test, df_pred_ece),3))
    print(f"{name} Hybrid:",round(metric(df_obs_test, df_pred_hybrid),3), '\n')

score_list(r2_score, 'R2')
score_list(mean_absolute_error, 'Mean absolute error')
score_list(mean_squared_error, 'Mean squared error')
    
score_list_oos(r2_score, 'R2')
score_list_oos(mean_absolute_error, 'Mean absolute error')
score_list_oos(mean_squared_error, 'Mean squared error')
    
plt.scatter(df_obs_mean_det, df_pred_hybrid_total)

m, b = np.polyfit(df_obs_mean_det.values.ravel(), df_pred_hybrid_total, 1)
plt.plot(df_obs_mean_det.values.ravel(), m*df_obs_mean_det.values.ravel() + b, label = 'Predicted regresssion line')

plt.plot(df_obs_mean_det, df_obs_mean_det, label = 'Perfect fit')

plt.ylabel('Predicted yield (ton/ha)')
plt.xlabel('Observed yield (ton/ha)')
plt.legend()
plt.show()



#%% Machine learning model training
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,  explained_variance_score
from sklearn.inspection import permutation_importance

def calibration(X,y, params = None):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # define the model
    if params is None:
        model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
        model.fit(X_train, y_train)
        
        full_model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                           max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2).fit(X, y) 

    
    if params is not None:
        model = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split'])
        model.fit(X_train, y_train)
        
        full_model = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=0, n_jobs=-1, 
                                      max_depth = params['max_depth'], max_features = params['max_features'],
                                      min_samples_leaf = params['min_samples_leaf'], min_samples_split = params['min_samples_split']).fit(X, y) 

    def MBE(y_true, y_pred):
        '''
        Parameters:
            y_true (array): Array of observed values
            y_pred (array): Array of prediction values
    
        Returns:
            mbe (float): Biais score
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
    print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
    print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
    print("MBE on test set:", round(MBE(y_test, y_pred),5))
    print("______")
    y_pred_total = model.predict(X)
    
    

    # perform permutation importance
    results = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=10, random_state=0, n_jobs=-1)
    # get importance
    df_importance = pd.DataFrame(results.importances_mean)
    df_importance.index = X.columns
    print(df_importance)
    # summarize feature importance
    plt.figure(figsize=(12,5)) #plot clusters
    plt.bar(df_importance.index, df_importance[0])
    plt.show()
    
    return y_pred, y_pred_total, model, full_model 


from sklearn.model_selection import GridSearchCV

def hyper_param_tuning(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
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
df_epic_train, df_epic_test, df_obs_train, df_obs_test = train_test_split(df_epic_grouped_det, df_obs_mean_det, test_size=0.2, random_state=0)

X, y = pd.DataFrame(df_epic_grouped_det), df_obs_mean_det.values.ravel()

for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)
    regr_rf.fit(X_train, y_train)
    
    y_pred = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_pred),2))
    
print('Model 2: EPIC results:')
y_pred_epic_agg_br, y_pred_total_epic_agg_br, model_epic_agg_br, full_model_epic_agg_br  = calibration(X, y)


# # ------------------------------------------------------------------------------------------------------------------
# # Tune hyper-parameters
# params_cv_chosen_epic_agg_br, best_grid_epic_br = hyper_param_tuning(X, y)

# # Save hyper-parameters
# with open('params_cv_chosen_epic_agg_br.pickle', 'wb') as f:
#     pickle.dump(params_cv_chosen_epic_agg_br, f)
# # ------------------------------------------------------------------------------------------------------------------

# Tunned model
with open('params_cv_chosen_epic_agg_br.pickle', 'rb') as f:
    params_cv_chosen_epic_agg_br = pickle.load(f)
    
y_pred_epic_agg_br2, y_pred_total_epic_agg_br2, model_epic_agg_br2, full_model_epic_agg_br2 = calibration(X, y, params = params_cv_chosen_epic_agg_br )


df_pred_epic = pd.DataFrame(y_pred_epic_agg_br, index = df_epic_test.index)
df_pred_epic_total = pd.DataFrame(y_pred_total_epic_agg_br, index = df_epic_grouped_det.index)



#%% CLIM 2
df_obs_mean_det = df_obs_mean_det.where(df_clim_agg_br['prcptot_1']>-1000).dropna()

feature_importance_selection(df_clim_agg_br, df_obs_mean_det)

X, y = df_clim_agg_br, df_obs_mean_det.values.flatten().ravel()

for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)
    regr_rf.fit(X_train, y_train)
    
    y_pred = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:EPIC:",round(r2_score(y_test, y_pred),2))
# Standard model
print('Model 2: ECE results:')

y_pred_clim_agg_br, y_pred_total_clim_agg_br, model_clim_agg_br, full_model_clim_agg_br  = calibration(X, y)
df_pred_ece = pd.DataFrame(y_pred_clim_agg_br, index = df_epic_test.index)
df_pred_ece_total = pd.DataFrame(y_pred_total_clim_agg_br, index = df_clim_agg_br.index)

#%% HYBRID
# Define hybrid as:
df_epic_grouped_det = df_epic_grouped_det.where(df_clim_agg_br['prcptot_1']>-1000).dropna()
df_hybrid = pd.concat([df_epic_grouped_det, df_clim_agg_br], axis =1 )
X, y = df_hybrid, df_obs_mean_det.values.ravel()

for test_size in [0.1,0.2,0.3,0.4,0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    
    regr_rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1, max_depth = 10)
    regr_rf.fit(X_train, y_train)
    
    y_pred = regr_rf.predict(X_test)
    
    print(f"R2 {test_size} OBS-RF:Hybrid:",round(r2_score(y_test, y_pred),2))
    
# evaluate the model
feature_importance_selection(df_hybrid, df_obs_mean_det)
print('Model 2: Hybrid results:')

y_pred_hyb_agg_br, y_pred_total_hyb_agg_br, model_hyb_agg_br, full_model_hyb_agg_br  = calibration(X, y)


# # ------------------------------------------------------------------------------------------------------------------
# # Tune hyper-parameters
# params_cv_chosen_hybrid_agg_br, best_grid_hybrid_br = hyper_param_tuning(X, y)

# # Save hyper-parameters
# with open('params_cv_chosen_hybrid_agg_br.pickle', 'wb') as f:
#     pickle.dump(params_cv_chosen_hybrid_agg_br, f)
# # ------------------------------------------------------------------------------------------------------------------

# Tunned model
with open('params_cv_chosen_hybrid_agg_br.pickle', 'rb') as f:
    params_cv_chosen_hybrid_agg_br = pickle.load(f)
    
y_pred_hyb_agg_br2, y_pred_total_hyb_agg_br2, model_hyb_agg_br2, full_model_hyb_agg_br2 = calibration(X, y, params = params_cv_chosen_hybrid_agg_br )


df_pred_hybrid = pd.DataFrame(y_pred_hyb_agg_br, index = df_epic_test.index)
df_pred_hybrid_total = pd.DataFrame(y_pred_total_hyb_agg_br, index = df_hybrid.index)

#%%
# Plot time series
plt.figure(figsize=(10,6))
plt.plot(df_obs_mean_det, '--',label = 'Observed', c = 'k')
plt.plot(df_epic_grouped_det, label = 'EPIC')
plt.plot(df_pred_epic_total, label = 'RF: EPIC')
# plt.plot(df_pred_ece_total, label = 'extreme indices')
plt.plot(df_pred_hybrid_total, label = 'Hybrid', c = 'r')

plt.legend()
plt.show()


def score_list(metric, name):
    print('ALL YEARS (mixed)')
    print(f"{name} EPIC:",round(metric(df_obs_mean_det, df_pred_epic_total),3))
    # print(f"{name} climatic Indices:",round(metric(df_obs_mean_det, df_pred_clim_total),3))
    print(f"{name} Extreme Indices:",round(metric(df_obs_mean_det, df_pred_ece_total),3))
    print(f"{name} Hybrid:",round(metric(df_obs_mean_det, df_pred_hybrid_total),3), '\n')
    
def score_list_oos(metric, name):
    print('TEST YEARS (Out of sample)')
    print(f"{name} EPIC :",round(metric(df_obs_test, df_pred_epic),3))
    # print(f"{name} climatic Indices:",round(metric(df_obs_test, df_pred_clim),3))
    print(f"{name} Extreme Indices:",round(metric(df_obs_test, df_pred_ece),3))
    print(f"{name} Hybrid:",round(metric(df_obs_test, df_pred_hybrid),3), '\n')

score_list(r2_score, 'R2')
score_list(mean_absolute_error, 'Mean absolute error')
score_list(mean_squared_error, 'Mean squared error')
    
score_list_oos(r2_score, 'R2')
score_list_oos(mean_absolute_error, 'Mean absolute error')
score_list_oos(mean_squared_error, 'Mean squared error')
    
plt.scatter(df_obs_mean_det, df_pred_hybrid_total)

m, b = np.polyfit(df_obs_mean_det.values.ravel(), df_pred_hybrid_total, 1)
plt.plot(df_obs_mean_det.values.ravel(), m*df_obs_mean_det.values.ravel() + b, label = 'Predicted regresssion line')

plt.plot(df_obs_mean_det, df_obs_mean_det, label = 'Perfect fit')

plt.ylabel('Predicted yield (ton/ha)')
plt.xlabel('Observed yield (ton/ha)')
plt.legend()
plt.show()
