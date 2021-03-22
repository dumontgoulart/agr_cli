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
from pylab import *

from mask_shape_border import mask_shape_border
from failure_probability import failure_probability

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

############### calendar
# mirca_2000 = xr.open_dataset("Soybeans.crop.calendar.nc") 
# # plot
# plt.figure(figsize=(20,10)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# mirca_2000['harvest'].plot(x='longitude', y='latitude',transform=ccrs.PlateCarree(), robust=True)
# ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
# plt.show()
#%% START SIMULATIONS HERE

### yield model - WOFOST
start_date, end_date = '1901-12-31','2015-12-31' #80 is 770, 70 is 0.764, 50 is 0.742, 40 is 748, 1901 = 0.722
# DS_y=xr.open_dataset("yield_soy_1979-2012.nc",decode_times=False).sel(time=slice(1,31))
# DS_y['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
# DS_y = DS_y.reindex(lat=DS_y.lat[::-1])
# DS_y = mask_shape_border(DS_y,soy_us_states) #US_shape
# DS_y = DS_y.dropna(dim = 'lon', how='all')
# DS_y = DS_y.dropna(dim = 'lat', how='all')
# DS_y_old = DS_y.sel(time=slice(start_date, end_date))

# # new version
DS_y2 = xr.open_dataset("yield_isimip_epic_3A_2.nc",decode_times=True).sel(lat=slice(0,50), lon=slice(-160,-10))
DS_y2 = mask_shape_border(DS_y2,soy_us_states) #US_shape
DS_y2 = DS_y2.dropna(dim = 'lon', how='all')
DS_y2 = DS_y2.dropna(dim = 'lat', how='all')
DS_y=DS_y2
DS_y=DS_y.sel(time=slice(start_date, end_date))

plt.plot(DS_y.to_dataframe().groupby(['time']).mean())
plt.show()

#second mask
ds_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc")
ds_iizumi = ds_iizumi.rename({'latitude': 'lat', 'longitude': 'lon'})
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )
DS_y = DS_y.where(ds_iizumi['yield'].mean('time') > 0.5 )
DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')

#%% load data - climate CRU
# DS_t_mean=xr.open_dataset("cru/cru_tmp.nc",decode_times=True).sel(time=slice('1980-01-01','2012-12-31'))
DS_t_max=xr.open_dataset("cru_ts4.04.1901.2019.tmx.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_t_mn=xr.open_dataset("cru_ts4.04.1901.2019.tmn.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_dtr=xr.open_dataset("cru_ts4.04.1901.2019.dtr.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_frs=xr.open_dataset("cru_ts4.04.1901.2019.frs.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_cld=xr.open_dataset("cru_ts4.04.1901.2019.cld.dat.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_prec=xr.open_dataset("cru/cru_pre.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_vap=xr.open_dataset("cru/cru_vap.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_wet=xr.open_dataset("cru/cru_wet.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_spei = xr.open_dataset("spei01.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_spei2 = xr.open_dataset("spei02.nc",decode_times=True).sel(time=slice(start_date, end_date))
DS_spei2 = DS_spei2.rename_vars({'spei': 'spei2_'})
# DS_rad = xr.open_dataset("surface_radiation_1980_2012_grid.nc",decode_times=True).sel(time=slice(start_date, end_date))
# DS_rad.coords['time'] = DS_spei.coords['time']

# Merge and mask - when using wet or frost days, add dt.days after DS['days'] ;;;;;;; DS_frs['frs'].dt.days,
DS_cli = xr.merge([DS_t_max.tmx,DS_dtr.dtr, DS_spei2.spei2_])
#alternative
DS_cli_us = DS_cli.where(DS_y['yield'].mean('time') > -0.1 )
if len(DS_cli_us.coords) >3 :
    DS_cli_us=DS_cli_us.drop('spatial_ref')

for feature in  list(DS_cli_us.keys()):
    print(feature)
    mean_cli = DS_cli_us[feature].to_dataframe().groupby(['time']).mean()
    mean_cli=mean_cli.loc[mean_cli.index.year != 2012]
    mean_cli_y = mean_cli.groupby(mean_cli.index.month).mean()
    stdev_cli = mean_cli.groupby(mean_cli.index.month).std()    
    #2012
    mean_cli_2012 = DS_cli_us[feature].sel(time='2012').to_dataframe().groupby(['time']).mean()
    mean_cli_2012.index = mean_cli_y.index
    # plot
    plt.plot(mean_cli_y, label=f"Mean {feature}")
    plt.fill_between(x=stdev_cli.index, y1=(mean_cli_y-stdev_cli).values.ravel(), y2=(mean_cli_y+stdev_cli).values.ravel() ,alpha=0.3, facecolor='navy', label = f"standard deviation of {feature}")
    plt.plot(mean_cli_2012, label = f"2012 {feature}")
    plt.legend()
    plt.title('Climatology against 2012')
    plt.show()
    
#%% clustering
from sklearn.cluster import AgglomerativeClustering
#crop data
da_yield_cropped = DS_y['yield'].dropna(dim = 'lon', how='all')
da_yield_cropped = da_yield_cropped.dropna(dim = 'lat', how='all')
da_yield_cropped = da_yield_cropped.fillna(1000)
#cluster it for 10 categoris + nan (1000)
hor_data = da_yield_cropped.stack(z=("lat", "lon"))
clustering = AgglomerativeClustering(linkage='average', n_clusters = 11).fit(hor_data.T)
hor_dataset = hor_data.to_dataset()
# update the values for lat/lon
hor_dataset['cluster_label'] = (('z'), clustering.labels_)
cluster_yield = hor_dataset['cluster_label'].unstack()
cluster_yield = cluster_yield.where( cluster_yield != hor_dataset['cluster_label'][-1].values)
unique_values = np.unique(cluster_yield)
unique_values = unique_values[~np.isnan(unique_values)]
cluster_names = ['Cluster '+str(i) for i in range(len(unique_values))]

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
cluster_yield.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=False, cmap = cm.get_cmap('tab20', len(unique_values)))
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()

# detrend feature
def detrend_feature_iso(dataarray_reference, dataarray_in, reference_value, NA_value, months_selected):
    range_piecewise = list(range(20, int(dataarray_in.time.dt.year[-1].values - dataarray_in.time.dt.year[0].values), 20))
    dataarray_iso = dataarray_in.where(dataarray_reference > reference_value, NA_value)
    mean_cli = dataarray_iso.mean(axis=0)
    dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0, bp=range_piecewise ), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=dataarray_in.attrs, name = dataarray_in.name ) + mean_cli
    dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ).sel(time = DS_cli_us.indexes['time'].month.isin(months_selected)) 
    return dataarray_iso_2

#convert to dataframe, reshape so every month is in a separate colum:
def reshape_data(dataarray):  #converts and reshape data
    dataframe = dataarray.to_dataframe().dropna(how='all')
    dataframe['month'] = dataframe.index.get_level_values('time').month
    dataframe['year'] = dataframe.index.get_level_values('time').year
    dataframe.set_index('month', append=True, inplace=True)
    dataframe.set_index('year', append=True, inplace=True)
    dataframe = dataframe.reorder_levels(['time', 'year','month', 'lat', 'lon'])
    dataframe.index = dataframe.index.droplevel('time')
    dataframe = dataframe.unstack('month')
    dataframe.columns = dataframe.columns.droplevel()
    return dataframe

# Features considered for this case
months_to_be_used = [6,7,8,9] #  range(1,11)

list_feature_names = list(DS_cli_us.keys())
column_names=[]
for i in list_feature_names:
    for j in months_to_be_used:
        column_names.append(i+str(j))

# calculating for each cluster the training matrix features + target:
all_clusters_partitioned = np.zeros((len(unique_values)), dtype=object)
for cluster in range(len(unique_values)): #for each cluster, define yield and climate features b*X=y
    
    # yield - target(y)
    DS_y_c = DS_y['yield'].where(cluster_yield == unique_values[cluster])
    range_piecewise = list(range(20, int(DS_y_c.time[-1].values - DS_y_c.time[0].values), 20))
    dataarray_iso = DS_y_c.where(DS_y_c > -2, -400)
    mean_cli = dataarray_iso.mean(axis=0)
    dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0, bp=range_piecewise), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=DS_y['yield'].attrs, name = DS_y['yield'].name ) + mean_cli
    dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ) 
    df_y_c_det = dataarray_iso_2.to_dataframe().dropna(how='all')
    # df_y_c_det.groupby('time').mean().plot()
    
    # climate features(X)
    df_features_list = []
    for feature in list_feature_names:        
        DS_cluster_feature = DS_cli_us[feature].where(cluster_yield == unique_values[cluster])
        da_det = detrend_feature_iso((DS_cli_us.tmx.where(cluster_yield == unique_values[cluster])), DS_cluster_feature,-300, -30000,months_to_be_used)
        df_cluster_feature = reshape_data(da_det)
        df_features_list.append(df_cluster_feature)
    
    df_clim_features = pd.concat(df_features_list, axis=1)
    df_clim_features.columns = column_names
    # aggregate it all
    all_clusters_partitioned[cluster] = pd.concat([df_clim_features,df_y_c_det], axis=1, sort=False) # final table with all features + yield
    
#%% test one cluster (this case it is number 3 - Illinois) for experiments

df_cli2 = all_clusters_partitioned[0].iloc[:,:-1]
#remove wet and precipitation because of multicolinearity with SPEI
# df_cli2 = pd.concat([df_cli2.iloc[:,0:15],df_cli2.iloc[:,15:25]], axis=1, sort=False)
# if 'spei2_7' in df_cli2:
#     df_cli2 = df_cli2.drop(columns=['spei2_7'])
if 'spei2_9' in df_cli2:
    df_cli2 = df_cli2.drop(columns=['spei2_9'])
# if 'dtr7' in df_cli2:
#     df_cli2 = df_cli2.drop(columns=['dtr7'])
# if 'dtr9' in df_cli2:
#     df_cli2 = df_cli2.drop(columns=['dtr9'])
df_y_f = pd.DataFrame(all_clusters_partitioned[0].loc[:,'yield'])
df_total = pd.concat([df_cli2,df_y_f], axis=1, sort=False)

#%% # Run analysis one cluster
lr_f1, lr_f2, lr_auc, feature_list, brf_model = failure_probability(df_cli2, df_y_f, show_scatter = False, show_partial_plots= True)

#%% The whole area together, no cluster
df_features_list = []
for feature in list_feature_names:        
    DS_cluster_feature = DS_cli_us[feature]
    da_det = detrend_feature_iso((DS_cli_us.tmx), DS_cluster_feature,-300, -30000, months_to_be_used)
    df_cluster_feature = reshape_data(da_det)
    df_features_list.append(df_cluster_feature)
df_clim_features = pd.concat(df_features_list, axis=1)
df_clim_features.columns = column_names

DS_y['yield'].to_dataframe().groupby(['time']).mean().plot()

dataarray_iso = DS_y['yield'].where(DS_y['yield'] > -2, -400)
mean_cli = dataarray_iso.mean(axis=0)
dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0, bp=range_piecewise), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=DS_y['yield'].attrs, name = DS_y['yield'].name ) + mean_cli
dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ) 
df_y_total_det = dataarray_iso_2.to_dataframe().dropna(how='all')

df_y_total_det.groupby(['time']).mean().plot()

df_cli_total = df_clim_features 
# df_cli_total = pd.concat([df_cli_total.iloc[:,5:15],df_cli_total.iloc[:,15:25]], axis=1, sort=False)
if 'spei2_7' in df_cli_total:
    df_cli_total = df_cli_total.drop(columns=['spei2_7'])
if 'spei2_9' in df_cli_total:
    df_cli_total = df_cli_total.drop(columns=['spei2_9'])
# if 'dtr7' in df_cli_total:
#     df_cli_total = df_cli_total.drop(columns=['dtr7','dtr9'])
df_total_total = pd.concat([df_cli_total,df_y_total_det], axis=1, sort=False)

df_severe_test = pd.DataFrame( np.where(df_y_total_det < df_y_total_det.mean()-df_y_total_det.std(),True, False), index = df_y_total_det.index,columns = ['severe_loss'] ).astype(int)
df_clim_avg_features_fail = df_cli_total.loc[df_severe_test['severe_loss'] == 1]
df_clim_avg_features_no_fail = df_cli_total.loc[df_severe_test['severe_loss'] == 0]

fig, axes  = plt.subplots(2,int(np.ceil(len(df_clim_avg_features_fail.columns)/2)), figsize=(10, 10), dpi=150)
for i, ax in enumerate(axes.flatten()):
    mean_cli_no_fail = df_clim_avg_features_no_fail.iloc[:,i]
    sns.violinplot(y = mean_cli_no_fail.values ,  color="darkblue",ax=ax)
    #failures
    mean_cli_fail = df_clim_avg_features_fail.iloc[:,i]
    ax = sns.violinplot(y=mean_cli_fail.values, color="red",ax=ax)
    plt.setp(ax.collections, alpha=.5)
    ax.set_title(f"{df_clim_avg_features_fail.columns[i]}")
fig.suptitle('Comparison failure (red) and no failure (blue)', fontsize=20)
fig.tight_layout()
plt.show()

### PLOTS
# plt.figure(figsize=(20,10)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# DS_cli_us.tmx.mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='tab20')
# ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
# plt.show()

# plt.figure(figsize=(20,10)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# DS_y['yield'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='tab20')
# ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
# plt.show()
#%%
#alternative summing either 7,8 for all variables, or keeping only spei2_8 as it is
df_cli_total_alt = df_cli_total.iloc[:,5:10]
df_cli_total_alt = pd.concat([df_cli_total.iloc[:,5:7].mean(axis=1),df_cli_total.iloc[:,8:9].mean(axis=1),df_cli_total.iloc[:,9:10].mean(axis=1)], axis=1)
df_cli_total_alt.columns=['tmx','dtr', 'spei']
# not good so far, change detrending scheme
lr_f1_total, lr_f2_total, lr_auc_total, feature_list_total, brf_total_model = failure_probability(df_cli_total_alt, df_y_total_det, show_scatter = False, show_partial_plots = True)

#%% Sparse 5 units
# sparse_lvl=1
# DS_y_s = DS_y['yield'][::,::sparse_lvl,::sparse_lvl]

# df_features_list = []
# for feature in list_feature_names:        
#     DS_cluster_feature = DS_cli_us[feature][::,::sparse_lvl,::sparse_lvl]
#     da_det = detrend_feature_iso((DS_cli_us.tmx[::,::sparse_lvl,::sparse_lvl]), DS_cluster_feature,-300, -30000, months_to_be_used)
#     df_cluster_feature = reshape_data(da_det)
#     df_features_list.append(df_cluster_feature)
# df_clim_features = pd.concat(df_features_list, axis=1)
# df_clim_features.columns = column_names

# df_y_total = DS_y_s.to_dataframe().dropna(how='all')
# df_y_total_det = pd.DataFrame( signal.detrend(df_y_total, axis=0), index=df_y_total.index, columns = df_y_total.columns) + df_y_total.mean()
# df_cli_total = df_clim_features 
# #remove wet and precipitation because of multicolinearity with SPEI
# # df_cli_total = pd.concat([df_cli_total.iloc[:,5:15],df_cli_total.iloc[:,15:25]], axis=1, sort=False)
# # if 'spei2_8' in df_cli_total:
# #     df_cli_total = df_cli_total.drop(columns=['spei2_7','spei2_9'])
# df_total_total = pd.concat([df_cli_total,df_y_total_det], axis=1, sort=False)

# plt.figure(figsize=(20,10)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# DS_cli_us.tmx[::,::sparse_lvl,::sparse_lvl].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='tab20')
# ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
# plt.show()

# plt.figure(figsize=(20,10)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# DS_y['yield'][::,::sparse_lvl,::sparse_lvl].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Yield kg/ha'}, cmap='tab20')
# ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
# plt.show()

# lr_f1_total, lr_f2_total, lr_auc_total, feature_list_total, brf_model = failure_probability(df_cli_total, df_y_total_det, show_scatter = False, show_partial_plots = True)


#%% multiple clusters 
f1_list = []
f2_list = []
pr_auc_list = []
features_per_cluster = []
for i in range(len(all_clusters_partitioned)):
    print("\n Cluster:", i , "\n _____________________________")
    # import from clusters   
    df_cli2 = all_clusters_partitioned[i].iloc[:,:-1]
    # df_cli2 = pd.concat([df_cli2.iloc[:,5:15],df_cli2.iloc[:,15:25]], axis=1, sort=False)  #remove wet and precipitation because of multicolinearity with SPEI
    
    df_y_f = pd.DataFrame(all_clusters_partitioned[i].loc[:,'yield'])
    df_total = pd.concat([df_cli2,df_y_f], axis=1, sort=False)
    
    # df_cli2_alt = pd.concat([df_cli2.iloc[:,0:2].mean(axis=1),df_cli2.iloc[:,2:4].mean(axis=1),df_cli2.iloc[:,4:6].mean(axis=1)], axis=1)
    # df_cli2_alt.columns=['tmx','dtr', 'spei']
    lr_f1, lr_f2, lr_auc, feature_list, brf_model = failure_probability(df_cli2, df_y_f, show_scatter = False)

    f1_list.append(lr_f1)
    f2_list.append(lr_f2)
    pr_auc_list.append(lr_auc)
    features_per_cluster.append(feature_list)
    print("______________________________")
    print("##############################")
    print("______________________________")
        
 #%% tables
sc_all_array = np.repeat(lr_f2_total, len(f2_list))
table_score = np.array([ np.array(f2_list),sc_all_array, (np.array(f2_list) - np.array(sc_all_array))]).T
df_table = pd.DataFrame(table_score, index =unique_values, columns=(['Optimization - Local','Overall baseline', 'Difference local - baseline']) )
pd.set_option("display.max_columns", 6)
print(df_table)

features_table = pd.DataFrame(features_per_cluster, index = unique_values)
print(features_table)

plt.figure(figsize=(10,6), dpi=144) 
plt.title('R2 Score at a state level')
plt.bar(x=df_table.index, height =df_table['Optimization - Local'], label='Cluster optimization')
# plt.scatter(x=df_table.index, y='Optimization - Mean', data=df_table, label='Local - Features from mean model')
plt.hlines(y=df_table['Overall baseline'], xmin=df_table.index[0], xmax=df_table.index[-1], label='Overall model', color = 'orange')
plt.ylim(0, 1.1)
plt.xlabel("Cluster index")
plt.ylabel("F2 score")
plt.legend(loc='lower right')
plt.savefig('clusters_performance.png', bbox_inches='tight')
plt.show()

features_cluster_total = features_per_cluster #+ feature_list_total
features_cluster_total.append(feature_list_total) #+ 
from collections import defaultdict
flat_list = [item for sublist in features_cluster_total for item in sublist]
d = defaultdict(list)
for k, v in flat_list:
   d[k].append(v)
d = {k: v for k, v in sorted(d.items(), key=lambda item: np.mean(item[1]))}


import seaborn as sns
sorted_keys, sorted_vals = [*zip(*d.items())] 
sns.boxplot(data=sorted_vals, width=.4, showmeans=True).set(title = "Most important features", ylabel = 'Importance', xlabel = 'Features')
sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
# category labels
plt.xticks(plt.xticks()[0], sorted_keys)
plt.savefig('features_importance.png', bbox_inches='tight')
plt.show()

#%% TESST - CC
from sklearn.model_selection import train_test_split
import seaborn as sns

df_y_f = df_y_f
df_severe =pd.DataFrame( np.where(df_y_f < df_y_f.mean() - df_y_f.std(),True, False), index = df_y_f.index, columns = ['severe_loss'] ).astype(int)

#divide data train and test
X_train, X_test, y_train, y_test = train_test_split(df_cli2, df_severe, test_size=0.3, random_state=0)
#predictions
y_pred = brf_model.predict(X_test)
n_fail_hist=(sum(y_pred))
ratio_fail_hist=(sum(y_pred)/len(y_pred))
X_test_proj = X_test.copy()

# #generate scenarios and plots
# list_failure=[]
# list_ratio=[]
# range_temp = range(1,6)
# for increase_proj in range_temp:
#     X_test_proj.loc[:,'tmx9'] = X_test.loc[:,'tmx9'].copy() + increase_proj
#     # X_test_proj.loc[:,'tmx7'] = X_test.loc[:,'tmx7'].copy() + increase_proj
#     # X_test_proj.loc[:,'tmx9'] = X_test.loc[:,'tmx9'].copy() + increase_proj
#     y_proj_pred = brf_model.predict(X_test_proj)
#     n_fail_proj=(sum(y_proj_pred))
#     increase_failure=(sum(y_proj_pred)/len(y_proj_pred))

#     list_ratio.append(increase_failure)
#     list_failure.append( (n_fail_proj-n_fail_hist)/n_fail_hist )

# plt.figure(figsize=(10,6), dpi=100)
# plt.scatter(range_temp, list_failure, c='black')
# plt.title(f"Failure events wrt history", fontsize=20)
# plt.xlabel("Celisus degrees", fontsize=16)
# plt.ylabel("Predicted failure increase", fontsize=16)

# plt.figure(figsize=(10,6), dpi=100)
# plt.scatter(range_temp, list_ratio, c='black')
# plt.title(f"Ratio of failure events by temperature increase", fontsize=20)
# plt.xlabel("Celisus degrees", fontsize=16)
# plt.ylabel("Ratio of failure per total events", fontsize=16)
# plt.axhline(np.mean(ratio_fail_hist))

perturbation_range = range(-2,4)
features_proj = X_test_proj.columns[0:8]
failure_ratio = np.zeros((len(perturbation_range), len(features_proj) ), dtype=object)
increase_failure = np.zeros((len(perturbation_range), len(features_proj) ), dtype=object)
for feature in range(len(features_proj)):
    for perturbation in range(len(perturbation_range)):
        X_test_proj = X_test.copy()
        X_test_proj.loc[:,X_test_proj.columns[feature]] = X_test.loc[:,X_test.columns[feature]].copy() + perturbation_range[perturbation]
        
        y_proj_pred = brf_model.predict(X_test_proj)
        n_fail_proj=(sum(y_proj_pred))
    
        increase_failure[perturbation,feature]  = (n_fail_proj-n_fail_hist)/n_fail_hist 
        failure_ratio[perturbation,feature] = (sum(y_proj_pred)/len(y_proj_pred))
df_ratio = pd.DataFrame(failure_ratio, columns=features_proj, index = perturbation_range )
df_increase = pd.DataFrame(increase_failure, columns=features_proj, index = perturbation_range )
# df_ratio.plot() # df_increase.plot()

df_melt = df_increase.reset_index().melt('index', var_name='Features',  value_name='Predicted failure increase')
df_melt['Predicted failure increase'] = df_melt['Predicted failure increase'].astype(float)
plt.figure(figsize = (8,8), dpi=144)
g = sns.lineplot(x="index", y="Predicted failure increase", hue='Features', data=df_melt,style="Features", markers=True, dashes = False)
g.set(xlabel='Celsius degree [SPEI] increase')
plt.show()

df_melt2 = df_ratio.reset_index().melt('index', var_name='Features',  value_name='Ratio of failure per total events')
df_melt2['Ratio of failure per total events'] = df_melt2['Ratio of failure per total events'].astype(float)
plt.figure(figsize = (8,8), dpi=144)
f = sns.lineplot(x="index", y="Ratio of failure per total events", hue='Features', data=df_melt2, style="Features", markers=True, dashes = False)
plt.axhline(np.mean(ratio_fail_hist), color = 'black')
f.set(xlabel='Celsius degree [SPEI] increase')
plt.show()




