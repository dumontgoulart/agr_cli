#%% yield model - WOFOST
start_date, end_date = '1970-12-31','2015-12-31'
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

#second mask
ds_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc")
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )
ds_iizumi = ds_iizumi.rename({'latitude': 'lat', 'longitude': 'lon'})
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

#%% Merge and mask - when using wet or frost days, add dt.days after DS['days'] ;;;;;;; DS_frs['frs'].dt.days,
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
    
    
#%% States iteration

months_to_be_used = [7,8] #  range(1,11)      
list_feature_names = list(DS_cli_us.keys())
column_names=[]
for i in list_feature_names:
    for j in months_to_be_used:
        column_names.append(i+str(j))

master_climate = np.zeros((len(state_names)), dtype=object)
master_yield= np.zeros((len(state_names)), dtype=object)    
all_states = np.zeros((len(state_names)), dtype=object)
for i in range(len(state_names)):
    soy_state = usa[usa['NAME_1'] == (state_names[i])]
    #climate mask
    DS_y_state = mask_shape_border(DS_y, soy_state) #US-shape
    DS_y_state = DS_y_state.dropna(dim = 'lon', how='all')
    DS_y_state = DS_y_state.dropna(dim = 'lat', how='all')
    if len(DS_y_state.coords) >3 :
        DS_y_state=DS_y_state.drop('spatial_ref')
    #alternative
    DS_cli_us_state = DS_cli_us.where(DS_y_state['yield'].mean('time') > -0.1 )
    if len(DS_cli_us_state.coords) >3 :
        DS_cli_us_state=DS_cli_us_state.drop('spatial_ref')
    # apply to array
    master_climate[i]=DS_y_state
    master_yield[i] = DS_cli_us_state
    
    DS_y_c = DS_y_state['yield']
    dataarray_iso = DS_y_c.where(DS_y_c > -2, -400)
    mean_cli = dataarray_iso.mean(axis=0)
    dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=DS_y_state['yield'].attrs, name = DS_y_state['yield'].name ) + mean_cli
    dataarray_iso_2 = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ) 
    df_y_c_det = dataarray_iso_2.to_dataframe().dropna(how='all')
    # df_y_c_det.groupby('time').mean().plot()
    # climate features(X)
    df_features_list = []
    for feature in list_feature_names:        
        da_det = detrend_feature_iso(DS_cli_us_state.tmx, DS_cli_us_state[feature],-300, -30000,months_to_be_used)
        df_cluster_feature = reshape_data(da_det)
        df_features_list.append(df_cluster_feature)
    
    df_clim_features = pd.concat(df_features_list, axis=1)
    df_clim_features.columns = column_names
    # aggregate it all
    all_states[i] = pd.concat([df_clim_features,df_y_c_det], axis=1, sort=False) # final table with all features + yield  


plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
master_yield[1].tmx.mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=False)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()

#%% test one cluster (this case it is number 3 - Illinois) for experiments

df_cli2 = all_states[1].iloc[:,:-1]
#remove wet and precipitation because of multicolinearity with SPEI
# df_cli2 = pd.concat([df_cli2.iloc[:,0:15],df_cli2.iloc[:,15:25]], axis=1, sort=False)
if 'spei2_7' in df_cli2:
    df_cli2 = df_cli2.drop(columns=['spei2_7'])
if 'spei2_9' in df_cli2:
    df_cli2 = df_cli2.drop(columns=['spei2_9'])
# if 'dtr7' in df_cli2:
#     df_cli2 = df_cli2.drop(columns=['dtr7'])
# if 'dtr9' in df_cli2:
#     df_cli2 = df_cli2.drop(columns=['dtr9'])
df_y_f = pd.DataFrame(all_states[1].loc[:,'yield'])
df_total = pd.concat([df_cli2,df_y_f], axis=1, sort=False)

#%% # Run analysis one cluster
lr_f1, lr_f2, lr_auc, feature_list, brf_model = failure_probability(df_cli2, df_y_f, show_scatter = False, show_partial_plots= True)
    
#%% TOTAL US STATES

df_features_list = []
for feature in list_feature_names:        
    DS_cluster_feature = DS_cli_us[feature]
    da_det = detrend_feature_iso((DS_cli_us.tmx), DS_cluster_feature,-300, -30000, months_to_be_used)
    df_cluster_feature = reshape_data(da_det)
    df_features_list.append(df_cluster_feature)
df_clim_features = pd.concat(df_features_list, axis=1)
df_clim_features.columns = column_names

if 'spatial_ref' in DS_y['yield'].coords :
    DS_y=DS_y.drop('spatial_ref')
df_y_total = DS_y['yield'].to_dataframe()
df_y_total=df_y_total.dropna(how='all')
df_y_total_det = pd.DataFrame( signal.detrend(df_y_total, axis=0), index=df_y_total.index, columns = df_y_total.columns) + df_y_total.mean()

plt.plot(DS_y.to_dataframe().groupby(['time']).mean())
plt.plot(df_y_total_det.groupby(['time']).mean())
plt.show()

df_cli_total = df_clim_features 
if 'spei2_7' in df_cli_total:
    df_cli_total = df_cli_total.drop(columns=['spei2_7'])
if 'spei2_9' in df_cli_total:
    df_cli_total = df_cli_total.drop(columns=['spei2_9'])
# if 'dtr7' in df_cli_total:
#     df_cli_total = df_cli_total.drop(columns=['dtr7','dtr9'])
    
df_total_total = pd.concat([df_cli_total,df_y_total_det], axis=1, sort=False)
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

lr_f1_total, lr_f2_total, lr_auc_total, feature_list_total, brf_model = failure_probability(df_cli_total, df_y_total_det, show_scatter = False, show_partial_plots = True)
#%% Run all states
f1_list = []
f2_list = []
pr_auc_list = []
features_per_cluster = []
for i in range(len(all_states)):
    print("\n Cluster:", i , "\n _____________________________")
    # import from clusters   
    df_cli2 = all_states[i].iloc[:,:-1]
    df_y_f = pd.DataFrame(all_states[i].loc[:,'yield'])
    df_total = pd.concat([df_cli2,df_y_f], axis=1, sort=False)

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
df_table = pd.DataFrame(table_score, index =state_names, columns=(['Optimization - Local','Overall baseline', 'Difference local - baseline']) )
pd.set_option("display.max_columns", 6)
print(df_table)

features_table = pd.DataFrame(features_per_cluster, index = state_names)
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
    
