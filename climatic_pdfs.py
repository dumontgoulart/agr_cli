import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

# Crop space to either US or soy states
usa = gpd.read_file('gadm36_USA_1.shp', crs="epsg:4326") 
us1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
state_names = ['Iowa','Illinois','Minnesota','Indiana','Nebraska','Ohio', 'South Dakota','North Dakota', 'Missouri','Arkansas']
soy_us_states = usa[usa['NAME_1'].isin( state_names)]

# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_br_names = ['Rio Grande do Sul','Paraná'] #'Rio Grande do Sul','Paraná' #'Mato Grosso','Goiás' 
soy_br_states = bra[bra['NAME_1'].isin(state_br_names)]

# Crop space to either ARG or soy states
arg = gpd.read_file('gadm36_ARG_1.shp', crs="epsg:4326") 
ar1_shapes = list(shpreader.Reader('gadm36_ARG_1.shp').geometries())
state_ar_names = ['Buenos Aires','Santa Fe', 'Córdoba'] #'Rio Grande do Sul','Paraná' #'Mato Grosso','Goiás' 
soy_ar_states = arg[arg['NAME_1'].isin(state_ar_names)]

# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_brc_names = ['Mato Grosso','Goiás'] #'Rio Grande do Sul','Paraná' #'Mato Grosso','Goiás' 
soy_br_states = bra[bra['NAME_1'].isin(state_brc_names)]

#%% yield model - WOFOST
start_date, end_date = '1959-12-31','2015-12-31'

DS_y2 = xr.open_dataset("yield_isimip_epic_3A_2.nc",decode_times=True).sel(lat=slice(0,50), lon=slice(-160,-10))
DS_y2 = mask_shape_border(DS_y2,soy_us_states) #US_shape
DS_y2 = DS_y2.dropna(dim = 'lon', how='all')
DS_y2 = DS_y2.dropna(dim = 'lat', how='all')
DS_y=DS_y2
DS_y=DS_y.sel(time=slice(start_date, end_date))

#second mask
DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )

ds_mask = xr.open_dataset("mirca_2000_mask_soybean_rainfed.nc")
DS_y = DS_y.where(ds_mask['soybean_rainfed'] >= 0.9 )

# ds_iizumi = xr.open_dataset("soybean_iizumi_1981_2016.nc")
# DS_y = DS_y.where(DS_y['yield'].mean('time') > 0.5 )
# ds_iizumi = ds_iizumi.rename({'latitude': 'lat', 'longitude': 'lon'})
# DS_y = DS_y.where(ds_iizumi['yield'].mean('time') > 0.5 )

DS_y = DS_y.dropna(dim = 'lon', how='all')
DS_y = DS_y.dropna(dim = 'lat', how='all')
if len(DS_y.coords) >3 :
    DS_y=DS_y.drop('spatial_ref')
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
DS_cli = xr.merge([DS_t_max.tmx,DS_dtr.dtr, DS_spei2.spei2_, DS_prec.pre ]) #DS_dtr.dtr,
# #mask
DS_cli_us = DS_cli.where(DS_y['yield'].mean('time') > -0.1 )
if len(DS_cli_us.coords) >3 :
    DS_cli_us=DS_cli_us.drop('spatial_ref')
    
plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_cli_us['tmx'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

# Features considered for this case
months_to_be_used =  [7,8]
list_feature_names = list(DS_cli_us.keys())
column_names=[]
for i in list_feature_names:
    for j in months_to_be_used: #range(6,10):
        column_names.append(i+str(j))

# second way - check if they match
df_features_avg_list_2 = []
for feature in list_feature_names:        
    df_feature_2 = DS_cli_us[feature].to_dataframe().groupby(['time']).mean()
    df_feature_2_reshape = reshape_data(df_feature_2).loc[:,months_to_be_used]
    df_feature_2_det = pd.DataFrame( signal.detrend(df_feature_2_reshape, axis=0,bp=range_test ), index=df_feature_2_reshape.index, columns = df_feature_2_reshape.columns) + df_feature_2_reshape.mean(axis=0)
    df_features_avg_list_2.append(df_feature_2_det)    

df_clim_avg_features_2 = pd.concat(df_features_avg_list_2, axis=1)
df_clim_avg_features_2.columns = column_names
df_clim_total_2 = pd.concat([df_clim_avg_features_2,df_epic], axis=1, sort=False) # final table with all features + yield

#convert for future simulations
df_clim_avg_features = df_clim_avg_features_2

for feature in list_feature_names:   
    test = DS_cli_us[feature].sel(time=DS_cli_us.time.dt.month.isin([months_to_be_used])).to_dataframe().groupby(['time']).mean()
    test_reshape= reshape_data(test).loc[:,months_to_be_used]
    for i in test_reshape.columns:
        plt.figure(figsize=(9, 8), dpi=144)
        plt.plot(test_reshape[i], label=f'{test.keys()[0]}_{i}', color = 'darkblue' )
        plt.plot(df_clim_avg_features[f'{test.keys()[0]}{i}'], label =f' detrended {test.keys()[0]}_{i}',  color = 'orangered', alpha = 1)
        plt.legend(loc="upper left")

# if 'spei2_6' in df_clim_avg_features:
#     df_clim_avg_features = df_clim_avg_features.drop(columns=['spei2_6'])
if 'spei2_7' in df_clim_avg_features:
    df_clim_avg_features = df_clim_avg_features.drop(columns=['spei2_7'])
if 'spei2_9' in df_clim_avg_features:
    df_clim_avg_features = df_clim_avg_features.drop(columns=['spei2_9'])
# if 'tmx7' in df_clim_avg_features:
#     df_clim_avg_features = df_clim_avg_features.drop(columns=['tmx7'])
# if 'dtr7' in df_clim_avg_features:
#     df_clim_avg_features = df_clim_avg_features.drop(columns=['dtr7'])


df_clim_avg_features_ar1 = pd.concat([df_clim_avg_features_ar.iloc[:,0:2],df_clim_avg_features_ar.iloc[:,3:5], df_clim_avg_features_ar.iloc[:,6:8], df_clim_avg_features_ar.iloc[:,9:10] ], axis=1)
kwargs = dict(histtype='stepfilled', alpha=0.3)
plt.hist(df_clim_avg_features_us.iloc[:,0], **kwargs)
plt.hist(df_clim_avg_features_brc.iloc[:,0], **kwargs)
plt.hist(df_clim_avg_features_br_s.iloc[:,0], **kwargs)
plt.hist(df_clim_avg_features_ar.iloc[:,0], **kwargs)

# test for temperature
df_con_hist = pd.concat( [df_clim_avg_features_us.iloc[:,0],df_clim_avg_features_brc.iloc[:,0],df_clim_avg_features_br_s.iloc[:,0], df_clim_avg_features_ar.iloc[:,0] ],axis=1)
df_con_hist.columns = ['us','central brazil','south brazil','argentina']

plt.figure(figsize = (6,6), dpi=144)
fig = sns.displot(df_con_hist,kind="kde",height=7.5, aspect=1)
fig.set(xlabel="Temperature")
plt.tight_layout()
plt.show()

# sns.displot(df_clim_avg_features_us.iloc[:,0],kind="kde")
# sns.displot(df_clim_avg_features_brc.iloc[:,0],kind="kde")
# sns.displot(df_clim_avg_features_br_s.iloc[:,0],kind="kde")
# sns.displot(df_clim_avg_features_ar.iloc[:,0],kind="kde")
list_feautures_names = ['temperature Celisus Degrees','Precipitation mm/month','Diurnal Temperature Range (C)','SPEI']
for feature in range(len(df_clim_avg_features_us.columns)):
    df_con_hist = pd.concat( [df_clim_avg_features_us.iloc[:,feature],df_clim_avg_features_brc.iloc[:,feature],df_clim_avg_features_br_s.iloc[:,feature], df_clim_avg_features_ar1.iloc[:,feature] ],axis=1)
    df_con_hist.columns = ['us','central brazil','south brazil','argentina']
    
    plt.figure(figsize = (6,6), dpi=144)
    fig = sns.displot(df_con_hist,kind="kde",height=7.5, aspect=1, linewidth=3,fill=True,alpha=.1)
    fig.set(xlabel=df_clim_avg_features_us.columns[feature][:-1])
    ax.set_ylabel(dict_attrs[mean_cli_no_fail.name[:-1] if len([x for x in mean_cli_no_fail.name[-2:] if x.isdigit()]) == 1 else mean_cli_no_fail.name[:-2]])

    plt.tight_layout()
    plt.show()




    