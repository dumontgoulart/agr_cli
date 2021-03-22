
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,ShuffleSplit
from sklearn.linear_model import LinearRegression,LogisticRegressionCV,LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score

from failure_probability import failure_probability

#### precipitation around jul-sep
periods_to_be_used = range(13,14) #range(12,16) for Seasons, range(0,12) for months
df_ivory_tmx= pd.read_csv (r'cocoa_cote_divoire/crucy.v4.04.1901.2019.Ivory_Coast.tmx.per',skipinitialspace=True, index_col=0, skiprows=3, delimiter=' ').loc['1961':'2015'].iloc[:,periods_to_be_used].add_prefix('tmx_')
df_ivory_tmn= pd.read_csv (r'cocoa_cote_divoire/crucy.v4.04.1901.2019.Ivory_Coast.tmn.per',skipinitialspace=True, index_col=0, skiprows=3, delimiter=' ').loc['1961':'2015'].iloc[:,periods_to_be_used].add_prefix('tmn_')
df_ivory_dtr= pd.read_csv (r'cocoa_cote_divoire/crucy.v4.04.1901.2019.Ivory_Coast.dtr.per',skipinitialspace=True, index_col=0, skiprows=3, delimiter=' ').loc['1961':'2015'].iloc[:,periods_to_be_used].add_prefix('dtr_')
df_ivory_pet= pd.read_csv (r'cocoa_cote_divoire/crucy.v4.04.1901.2019.Ivory_Coast.pet.per',skipinitialspace=True, index_col=0, skiprows=3, delimiter=' ').loc['1961':'2015'].iloc[:,periods_to_be_used].add_prefix('pet_')
df_ivory_pre= pd.read_csv (r'cocoa_cote_divoire/crucy.v4.04.1901.2019.Ivory_Coast.pre.per',skipinitialspace=True, index_col=0, skiprows=3, delimiter=' ').loc['1961':'2015'].iloc[:,periods_to_be_used].add_prefix('pre_')
df_ivory_clim = pd.concat([df_ivory_tmx,df_ivory_dtr,df_ivory_pet,df_ivory_pre], axis=1, sort=False)

#SPEI
def reshape_data(dataframe):  #converts and reshape data
    dataframe['month'] = dataframe.index.get_level_values('time').month
    dataframe['year'] = dataframe.index.get_level_values('time').year
    dataframe.set_index('month', append=True, inplace=True)
    dataframe.set_index('year', append=True, inplace=True)
    dataframe = dataframe.reorder_levels(['time', 'year','month'])
    dataframe.index = dataframe.index.droplevel('time')
    dataframe = dataframe.unstack('month')
    dataframe.columns = dataframe.columns.droplevel()
    return dataframe
DS_spei = xr.open_dataset("spei02.nc",decode_times=True).sel(time=slice('1961-01-01','2015-12-31'))
ci = gpd.read_file('cocoa_cote_divoire/gadm36_CIV_0.shp', crs="epsg:4326") 
ci1_shapes = list(shpreader.Reader('cocoa_cote_divoire/gadm36_CIV_0.shp').geometries())
DS_spei_ci = mask_shape_border(DS_spei,ci) 
dataarray_iso = DS_spei_ci.spei.where(DS_spei_ci.spei > -1000, -300000)
dataarray_iso_1 =  xr.DataArray(signal.detrend(dataarray_iso, axis=0), dims=dataarray_iso.dims, coords=dataarray_iso.coords, attrs=dataarray_iso.attrs, name = DS_spei_ci.spei.name ) + dataarray_iso.mean(axis=0)
da_spei_det = dataarray_iso_1.where(dataarray_iso_1 > -100, np.nan ).sel(time = dataarray_iso_1.indexes['time'].month.isin([7,8,9,10])) 
df_spei = da_spei_det.to_dataframe().groupby(['time']).mean() 
df_spei_ci = reshape_data(df_spei).add_prefix('spei_')

range_test = list(range(10, int(df_ivory_clim.index[-1]-df_ivory_clim.index[0]), 10))
#detrend
df_clim_detrend_1 =  pd.DataFrame(signal.detrend(df_ivory_clim, axis=0,bp=range_test) + df_ivory_clim.mean(axis=0).values, index = df_ivory_clim.index, columns = df_ivory_clim.columns)
#plots comparing detrend
df_ivory_tmx.iloc[:,0].plot()
df_clim_detrend_1.iloc[:,0].plot()
plt.show()
# add spei to clim features
df_clim_detrend = pd.concat([df_clim_detrend_1, df_spei_ci], axis =1 )
##### Yield
df_ivory= pd.read_csv (r'cocoa_cote_divoire/FAOSTAT_data_11-25-2020.csv').set_index('Year').loc['1961':'2015']
df_ivory_yield = df_ivory[['Value']]/10000
#detrend
df_ivory_yield_det =  pd.DataFrame(signal.detrend(df_ivory_yield, axis=0,bp=range_test) +df_ivory_yield.mean(axis=0).values, index =df_ivory_yield.index, columns = df_ivory_yield.columns)
plt.plot(df_ivory_yield_det)
plt.show()

# Run analysis one cluster

lr_f1, lr_f2, lr_auc, feature_list, brf_model  = failure_probability(df_clim_detrend, df_ivory_yield_det, show_scatter = True)



######################

# PALM OIL
start_date, end_date = '1979-12-31','2015-12-31'

# Crop space to either US or soy states
usa = gpd.read_file('gadm36_IDN_1.shp', crs="epsg:4326") 
ma1_shapes = list(shpreader.Reader('gadm36_IDN_1.shp').geometries())
state_names = ['Riau','Sumatera Barat','Sumatera Selatan', 'Sumatera Utara','Kalimantan Barat', 'Kalimantan Selatan','Kalimantan Tengah', 'Kalimantan Timur']
soy_us_states = usa[usa['NAME_1'].isin( state_names)]

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

DS_cli = xr.merge([DS_t_max.tmx, DS_prec.pre,DS_dtr.dtr, DS_spei2.spei2_]) #,DS_prec.pre, DS_pet.pet
# #mask
DS_cli_idn = mask_shape_border(DS_cli,soy_us_states) 
if len(DS_cli_idn.coords) >3 :
    DS_cli_idn=DS_cli_idn.drop('spatial_ref')


# plt.figure(figsize=(14,10)) #plot clusters
# ax=plt.axes(projection=ccrs.Mercator())
# DS_cli_idn['tmx'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
# ax.add_geometries(ma1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([90,140,-10,10], ccrs.PlateCarree())
# plt.show()

##### Yield
df_idn = pd.read_csv (r'palm_oil_yield.csv').set_index('Year').loc['1980':'2015']
df_idn_yield = df_idn[['Value']]/10000
#detrend
range_test = list(range(20, int(df_idn_yield.index[-1]-df_idn_yield.index[0]), 20))

df_idn_yield_det =  pd.DataFrame(signal.detrend(df_idn_yield, axis=0,bp=range_test) +df_idn_yield.mean(axis=0).values, index =df_idn_yield.index, columns = df_idn_yield.columns)
plt.plot(df_idn_yield)
plt.plot(df_idn_yield_det)
plt.show()

#convert to dataframe and separate months
months_to_be_used =  range(3,6) #[7,8]
list_feature_names = list(DS_cli_idn.keys())
column_names=[]
for i in list_feature_names:
    for j in months_to_be_used: #range(6,10):
        column_names.append(i+str(j))
        
df_features_avg_list_2 = []
for feature in list_feature_names:        
    df_feature_2 = DS_cli_idn[feature].to_dataframe().groupby(['time']).mean()
    df_feature_2_reshape = reshape_data(df_feature_2).loc[:,months_to_be_used]
    df_feature_2_det = pd.DataFrame( signal.detrend(df_feature_2_reshape, axis=0,bp=range_test ), index=df_feature_2_reshape.index, columns = df_feature_2_reshape.columns) + df_feature_2_reshape.mean(axis=0)
    df_features_avg_list_2.append(df_feature_2_det)    

df_clim_avg_features_2 = pd.concat(df_features_avg_list_2, axis=1)
df_clim_avg_features_2.columns = column_names



# Run analysis one cluster

lr_f1, lr_f2, lr_auc, feature_list, brf_model  = failure_probability(df_clim_avg_features_2, df_idn_yield_det, show_scatter = True)




###

#Fraction of rainfed as rainfed area divided by the total
fraction_rf_to_ir = data_rain /( total_harvested_area)

fraction_rf_to_ir_90 = fraction_rf_to_ir.where(fraction_rf_to_ir > 0.9)



fraction_rf_to_ir_90 = fraction_rf_to_ir_90.iloc[::-1]
fraction_rf_to_ir_90.columns=DS_pet.lon.values
fraction_rf_to_ir_90.index = DS_pet.lat.values
data_test = fraction_rf_to_ir_90


# mask_harv = xr.Dataset.from_dataframe(data)
DS_base = xr.open_dataset("cru_ts4.04.1901.2019.tmx.dat.nc",decode_times=True)
DS_base_mean = DS_base.tmx.mean('time')
DS_base_mean.values = data_test
DS_base_mean.name = 'palmoil_rainfed'
DS_base_mean.to_netcdf('mirca_2000_mask_palmoil_rainfed.nc')


# Crop space to either US or soy states
usa = gpd.read_file('gadm36_IDN_1.shp', crs="epsg:4326") 
ma1_shapes = list(shpreader.Reader('gadm36_IDN_1.shp').geometries())
state_names = ['Riau','Sumatera Barat','Sumatera Selatan', 'Sumatera Utara','Kalimantan Barat', 'Kalimantan Selatan','Kalimantan Tengah', 'Kalimantan Timur']
soy_us_states = usa[usa['NAME_1'].isin( state_names)]
