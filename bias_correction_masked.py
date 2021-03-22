import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/Paper_drought/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import seaborn as sns

def bias_correction_masked(mask, start_date, end_date, cru_detrend = False):
    """
    This function takes as input the EC_earth model projections for PD, 2C and 3C 
    and the mask for a region (US, Brazil, etc.) and corrects the bias to match
    CRU cru_ts4.04 for the climatology 1990-2020.
    
    Parameters:
    Mask: regions to which the data should be cropped to. Should have a value where
    values > 0 can be mapped. EX: Dataset with yields, where climate cells are selected
    only for grids with positive (real) yields. 
    
    Detrend: Whether or not to consider detrending the CRU dataset used to 
    correct the bias for the EC earth model.
        
    Important: needs to match EC_earth resolution, which means it needs to be rescaled. Follows:
    $ cdo -remapycon,tasmax_m_ECEarth_PD_ensemble_2035-4035.nc YOUR_FILE.nc YOUR_FILE_LOWRES.nc
        
    Returns:
    DS_cli_ec, DS_cli_ec_2C: The EC_earth projections biased corrected and masked.
    
    Created on Wed Feb 10 17:19:09 2021 
    by @HenriqueGoulart
    """
   
    #%% Openinig and cleaning data
    # Function
    def states_mask(input_gdp_shp, state_names) :
        country = gpd.read_file(input_gdp_shp, crs="epsg:4326") 
        country_shapes = list(shpreader.Reader(input_gdp_shp).geometries())
        soy_states = country[country['NAME_1'].isin(state_names)]
        states_area = soy_states['geometry'].to_crs({'proj':'cea'}) 
        states_area_sum = (sum(states_area.area / 10**6))
        return soy_states, country_shapes, states_area_sum    
    
    state_names = ['Iowa','Illinois','Minnesota','Indiana','Nebraska','Ohio',
                   'South Dakota','North Dakota', 'Missouri','Arkansas']
    soy_us_states, us1_shapes, us_states_area_sum = states_mask('gadm36_USA_1.shp', state_names)
    
    state_names = ['Rio Grande do Sul','Paraná']
    soy_br_states, br1_shapes, brs_states_area_sum = states_mask('gadm36_BRA_1.shp', state_names)
    
    state_names = ['Buenos Aires','Santa Fe', 'Córdoba'] 
    soy_ar_states, ar1_shapes, ar_states_area_sum = states_mask('gadm36_ARG_1.shp', state_names)
    
    state_names = ['Mato Grosso','Goiás']
    soy_brc_states, br1_shapes, brc_states_area_sum = states_mask('gadm36_BRA_1.shp', state_names)

    # Needs to import DS_Yield from other script
    
    if isinstance(mask, xr.core.dataset.Dataset) == True:
        mask_ref = mask[list(mask.keys())[0]].mean('time') 
    elif isinstance(mask, xr.core.dataarray.DataArray) == True:
        mask_ref = mask.mean('time') 
    else:
        raise ValueError('Mask should be either Dataset or Dataarray')
        
    # CRU data
    DS_t_max_cru = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.tmx.dat_lr.nc",
                                   decode_times=True).sel(time=slice(start_date, end_date))
    DS_t_max_cru_us = DS_t_max_cru.where(mask_ref > -0.1 ) # mask
    
    DS_dtr_cru = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.dtr.dat_lr.nc",
                                 decode_times=True).sel(time=slice(start_date, end_date))
    DS_dtr_cru_us = DS_dtr_cru.where(mask_ref > -0.1 )
    
    DS_pre_cru = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.pre.dat_lr.nc",
                                 decode_times=True).sel(time=slice(start_date, end_date))
    DS_pre_cru_us = DS_pre_cru.where(mask_ref > -0.1 )
    
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
    
    # Create Bias correction dataset: Detrend time series for higher range and add mean for 1990-2020. 
    if cru_detrend == True:
        start_date_mean, end_date_mean = '31-12-1916', '31-12-2016'
        DS_tmx_cru_us_det = xr.DataArray( detrend_dim(DS_t_max_cru_us.tmx, 'time') + DS_t_max_cru_us.tmx.sel(
            time=slice(start_date_mean, end_date_mean)).mean('time'), name= DS_t_max_cru_us.tmx.name, attrs = DS_t_max_cru_us.tmx.attrs )
        DS_dtr_cru_us_det = xr.DataArray( detrend_dim(DS_dtr_cru_us.dtr, 'time') + DS_dtr_cru_us.dtr.sel(
            time=slice(start_date_mean, end_date_mean)).mean('time'), name= DS_dtr_cru_us.dtr.name, attrs = DS_dtr_cru_us.dtr.attrs)
        DS_pre_cru_us_det = xr.DataArray( detrend_dim(DS_pre_cru_us.pre, 'time') + DS_pre_cru_us.pre.sel(
            time=slice(start_date_mean, end_date_mean)).mean('time'), name= DS_pre_cru_us.pre.name, attrs = DS_pre_cru_us.pre.attrs)
        
    if cru_detrend == False: #No detrending
        DS_tmx_cru_us_det = DS_t_max_cru_us.tmx
        DS_dtr_cru_us_det = DS_dtr_cru_us.dtr
        DS_pre_cru_us_det = DS_pre_cru_us.pre
    
    # Merge everything
    DS_cru_merge = xr.merge([DS_tmx_cru_us_det, DS_dtr_cru_us_det, DS_pre_cru_us_det])
    # Benchmark of undetrended variables  
    DS_cru_prov = xr.merge([DS_t_max_cru_us.tmx, DS_dtr_cru_us.dtr, DS_pre_cru_us.pre])
        
    # Test detrending - comparison of time series
    for feature in list(DS_cru_merge.keys()):
        df_feature = DS_cru_prov[feature].to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
        df_feature_8 = df_feature.loc[df_feature.index.month == 8]
    
        df_feature_det = DS_cru_merge[feature].to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
        df_feature_det_8 = df_feature_det.loc[df_feature_det.index.month == 8]
                
        plt.plot(df_feature_8, label = f'{feature} normal 8')
        plt.plot(df_feature_det_8, label = f"{feature} det 8")
        plt.legend()
        plt.show()
        
        # KDEPLOT to compare pdfs
        df_feature_8['Scenario'] = 'Not detrended'
        df_feature_det_8['Scenario'] = 'Detrended'
        df_hist_us = pd.concat( [df_feature_8, df_feature_det_8],axis=0)

        plt.figure(figsize = (6,6), dpi=144)
        fig = sns.kdeplot( data = df_hist_us, x= df_hist_us[feature], hue="Scenario", fill=True, alpha=.2)
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
    DS_tmx_ec['tasmax'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-105,-25,-50,50], ccrs.PlateCarree())
    plt.show()
    
    # Test plot to see if it's good    
    plt.figure(figsize=(20,10)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    DS_dtr_cru_us['dtr'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-105,-65,20,50], ccrs.PlateCarree())
    plt.show()
    
    # Measure diference between model and observed data
    subt = DS_tmx_ec['tasmax'].mean('time') - DS_t_max_cru_us['tmx'].mean('time')
    
    plt.figure(figsize=(20,10)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    subt.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
    ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-105,-65,20,50], ccrs.PlateCarree())
    ax.set_title('Difference between CRU and EC-earth')
    plt.show()
    
    #%% BIAS CORRECTION - Convert and tests
    from xclim import sdba
    import scipy.stats as stats
    
    def bias_analysis(obs_data, model_data):
        """
        Bias analysis graphs by entering the observed data and the model to be corrected.
        
        Parameters:
        
        obs_data: data to serve as training reference, the observed dataset;
        model_data: the data that is required to be adjusted, usually the model data.
            
        No return, just graphs showing bias
        
        Created on Wed Feb 10 17:19:09 2021 
        by @HenriqueGoulart
        """
        df_cru_cli=obs_data.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
        df_ec=model_data[list(model_data.keys())[0]].to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
        
        # Compare mean annual cycle
        df_cru_year = df_cru_cli.groupby(df_cru_cli.index.month).mean()
        df_ec_year = df_ec.groupby(df_ec.index.month).mean()
        # plot
        plt.figure(figsize = (6,6), dpi=200)
        plt.plot(df_cru_year, label = 'CRU', color = 'darkblue')
        plt.plot(df_ec_year, label = 'EC-Earth', color = 'red' )
        plt.ylabel(obs_data.attrs['units'])
        plt.title(f'Mean annual cycle - {obs_data.name}') 
        plt.legend(loc="lower left")
        plt.show()
        
        # Compare std each year
        df_cru_year_std = df_cru_cli.groupby(df_cru_cli.index.month).std()
        df_ec_year_std = df_ec.groupby(df_ec.index.month).std()
        # plot
        plt.figure(figsize = (6,6), dpi=200)
        plt.plot(df_cru_year_std, label = 'CRU', color = 'darkblue')
        plt.plot(df_ec_year_std, label = 'EC-Earth', color = 'red' )
        plt.ylabel(obs_data.attrs['units'])
        plt.title(f'Variability around the mean - {obs_data.name}')
        plt.legend(loc="lower left")
        plt.show()
        
       # Compare Q-Q plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        stats.probplot(df_ec.iloc[:,0], dist=stats.beta, sparams=(3,2),plot=plt,fit=False)
        stats.probplot(df_cru_cli.iloc[:,0], dist=stats.beta, sparams=(3,2), plot=plt,fit=False)
        ax.get_lines()[0].set_color('C1')
        plt.legend(loc="lower left")
        plt.show()
        
    # bias correction for tasmax Quantile mapping - First compare the original, then adjust the bias and compare the bias corrected version
    bias_analysis( DS_cru_merge.tmx, DS_tmx_ec)
    dqm_tmx = sdba.adjustment.DetrendedQuantileMapping(nquantiles=25, group='time.month', kind='+')
    dqm_tmx.train(DS_cru_merge['tmx'],DS_tmx_ec['tasmax'])
    DS_tmx_ec_cor = dqm_tmx.adjust(DS_tmx_ec['tasmax'], interp='linear')
    DS_tmx_ec_cor = DS_tmx_ec_cor.to_dataset(name= 'tmx')
    bias_analysis(DS_cru_merge.tmx, DS_tmx_ec_cor)
    
    # bias correction for dtr Quantile mapping - First compare the original, then adjust the bias and compare the bias corrected version
    bias_analysis(DS_cru_merge.dtr, DS_dtr_ec)
    dqm_dtr = sdba.adjustment.DetrendedQuantileMapping(nquantiles=25, group='time.month', kind='+')
    dqm_dtr.train(DS_cru_merge['dtr'],DS_dtr_ec['dtr'])
    DS_dtr_ec_cor = dqm_dtr.adjust(DS_dtr_ec['dtr'], interp='linear')
    DS_dtr_ec_cor = DS_dtr_ec_cor.to_dataset(name= 'dtr')
    bias_analysis(DS_cru_merge.dtr, DS_dtr_ec_cor)
    
    # bias correction for precipitation Quantile mapping - First compare the original, then adjust the bias and compare the bias corrected version
    bias_analysis(DS_cru_merge.pre, DS_pre_ec)
    dqm_pr = sdba.adjustment.DetrendedQuantileMapping(nquantiles=25, group='time.month', kind='+')
    dqm_pr.train(DS_cru_merge['pre'],DS_pre_ec['pr'])
    DS_pre_ec_cor = dqm_pr.adjust(DS_pre_ec['pr'], interp='linear')
    DS_pre_ec_cor = DS_pre_ec_cor.to_dataset(name= 'pre')
    bias_analysis(DS_cru_merge.pre, DS_pre_ec_cor)
    
    # Merge in one dataset
    DS_cli_ec = xr.merge([DS_tmx_ec_cor.tmx, DS_dtr_ec_cor.dtr, DS_pre_ec_cor.pre])
    
    for feature in list(DS_cli_ec.keys()):
        subt_cor = DS_cli_ec[feature].mean('time') - DS_cru_merge[feature].mean('time')
        
        plt.figure(figsize=(20,10)) #plot clusters
        ax=plt.axes(projection=ccrs.Mercator())
        subt_cor.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,levels=10)
        ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
        ax.set_extent([-105,-65,20,50], ccrs.PlateCarree())
        ax.set_title(f'Difference between CRU and EC-earth for {feature}')
        plt.show()
        
    #%% 2C simulation - future data
    
    #Temp - Kelvin to celisus
    DS_tmx_ec_2C = open_regularize("EC_earth_2C/tasmax_m_ECEarth_2C_ensemble_2062-4062.nc", DS_pre_cru_us['pre']) 
    # precipitation
    DS_pre_ec_2C = open_regularize("EC_earth_2C/pr_m_ECEarth_2C_ensemble_2062-4062.nc", DS_pre_cru_us['pre']) 
    # dtr
    DS_dtr_ec_2C = open_regularize("EC_earth_2C/dtr_m_ECEarth_2C_ensemble_2062-4062.nc", DS_pre_cru_us['pre']) 
    
    # bias correction for tasmax
    bias_analysis(DS_cru_merge.tmx, DS_tmx_ec_2C)
    DS_tmx_ec_2C_cor = dqm_tmx.adjust(DS_tmx_ec_2C['tasmax'], interp='linear')
    DS_tmx_ec_2C_cor = DS_tmx_ec_2C_cor.to_dataset(name= 'tmx')
    bias_analysis(DS_cru_merge.tmx, DS_tmx_ec_2C_cor)
    
    # bias correction for dtr Quantile mapping
    bias_analysis(DS_cru_merge.dtr, DS_dtr_ec_2C)
    DS_dtr_ec_2C_cor = dqm_dtr.adjust(DS_dtr_ec_2C['dtr'], interp='linear')
    DS_dtr_ec_2C_cor = DS_dtr_ec_2C_cor.to_dataset(name= 'dtr')
    bias_analysis(DS_cru_merge.dtr, DS_dtr_ec_2C_cor)
    
    # bias correction for precipitation Quantile mapping
    bias_analysis(DS_cru_merge.pre, DS_pre_ec_2C)
    DS_pre_ec_2C_cor = dqm_pr.adjust(DS_pre_ec_2C['pr'], interp='linear')
    DS_pre_ec_2C_cor = DS_pre_ec_2C_cor.to_dataset(name= 'pre')
    bias_analysis(DS_cru_merge.pre, DS_pre_ec_2C_cor)
    
    # Merge in one dataset
    DS_cli_ec_2C = xr.merge([DS_tmx_ec_2C_cor.tmx, DS_dtr_ec_2C_cor.dtr, DS_pre_ec_2C_cor.pre])

    return(DS_cli_ec, DS_cli_ec_2C)

#%% Function conversion

def function_conversion(DS_cli_ec_PD, DS_cli_ec_2C, DS_cli_ec_3C = None, months_to_be_used=[7,8], water_year = False):
    """
    This function takes as input the bias corrected EC_earth model projections,
    the months to be selected for the season (months_to_be_used = [7,8]),
    
    Parameters:
    
    DS_cli_ec_PD, DS_cli_ec_2C, DS_cli_ec_3C: The datasets for present day climate and for the 2C (optional 3C);
    months_to_be_used: which months to use when converting to dataframe;
    water_year: if calculations need to be carried out for interannual periods.
    
    Returns:
    
    The formatted dataframes representing EC_earth projections for PD,2C,3C
    
    Created on Wed Feb 10 17:19:09 2021 
    by @HenriqueGoulart
    """
    # Features considered for this case
    column_names = [i+str(j) for i in list(DS_cli_ec_PD.keys()) for j in months_to_be_used]
    DS_cru = xr.open_dataset("EC_earth_PD/cru_ts4.04.1901.2019.tmx.dat_lr.nc",
                             decode_times=True).sel(time=slice('31-12-1989', '31-12-2020'))
    
    #convert to dataframe, reshape so every month is in a separate colum:
    def reshape_data(dataarray):  #converts and reshape data
        if isinstance(dataarray, pd.DataFrame):
            dataframe = dataarray.dropna(how='all')
            dataframe['month'] = dataframe.index.get_level_values('time').month
            dataframe['year'] = (np.repeat(range(0,2000), 12))
            dataframe.set_index('month', append=True, inplace=True)
            dataframe.set_index('year', append=True, inplace=True)
            dataframe = dataframe.reorder_levels(['time', 'year','month'])
            dataframe.index = dataframe.index.droplevel('time')
            dataframe = dataframe.unstack('month')
            dataframe.columns = dataframe.columns.droplevel()
        else:    
            dataframe = dataarray.to_dataframe().dropna(how='all')
            dataframe['month'] = dataframe.index.get_level_values('time').month
            dataframe['year'] = (np.repeat(range(0,2000), 12))
            dataframe.set_index('month', append=True, inplace=True)
            dataframe.set_index('year', append=True, inplace=True)
            dataframe = dataframe.reorder_levels(['time', 'year','month', 'lat', 'lon'])
            dataframe.index = dataframe.index.droplevel('time')
            dataframe = dataframe.unstack('month')
            dataframe.columns = dataframe.columns.droplevel()
        return dataframe
       
    # Function to transform the dataset into dataframe for each selected month
    def dataset_to_dataframe(DS_cli_ec):
        df_features_ec = []
        for feature in list(DS_cli_ec.keys()):        
            df_feature_2 = DS_cli_ec[feature].to_dataframe().groupby(['time']).mean()
            
            # arrange time so water year stays a single year (12,1,2,3...) if not, skip
            if (water_year == True):
                df_feature_2.index = np.tile(DS_cru.time.sel(time=slice('31-12-2010', '31-12-2015')), 400)
                df_feature_2.index.name = 'time'
                df_feature_2['year'] = df_feature_2.index.year.where(df_feature_2.index.month < 10, 
                                                                     df_feature_2.index.year + 1)
                df_feature_2['month'] = pd.DatetimeIndex(df_feature_2.index).month
                df_feature_2['day'] = pd.DatetimeIndex(df_feature_2.index).day
                df_feature_2['time'] = pd.to_datetime(df_feature_2.iloc[:,1:4])
                df_feature_2.index = df_feature_2['time']
                df_feature_2.drop(['month','day','time','year'], axis=1, inplace = True)
            
            df_feature_2_reshape = reshape_data(df_feature_2).loc[:,months_to_be_used]
            df_features_ec.append(df_feature_2_reshape)            
        
        df_features_ec = pd.concat(df_features_ec, axis=1) # format data
        df_features_ec.columns = column_names
        
        # Adapt the structure to match the RF structure
        if len(df_features_ec.columns) == 6:
            df_features_ec_season_local = pd.concat( [df_features_ec.iloc[:,0:2].mean(axis=1),
                                                      df_features_ec.iloc[:,2:4].mean(axis=1),
                                                      df_features_ec.iloc[:,4:6].mean(axis=1)], axis=1 )
            df_features_ec_season_local.columns=[f'tmx_{months_to_be_used[0]}_{months_to_be_used[1]}',
                                                 f'dtr_{months_to_be_used[0]}_{months_to_be_used[1]}', 
                                                 f'precip_{months_to_be_used[0]}_{months_to_be_used[1]}']
            return df_features_ec,df_features_ec_season_local
        
        elif len(df_features_ec.columns) == 9:
            df_features_ec_season_local = pd.concat( [df_features_ec.iloc[:,0:3].mean(axis=1),
                                                      df_features_ec.iloc[:,3:6].mean(axis=1),
                                                      df_features_ec.iloc[:,6:9].mean(axis=1)], axis=1 )
            df_features_ec_season_local.columns=[f'tmx_{months_to_be_used[0]}_{months_to_be_used[1]}_{months_to_be_used[2]}',
                                                 f'dtr_{months_to_be_used[0]}_{months_to_be_used[1]}_{months_to_be_used[2]}', 
                                                 f'precip_{months_to_be_used[0]}_{months_to_be_used[1]}_{months_to_be_used[2]}']
            return df_features_ec,df_features_ec_season_local
     
    # PRESENT DAY
    df_features_ec,df_features_ec_season = dataset_to_dataframe(DS_cli_ec_PD)
    print("PD done!")
    # FUTURE 2C
    df_features_ec_2C,df_features_ec_season_2C = dataset_to_dataframe(DS_cli_ec_2C)
    
    # FUTURE 3C
    if DS_cli_ec_3C is None:
        return df_features_ec_season,df_features_ec_season_2C
    
    elif DS_cli_ec_3C is not None:
        df_features_ec_3C,df_features_ec_season_3C = dataset_to_dataframe(DS_cli_ec_3C)
        
        return df_features_ec_season, df_features_ec_season_2C, df_features_ec_season_3C
    
# second way - check if they match

#%% Scenario exploration

def predictions_permutation(brf_model, df_clim_agg_chosen, df_features_ec_season,
                            df_features_ec_2C_season = None, df_clim_2012 = None): #df_features_ec_3C_season = None 

    """
    This function takes as input the bias corrected EC_earth model projections,
    the months to be selected for the season (months_to_be_used = [7,8]),
    
    Parameters:
    brf_model: the machine learning model trained for the area;
    df_features_ec_season: dataframe containing the climatic features as input
    (processed by previous function)
    df_features_ec_season_2C = dataframe containing cl. features for future period
        
    Returns:
    Series of plots and prints showing the predictions of the ML for different
    time periods.
    score_prc: the ratio (score) indicating the amount of seasons with failure
    per total seasons.
    
    Created on Wed Feb 10 17:19:09 2021 
    by @HenriqueGoulart
    """
    # Storyline probability of failure 2012
    y_pred_2012 = brf_model.predict_proba(df_clim_2012.values.reshape(1, -1))[0][1]
    print("2012 prediction is: ",y_pred_2012)
    # Predictions for Observed data PD
    def predictions(brf_model,df_features_ec_season):
        
        y_pred = brf_model.predict(df_features_ec_season)
        score_prc = sum(y_pred)/len(y_pred) 
        print("\n The total failures are:", sum(y_pred),
              " And the ratio of failure seasons by total seasons is:", score_prc, "\n")     
        probs = brf_model.predict_proba(df_features_ec_season)        
        if df_clim_2012 is not None:
            limit_2012 = y_pred_2012
            seasons_over_2012 = df_features_ec_season[probs[:,1]>=limit_2012]
            print(f"\n Number of {limit_2012}% events: {len(seasons_over_2012)} and mean conditions are:", 
                  np.mean(seasons_over_2012))
        
        return y_pred, score_prc, probs, seasons_over_2012
    
    def return_period(df, colname):

        # Sort data smallest to largest
        sorted_data = df.sort_values(by=colname)
        # Count total obervations
        n = sorted_data.shape[0]      
        # Add a numbered column 1 -> n to use in return calculation for rank
        sorted_data.insert(0, 'rank', range(1, 1 + n))        
        # Calculate probability
        sorted_data["probability"] = (n - sorted_data["rank"] + 1) / (n + 1)        
        # Calculate return - yearly data, no need to further trnasform
        sorted_data["return-years"] = (1 / sorted_data["probability"])
    
        return(sorted_data)
    
    def plot_probs_failure(probs, probs_perm):
        
        # put them together in the same dataframe for plotting
        probs_agg=pd.DataFrame( [probs[:,1],probs_perm[:,1]]).T
        probs_agg.columns=['Ordered','Permuted']
    
        # plots comparing prediction confidence for obs and perumuted
        probs_agg_melt = probs_agg.melt(value_name='Failure probability').assign(data='Density')
        
        sns.displot(probs_agg_melt, x="Failure probability", hue="variable", kde=False)
        plt.show()
        
        # Compare the number of cases above a failure threshold
        fails_prob_together = np.empty([len(thresholds),2])
        i=0
        for prc in thresholds: 
            # print(f'The number of observed seasons with failure probability over {prc}% is:', 
            #       len(probs[:,1][probs[:,1]>prc/100]), 'and permuted is: ',
            #       len(probs_perm[:,1][probs_perm[:,1]>prc/100]))
            
            fails_prob_together[i,:] = (len(probs[:,1][probs[:,1]>prc/100]),
                                        len(probs_perm[:,1][probs_perm[:,1]>prc/100]))
            i=i+1
        
        # Create dataframe with all failure probabilities for ordered and permuted cases
        df_fails_prob_together = pd.DataFrame( fails_prob_together, index = thresholds, 
                                              columns = probs_agg.columns)
        
        # Plot figure to compare the amount of cases above the thresholds
        plt.figure(figsize=(6,6), dpi=150)
        sns.lineplot(data=df_fails_prob_together)
        plt.axvline(x=50, alpha=0.5,c='k',linestyle=(0, (5, 5)))
        plt.ylabel('Amount of cases')
        plt.xlabel('Threshold (%)')
        plt.title('Number of cases above a failure prediction level')
        plt.show()
        
        plt.figure(figsize=(6,6), dpi=150)
        sns.lineplot(data=df_fails_prob_together, 
                     y=df_fails_prob_together['Ordered']/df_fails_prob_together['Permuted'], 
                     x = df_fails_prob_together.index)
        plt.ylabel('Ratio Ordered / permuted')
        plt.xlabel('Threshold (%)')
        plt.title('Ratio between ordered and permuted per threshold level')
        plt.show()
        
        sorted_probs = return_period(probs_agg, 'Ordered')
        
        plt.scatter(x=sorted_probs["return-years"], y=sorted_probs["Ordered"]) 
        plt.xscale('log')
        plt.xlabel('years')
        plt.ylabel('Failure probability')
        plt.title("Confidence event and return period")
        plt.show()
        # print("The mean ratio across all thresholds is:", np.mean(df_fails_prob_together['Ordered']/df_fails_prob_together['Permuted']))
    
        return probs_agg, sorted_probs
    
    def probs_bootstrap(df_features_ec_season, probs, size_sample = 10):
        probs_perm = np.empty([len(df_features_ec_season), size_sample])
        for i in range(size_sample): 
            df_bootstrap = df_features_ec_season.apply(np.random.RandomState(seed=i).permutation, axis=0)    
            y_pred_i = brf_model.predict(df_bootstrap)
            probs_i = brf_model.predict_proba(df_bootstrap)
            probs_perm[:,i] = probs_i[:,1]
            
        # Plot ensemble of permutations against ordered data
        df_probs_perm = pd.DataFrame(probs_perm)
        fails_prob_ord = np.empty([len(thresholds),1])
        fails_prob_together = np.empty([len(thresholds),df_probs_perm.shape[1]])
        i=0
        for prc in thresholds: 
            fails_prob_ord[i] = len(probs[:,1][probs[:,1] > prc/100])
            fails_prob_together[i,:] = pd.DataFrame( df_probs_perm.apply(lambda x: x[x > prc/100].count()) ).T
            i=i+1
        
        # Create dataframe with all failure probabilities for ordered and permuted cases
        df_fails_prob_together = pd.DataFrame( fails_prob_together, index = thresholds, 
                                              columns = df_probs_perm.columns)
        # print( np.mean(df_fails_prob_together, axis=1),  np.std(df_fails_prob_together, axis=1))
        
        # plot for ensemble with CI 0.99
        fig, ax = plt.subplots()
        ci = 2.58 * np.std(df_fails_prob_together, axis=1)/np.mean(df_fails_prob_together,axis=1)
        ax.plot(df_fails_prob_together.index,fails_prob_ord )
        ax.plot(df_fails_prob_together.index, df_fails_prob_together.mean(axis=1), '--')
        ax.fill_between(df_fails_prob_together.index, (df_fails_prob_together.mean(axis=1)-ci), (df_fails_prob_together.mean(axis=1)+ci), color='r', alpha=.9)
        ax.set_ylabel('Amount of cases')
        ax.set_xlabel('Threshold (%)')
        ax.set_title(f'Cases above a failure prediction level for ({size_sample} members)')
        plt.show()
        
    # Define the specification of the plot
    thresholds=range(0,101,1)
    
    # PERMUTATION ---------- Preliminary analysis shows that precipitation is such an important variable that it alone can predict a good amount of the failures, irrespective of the other variables
    df_features_ec_season_permuted = df_features_ec_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
        
    # predictions for observed data PD
    y_pred, score_prc, probs, seasons_over_2012 = predictions(brf_model, df_features_ec_season)
    
    # Predictions for permuted
    y_pred_perm, score_prc_perm, probs_perm, seasons_over_2012_perm = predictions(brf_model,df_features_ec_season_permuted)
    
    # Difference between obs. and permuted
    print(f"Permuted failure seasons are: {sum(y_pred_perm)} and ordered are: {sum(y_pred)}. Compound role is {sum(y_pred_perm)/sum(y_pred)}.")
    print("The ratio between predicted failures in observed data and permuted data is:", 
          score_prc / score_prc_perm)
    print("The difference between predicted failures in observed data and permuted data is:", 
          score_prc - score_prc_perm, "\n ")
    
    # plots comparing prediction confidence for obs and perumuted
    probs_agg,sorted_probs = plot_probs_failure(probs, probs_perm)
    
    # check the ensembles to see how likely the ordered values stand wrt to the ensemble
    probs_bootstrap_test = probs_bootstrap(df_features_ec_season, probs)
    
    if df_features_ec_2C_season is None:
        return score_prc
    
    #%% Predictions for 2C degree
    elif df_features_ec_2C_season is not None:
    
        # PERMUTATION ---------- Preliminary analysis shows that precipitation is such an important variable that it alone can predict a good amount of the failures, irrespective of the other variables
        df_features_ec_2C_season_permuted = df_features_ec_2C_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
        
        # predictions for observed data PD
        y_pred_2C, score_prc_2C, probs_2C, seasons_over_2012_2C = predictions(brf_model, df_features_ec_2C_season)
        
        # Predictions for permuted
        y_pred_2C_perm, score_prc_perm_2C, probs_perm_2C, seasons_over_2012_perm_2C = predictions(brf_model, df_features_ec_2C_season_permuted)
                
        # Difference between obs. and permuted
        print("The ratio between predicted failures in observed data and permuted data is:", 
              score_prc_2C / score_prc_perm_2C)
        print("The difference between predicted failures in observed data and permuted data is:", 
              score_prc_2C - score_prc_perm_2C, '\n')
        
        # plots comparing prediction confidence for obs and perumuted
        probs_agg_2C, sorted_probs_2C = plot_probs_failure(probs_2C, probs_perm_2C)
   
        # check the ensembles to see how likely the ordered values stand wrt to the ensemble
        probs_2C_bootstrap_test = probs_bootstrap(df_features_ec_2C_season, probs_2C)
             
        ### Plot comparing 2C and PD return periods
        plt.figure(figsize=(6,6), dpi=150)
        sns.scatterplot(data = sorted_probs, x=sorted_probs["return-years"], 
                        y=sorted_probs["Ordered"], label='PD',linewidth=0 ) 
        sns.scatterplot(data = sorted_probs_2C, x=sorted_probs_2C["return-years"], 
                        y=sorted_probs_2C["Ordered"], label = '2C',linewidth=0 )
        
        if y_pred_2012 is not None:
            plt.axhline(y = y_pred_2012, linestyle = '--', color = 'k', label='2012')
        plt.xscale('log')
        plt.legend(loc="lower right")
        plt.xlabel('years')
        plt.ylabel('Failure probability')
        plt.title("Confidence event and return period for PD and 2C")
        plt.show()
        
        ### Graph comparing ensemble PD for 100 years against 100 years CRU
        probs_cru = brf_model.predict_proba(df_clim_agg_chosen)[:,1]  
        df_probs_cru=pd.DataFrame( probs_cru, columns = ['Ordered'])
        sorted_probs_cru = return_period(df_probs_cru, 'Ordered')
        
        # Reshape to have 100 years each column
        probs_ec_ensemble = np.reshape(probs[:,1], (100,20))
        probs_ec_ensemble = pd.DataFrame(probs_ec_ensemble)
        
        df_ordered = np.empty([100,20])
        df_return_year = np.empty([100,20])
        
        for i in list(probs_ec_ensemble.columns):
            sorted_probs_ec_ensemble = return_period(probs_ec_ensemble, [i])
            df_ordered[:,i] = sorted_probs_ec_ensemble[i]
            df_return_year[:,i] = sorted_probs_ec_ensemble["return-years"]
        
        df_return_year = pd.DataFrame(df_return_year)
        df_ordered = pd.DataFrame(df_ordered)
        
        # Statistics ensemble // If each obs needs to be ploted: # plt.scatter( x=df_return_year, y=df_ordered)
        ord_min = np.min(df_ordered, axis=1)
        ord_max = np.max(df_ordered, axis=1)
        ord_mean = np.mean(df_ordered, axis=1)
        
        plt.figure(figsize=(6,6), dpi=150)
        plt.fill_between(df_return_year[0], ord_min, ord_max,
                         facecolor="orange", # The fill color
                         color='blue',       # The outline color
                         alpha=0.2, label= 'Ensemble')          # Transparency of the fill
        plt.scatter( x=df_return_year[0], y=ord_mean,label = 'Mean ensemble',)
        sns.scatterplot(data = sorted_probs_cru, x=sorted_probs_cru["return-years"], 
                                y=sorted_probs_cru["Ordered"], label = 'CRU',linewidth=0 )
        plt.xscale('log')
        plt.legend(loc="lower right")
        plt.xlabel('years')
        plt.ylabel('Failure probability')
        plt.title("Confidence event and return period for PD and 2C")
        plt.savefig('paper_figures/return_period_ensemble.png', format='png', dpi=150)
        plt.show()

        
        # Compare 2C with PD       
        print("Comparison PD with 2C")
        print("The ratio between failures in 2C and PD is",score_prc_2C/score_prc, "\n")
        print("The increase in failures between 2C and PD is", (score_prc_2C - score_prc)*100,"% \n")
        
        # put them together in the same dataframe for plotting
        probs_agg_t2=pd.DataFrame( [probs[:,1],probs_2C[:,1]]).T
        probs_agg_t2.columns=['Present','2C']
        
        # plots comparing prediction confidence for each arrangement of data
        probs_agg_t2_melt = probs_agg_t2.melt(value_name='Failure probability').assign(data='Density')
        
        # ax = sns.violinplot(data=probs_agg_t2_melt, x="data", y='Failure probability',
        #                     hue='variable', split=True, inner="quartile",bw=.1)
        
        # sns.displot(probs_agg_t2_melt, x="Failure probability",hue="variable", kind="kde", fill='True')
        sns.displot(probs_agg_t2_melt, x="Failure probability",hue="variable",kde=False)
        plt.show()
        # Compare the number of cases above a failure threshold
        fails_prob_together_2C = np.empty([len(thresholds),2])
        i=0
        for prc in thresholds: 
            # print(f'The number of PD seasons with failure probability over {prc}% is:', 
            #       len(probs[:,1][probs[:,1]>prc/100]), 'and 2C is: ',
            #       len(probs_2C[:,1][probs_2C[:,1]>prc/100]))
                 
            fails_prob_together_2C[i,:] = (len(probs[:,1][probs[:,1]>prc/100]),
                                           len(probs_2C[:,1][probs_2C[:,1]>prc/100]))
            i=i+1
        df_fails_prob_together_2C = pd.DataFrame( fails_prob_together_2C, index = thresholds, columns = probs_agg_t2.columns)
        
        # Plot figure ti cinoare the amount of cases above the thresholds
        plt.figure(figsize=(6,6), dpi=150)
        sns.lineplot(data=df_fails_prob_together_2C)
        plt.axvline(x=50, alpha=0.5,c='k',linestyle=(0, (5, 5)))
        plt.ylabel('Amount of cases')
        plt.xlabel('Threshold (%)')
        plt.title('Number of cases above a failure prediction level')
        plt.show()
        
        plt.figure(figsize=(6,6), dpi=150)
        sns.lineplot(data=df_fails_prob_together_2C, y=df_fails_prob_together_2C['2C']/df_fails_prob_together_2C['Present'], x = df_fails_prob_together_2C.index)
        plt.ylabel('Ratio 2C / present')
        plt.xlabel('Threshold (%)')
        plt.title('Ratio between 2C and present for each threshold level')
        plt.show()

        # print("The mean ratio across all thresholds is:", np.mean(df_fails_prob_together_2C['2C']/df_fails_prob_together_2C['Present']))
        
        # Table with occurrences of similar extreme values to 2012
        table_events_prob2012 = pd.DataFrame([[len(seasons_over_2012),len(seasons_over_2012_2C)],
                                              [len(seasons_over_2012_perm),len(seasons_over_2012_perm_2C)]], 
                                             columns = ['PD','2C'], index = ['Ord.','Perm.'])
        print(table_events_prob2012) 
        # Table with scores comparison    
        table_scores = pd.DataFrame( [[score_prc, score_prc_2C, score_prc_2C/score_prc], 
                                      [score_prc_perm, score_prc_perm_2C, score_prc_perm_2C/score_prc_perm],
                                      [score_prc/score_prc_perm,score_prc_2C/score_prc_perm_2C, np.nan ]],
                                        columns = ['PD','2C', '2C/PD'], index = ['Ord.','Perm.', 'Ord./Perm.'] )
        print(table_scores) 
        
        return table_scores, table_events_prob2012

#%%  compound analysis
def compound_exploration(brf_model, df_features_ec_season, df_features_ec_2C_season = None, df_clim_2012 = None): #df_features_ec_3C_season = None 
    """
    This function analyses the climatic features from a compound perspective.
    Its aims are: 1) correlation structure importance in joint occurrences;
    2) evaluating how they change in warmer scenarios.
    
    Parameters: 
    brf_model: the machine learning model trained for the area;
    df_features_ec_season: climatic features (processed by previous function)
    df_features_ec_season_2C = climatic features for future period
        
    Returns: 
    fail_joint_obs, fail_joint_perm, fail_joint_obs_2C, fail_joint_perm_2C.
    The joint occurrence probabilities of the climatic variables given some extreme 
    conditions (percentiles) or compared with the average conditions of seasons 
    with failure (ML generated).
    The variations of the value are wrt to the permutations and climate scenarios.
        
    Created on Wed Feb 10 17:19:09 2021 
    by @HenriqueGoulart
    """
    # PERMUTATION 
    df_features_ec_season_permuted = df_features_ec_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
    
    # Predictions for PD
    y_pred = brf_model.predict(df_features_ec_season)
    
    # Predictions for permuted
    y_pred_perm = brf_model.predict(df_features_ec_season_permuted)
    
    # filtering average + std weather conditions where the prediction points to failure 
    mean_cond = np.mean( df_features_ec_season[y_pred == 1] , axis=0) - np.std( df_features_ec_season[y_pred == 1] , axis=0)
    mean_cond[2] = mean_cond[2] + 2* np.std( df_features_ec_season.iloc[:,2][y_pred == 1] , axis=0)

    def compound_analysis(df_features_ec_season, y_pred):
        # Scatter plot to understand the shape of the variables (correlation)
        df_features_ec_season_fail=pd.concat([
            df_features_ec_season,pd.DataFrame(np.array([y_pred == 1]).T,index=df_features_ec_season.index, columns=['Failure'])], axis=1)
        
        # sns.lmplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[1], 
        #            x=df_features_ec_season.columns[2],fit_reg=True, 
        #            scatter_kws={"s": 10}, hue='Failure',legend_out=False)        
        # plt.vlines(mean_cond[2],ymin = mean_cond[1],ymax=np.max(df_features_ec_season_fail.iloc[:,1]), 
        #            colors ='k', ls='--', alpha=0.6)
        # plt.hlines(mean_cond[1],xmax = mean_cond[2],xmin=np.min(df_features_ec_season_fail.iloc[:,2]),
        #            colors='k', ls='--', alpha=0.6 , label =f'Failure prediction domain')        
        # plt.title("Scatter plot and regression line")
        # plt.show()
        
        # sns.jointplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[1], 
        #               x=df_features_ec_season.columns[2], hue='Failure')
        
        sns.lmplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[0], 
                   x=df_features_ec_season.columns[2],fit_reg=True, 
                   scatter_kws={"s": 10}, hue='Failure',legend_out=False)
        plt.vlines(mean_cond[2],ymin = mean_cond[0], ymax=np.max(df_features_ec_season_fail.iloc[:,0]), 
                   colors ='k', ls='--', alpha=0.6)
        plt.hlines(mean_cond[0],xmax = mean_cond[2], xmin=np.min(df_features_ec_season_fail.iloc[:,2]),
                   colors='k', ls='--', alpha=0.6 , label =f'Failure prediction domain')
        plt.title("Scatter plot and regression line")
        plt.show()
        
        sns.jointplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[0], x=df_features_ec_season.columns[2],hue='Failure')
        
        sns.lmplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[0],
                   x=df_features_ec_season.columns[1],fit_reg=True, 
                   scatter_kws={"s": 10}, hue='Failure',legend_out=False)    
        plt.vlines(mean_cond[1],ymin = mean_cond[0],ymax=np.max(df_features_ec_season_fail.iloc[:,0]), 
                   colors ='k', ls='--', alpha=0.6)
        plt.hlines(mean_cond[0],xmin = mean_cond[1],xmax=np.max(df_features_ec_season_fail.iloc[:,1]),
                   colors='k', ls='--', alpha=0.6 , label =f'Failure prediction domain')
        plt.title("Scatter plot and regression line")
        plt.show()
       
        sns.jointplot(data=df_features_ec_season_fail, y=df_features_ec_season.columns[0], x=df_features_ec_season.columns[1],hue='Failure')
            
        # # 3D - turn off
        # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import   
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # sequence_containing_x_vals = df_features_ec_season.iloc[:,0]
        # sequence_containing_y_vals = df_features_ec_season.iloc[:,1]
        # sequence_containing_z_vals = df_features_ec_season.iloc[:,2]
        # ax.set_xlabel('Max temp')
        # ax.set_ylabel('Diurnal temp. range')
        # ax.set_zlabel('Precipitation')
        # ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
        # plt.show()
            
        # Correlation between variables
        for features_case in [df_features_ec_season,df_features_ec_season[y_pred == 1]]: #1st for all cases, then for only failure seasons
            corrmat = features_case.corr()
            top_corr_features = corrmat.index
            plt.figure(figsize = (6,6), dpi=144)
            g = sns.heatmap(features_case[top_corr_features].corr(),annot=True, cmap="RdYlGn",vmin=-1, vmax=1)
            if len(features_case) == len(df_features_ec_season):
                plt.title("Pearson's correlation for all seasons")
            elif len(features_case) == len(df_features_ec_season[y_pred == 1]):
                plt.title("Pearson's correlation for failure seasons")
            plt.show()
        
       
        # PERCENTILES - Count the frequency of 50, 78, 86 and 90prc for both variables at the same time.
        print("__________________________________________________________________________")
        # print("Percentiles (50,78,86,90):")
        for prc in [50, 78, 86, 90]:
            joint_prc = (sum(np.where( 
                (df_features_ec_season.iloc[:,0] > np.percentile(df_features_ec_season.iloc[:,0],prc)) & 
                (df_features_ec_season.iloc[:,1] > np.percentile(df_features_ec_season.iloc[:,1],prc)) & 
                (df_features_ec_season.iloc[:,2] < np.percentile(df_features_ec_season.iloc[:,2],(100-prc))), 1, 0)))
            
            # print(f"(>{prc};<{100-prc}): Ratio of joint-occurrence ({joint_prc}) wrt independent variables ({round((len(df_features_ec_season)*((100-prc)/100)**3),3)}) is:",
            #       round(joint_prc/(len(df_features_ec_season)*((100-prc)/100)**3),3))
            # print(f"(>{prc};<{100-prc}): Ratio of joint-occurrences per univariate extreme level is:",
            #       round(joint_prc/ len(df_features_ec_season[df_features_ec_season.iloc[:,1] > np.percentile(df_features_ec_season.iloc[:,1],prc)]),3), '\n')
            
        # MEAN FAILURE CONDITIONS - Define each percentile
        print("Comparison with mean (+std) conditions of failure seasons:")
        from scipy import stats
        for feature in df_features_ec_season.columns:
            percentile = stats.percentileofscore(df_features_ec_season[feature], mean_cond[feature])
            print(f"The percentile of variable {feature} for mean failure conditions is:",percentile )
        
        # Occurences of conditions for mean value of each variable
        fail_tmx = len(df_features_ec_season[df_features_ec_season.iloc[:,0] > mean_cond.iloc[0]]) # tmax
        fail_dtr = len(df_features_ec_season[df_features_ec_season.iloc[:,1] > mean_cond.iloc[1]]) # dtr
        fail_pre = len(df_features_ec_season[df_features_ec_season.iloc[:,2] < mean_cond.iloc[2]]) #prec
        
        # Comparison simultaneous occurences with univariate extremes and 2012 conditions 
        fail_joint = sum( np.where( 
            (df_features_ec_season.iloc[:,0] >  mean_cond.iloc[0]) & 
            (df_features_ec_season.iloc[:,1] >  mean_cond.iloc[1]) &
            (df_features_ec_season.iloc[:,2] < mean_cond.iloc[2]), 1, 0) )
        
        fail_or = sum( np.where( 
            (df_features_ec_season.iloc[:,0] >  mean_cond.iloc[0]) | 
            (df_features_ec_season.iloc[:,1] >  mean_cond.iloc[1]) |
            (df_features_ec_season.iloc[:,2] < mean_cond.iloc[2]), 1, 0) )
        
        print("Number of extreme conditions joint occurrences for all mean failure conditions are:",fail_joint, 
              "and for either one of the conditions:", fail_or)
        # print("Ratio of joint occurrences by extreme univariate occurrences (pre):",fail_joint/fail_pre)
        # print("Ratio of joint occurrences by extreme univariate occurrences (dtr):",fail_joint/fail_dtr)
        # print("Ratio of joint occurrences by extreme univariate occurrences (tmx):",fail_joint/fail_tmx)
        
        if df_clim_2012 is not None:
            JO_fail_2012 = sum( np.where( 
                (df_features_ec_season.iloc[:,0] >=  df_clim_2012.iloc[0]) & 
                (df_features_ec_season.iloc[:,1] >=  df_clim_2012.iloc[1]) &
                (df_features_ec_season.iloc[:,2] <= df_clim_2012.iloc[2]), 1, 0) )
            print("\n Number of seasons with conditions equal or more restrict than the 2012 season:", JO_fail_2012)
            
        print("__________________________________________________________________________","\n")
        
        return fail_joint, fail_or, JO_fail_2012
    
    print("- ORDERED")
    fail_joint_obs, fail_or_obs, JO_fail_2012_obs = compound_analysis(df_features_ec_season,y_pred)
    
    # permutation joint analysis
    print("- PERMUTED")
    fail_joint_perm, fail_or_perm, JO_fail_2012_perm = compound_analysis(df_features_ec_season_permuted,y_pred_perm)
    
    if df_features_ec_2C_season is None:
        return fail_joint_obs, fail_joint_perm
    
    #%% Compound analysis 2C
    if df_features_ec_2C_season is not None:
        
        # PERMUTATION 2C
        df_features_ec_2C_season_permuted = df_features_ec_2C_season.apply(np.random.RandomState(seed=1).permutation, axis=0)  
        
        # Predictions for 2C and permuted 2C
        y_pred_2C = brf_model.predict(df_features_ec_2C_season)
        y_pred_2C_perm = brf_model.predict(df_features_ec_2C_season_permuted)
        
        # mean conditions
        dif_mean =  np.mean( df_features_ec_2C_season[y_pred_2C == 1] , axis=0) - np.mean( df_features_ec_season[y_pred == 1] , axis=0)
        print("Mean conditions 2C - PD is: \n", dif_mean)
        # climate data original structure and permuted joint analysis
        print("2C ----------------------")
        fail_joint_obs_2C, fail_or_obs_2C, JO_fail_2012_obs_2C = compound_analysis(df_features_ec_2C_season,y_pred_2C)
        print("2C permuted ---------------------")
        fail_joint_perm_2C, fail_or_perm_2C, JO_fail_2012_perm_2C = compound_analysis(df_features_ec_2C_season_permuted, y_pred_2C_perm)
        
        # Amount of times more frequent 
        print("\n Final comparisons: \n")
        print("Ratio between PD correlated and permuted joint occurrences is",fail_joint_obs/fail_joint_perm)
        print("Ratio between 2C correlated and permuted joint occurrences is",fail_joint_obs_2C/fail_joint_perm_2C)
        print("Ratio between 2C and PD correlated joint occurrences",fail_joint_obs_2C/fail_joint_obs)
        print("Ratio between 2C and PD permuted joint occurrences",fail_joint_perm_2C/fail_joint_perm)
        
        # Table with all values indicating joint, univariate cases and predicted values
        scenario_columns = ['PD','PD perm','2C','2C perm']
        scenario_index = ['Joint (and)', 'RF', 'Joint + univariate (or)']
        df_joint =  pd.DataFrame([[fail_joint_obs, fail_joint_perm, fail_joint_obs_2C, fail_joint_perm_2C]],
                                 columns = scenario_columns)
        df_or =  pd.DataFrame([[fail_or_obs, fail_or_perm, fail_or_obs_2C, fail_or_perm_2C]], 
                              columns = scenario_columns)
        df_rf = pd.DataFrame([[sum(y_pred), sum(y_pred_perm), sum(y_pred_2C), sum(y_pred_2C_perm)]], 
                             columns = scenario_columns)
        
        df_joint_or_rf = pd.DataFrame(pd.concat([df_joint, df_rf, df_or], axis = 0))
        df_joint_or_rf.index = scenario_index
        df_joint_or_rf['PD/ PD perm']= df_joint_or_rf['PD']/df_joint_or_rf['PD perm']
        df_joint_or_rf['2C/ 2C perm'] = df_joint_or_rf['2C']/df_joint_or_rf['2C perm']
        
        # Plot graphs comparing the difference between 2C and PD
        for (df_features_ec_season_1, df_features_ec_2C_season_1, case) in zip(
                [df_features_ec_season, df_features_ec_season_permuted],
                [df_features_ec_2C_season, df_features_ec_2C_season_permuted],
                ['ord','perm']):
            
            df_features_ec_season_fail_PD =pd.concat([
                df_features_ec_season_1,
                pd.DataFrame(np.array([y_pred < -1]).T, index=df_features_ec_season_1.index, columns=['Scenario'])],axis=1)
            
            df_features_ec_season_fail_PD['Scenario'] = 'PD'
            df_features_ec_season_fail_PD['Scenario'][y_pred == 1] = 'Failure PD'
            
            df_features_ec_season_fail_2C =pd.concat([
                df_features_ec_2C_season_1,
                pd.DataFrame(np.array([y_pred_2C > -1 ]).T, index=df_features_ec_2C_season_1.index,columns=['Scenario'])],axis=1)
            
            df_features_ec_season_fail_2C['Scenario'] = '2C'
            df_features_ec_season_fail_2C['Scenario'][y_pred_2C == 1] = 'Failure 2C'
                        
            df_features_ec_season_scenarios = pd.concat([
                df_features_ec_season_fail_PD, df_features_ec_season_fail_2C], axis= 0)
            
            for (y_axis, x_axis) in zip([0,0],[2,1]): 
                # sns.lmplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis], 
                #            x=df_features_ec_season.columns[x_axis],fit_reg=True, 
                #            scatter_kws={"s": 10}, hue='Scenario',legend_out=False)
                # plt.title("Climatic variables for each scenario")
                if case == 'ord':
                    name_plot = "plot_ord_" + str(df_features_ec_season.columns[x_axis]) + str(df_features_ec_season.columns[y_axis]) 
                elif case == 'perm':
                    name_plot = "plot_perm_" + str(df_features_ec_season.columns[x_axis]) + str(df_features_ec_season.columns[y_axis]) 
                
                print(name_plot)
                sns.jointplot(data=df_features_ec_season_scenarios, y = df_features_ec_season.columns[y_axis],
                              x=df_features_ec_season.columns[x_axis], kind="kde", 
                              palette=["#92C6FF", "#fabbff", "#97F0AA","#FF9F9A"],
                              hue='Scenario',fill=True, joint_kws = {'alpha': 0.7},
                              hue_order= ['PD','2C','Failure PD','Failure 2C'])
                plt.savefig('paper_figures/{}.png'.format(name_plot), dpi=150)
                plt.show()
                # plt.title("Climatic variables for each scenario")
        
        # ONLY FAILURES
        print("FOR FAILURES ONLY: \n")
        df_features_ec_fail_season_PD = df_features_ec_season[y_pred == 1]      
        df_features_ec_fail_season_2C = df_features_ec_2C_season[y_pred_2C == 1]

        df_features_ec_fail_season_PD_permuted = df_features_ec_season_permuted[y_pred_perm == 1]
        df_features_ec_fail_season_2C_permuted = df_features_ec_2C_season_permuted[y_pred_2C_perm == 1]     
        
        # Amount of times more frequent 
        print("\n Final comparisons for failures dataset: \n")
        print("Ratio between correlated and permuted (PD) joint occurrences is", len(df_features_ec_fail_season_PD)/len(df_features_ec_fail_season_PD_permuted))
        print("Ratio between correlated and permuted (2C) joint occurrences is",len(df_features_ec_fail_season_2C)/len(df_features_ec_fail_season_2C_permuted))
        print("Ratio between 2C and PD correlated joint occurrences",len(df_features_ec_fail_season_2C)/len(df_features_ec_fail_season_PD))
        print("Ratio between 2C and PD permuted joint occurrences",len(df_features_ec_fail_season_2C_permuted)/len(df_features_ec_fail_season_PD_permuted))
            
        # Plot graphs comparing the difference between 2C and PD
        for (df_features_ec_season_1,df_features_ec_2C_season_1, case) in zip([
                df_features_ec_fail_season_PD, df_features_ec_fail_season_PD_permuted],
                [df_features_ec_fail_season_2C, df_features_ec_fail_season_2C_permuted],
                ['ord','perm']):
            
            df_features_ec_season_fail_PD =pd.concat([
                df_features_ec_season_1,pd.DataFrame(np.zeros(len(df_features_ec_season_1) ).T,
                                                     index=df_features_ec_season_1.index, columns=['Scenario'])],axis=1)
            df_features_ec_season_fail_PD['Scenario'] = 'Present day'
            
            df_features_ec_season_fail_2C =pd.concat([
                df_features_ec_2C_season_1,pd.DataFrame(np.ones(len(df_features_ec_2C_season_1) ).T,
                                                        index=df_features_ec_2C_season_1.index, columns=['Scenario'])],axis=1)
            df_features_ec_season_fail_2C['Scenario'] = '2C'
            
            df_features_ec_season_scenarios = pd.concat([
                df_features_ec_season_fail_PD,df_features_ec_season_fail_2C], axis= 0)
            
            for (y_axis, x_axis) in zip([0,0],[2,1]):
                if case == 'ord':
                    name_plot = "plot_ord_fail_" + str(df_features_ec_season.columns[x_axis]) + str(df_features_ec_season.columns[y_axis]) 
                elif case == 'perm':
                    name_plot = "plot_perm_fail_" + str(df_features_ec_season.columns[x_axis]) + str(df_features_ec_season.columns[y_axis]) 
                
                # sns.lmplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis], 
                #            x=df_features_ec_season.columns[x_axis],fit_reg=True, 
                #            scatter_kws={"s": 10}, hue='Scenario',legend_out=False)
                # plt.title("Climatic variables for each scenario")
                
                sns.jointplot(data=df_features_ec_season_scenarios, y=df_features_ec_season.columns[y_axis],
                              x=df_features_ec_season.columns[x_axis], kind="kde",
                              hue='Scenario',fill=True, joint_kws={'alpha': 0.7})    
                plt.savefig('paper_figures/{}.png'.format(name_plot), dpi=150)
                plt.show()
                # plt.title("Climatic variables for each scenario")
        
        # Table with occurrences of similar extreme values to 2012
        table_JO_prob2012 = pd.DataFrame([[JO_fail_2012_obs,JO_fail_2012_obs_2C],
                                              [JO_fail_2012_perm,JO_fail_2012_perm_2C]], 
                                             columns = ['PD','2C'], index = ['Ord.','Perm.'])
        print(df_joint_or_rf)
        print(table_JO_prob2012) 
        return df_joint_or_rf, table_JO_prob2012


#%%  compound analysis
def return_period_storyline(df_features_ec_season, df_features_ec_2C_season, df_jo_2012, brf_model, df_clim_2012): #df_features_ec_3C_season = None 
    """
    Return period graph generation accounting for storyline joint occurrence values
    
    Parameters: 
    brf_model: the machine learning model trained for the area;
    df_features_ec_season: climatic features (processed by previous function)
    df_features_ec_season_2C = climatic features for future period
    df_clim_2012 = climatic variables for 2012 season
       
    Returns: 
    Return period plot with storyline years .
        
    Created on Wed Feb 10 17:19:09 2021 
    by @HenriqueGoulart
    """
    # Predictions for Observed data PD
    thresholds=range(0,101,1)
    y_pred_2012 = brf_model.predict_proba(df_clim_2012.values.reshape(1, -1))[0][1]
    def predictions(brf_model,df_features_ec_season):
        
        y_pred = brf_model.predict(df_features_ec_season)
        score_prc = sum(y_pred)/len(y_pred) 
        print("\n The total failures are:", sum(y_pred),
              " And the ratio of failure seasons by total seasons is:", score_prc, "\n")     
        probs = brf_model.predict_proba(df_features_ec_season)        
        if df_clim_2012 is not None:
            limit_2012 = y_pred_2012
        else: 
            limit_2012 = 0.95
        seasons_over_2012 = df_features_ec_season[probs[:,1]>=limit_2012]
        print(f"\n Number of {limit_2012}% events: {len(seasons_over_2012)} and mean conditions are:", 
              np.mean(seasons_over_2012))
        
        return y_pred, score_prc, probs, seasons_over_2012
    
    def return_period(df, colname):

        # Sort data smallest to largest
        sorted_data = df.sort_values(by=colname)
        # Count total obervations
        n = sorted_data.shape[0]      
        # Add a numbered column 1 -> n to use in return calculation for rank
        sorted_data.insert(0, 'rank', range(1, 1 + n))        
        # Calculate probability
        sorted_data["probability"] = (n - sorted_data["rank"] + 1) / (n + 1)        
        # Calculate return - yearly data, no need to further trnasform
        sorted_data["return-years"] = (1 / sorted_data["probability"])
    
        return(sorted_data)
    
    def plot_probs_failure(probs, probs_perm):
        
        # put them together in the same dataframe for plotting
        probs_agg=pd.DataFrame( [probs[:,1],probs_perm[:,1]]).T
        probs_agg.columns=['Ordered','Permuted']
    
        # plots comparing prediction confidence for obs and perumuted
        probs_agg_melt = probs_agg.melt(value_name='Failure probability').assign(data='Density')
        
        # Compare the number of cases above a failure threshold
        fails_prob_together = np.empty([len(thresholds),2])
        i=0
        for prc in thresholds: 
            # print(f'The number of observed seasons with failure probability over {prc}% is:', 
            #       len(probs[:,1][probs[:,1]>prc/100]), 'and permuted is: ',
            #       len(probs_perm[:,1][probs_perm[:,1]>prc/100]))
            
            fails_prob_together[i,:] = (len(probs[:,1][probs[:,1]>prc/100]),
                                        len(probs_perm[:,1][probs_perm[:,1]>prc/100]))
            i=i+1
        
        # Create dataframe with all failure probabilities for ordered and permuted cases
        df_fails_prob_together = pd.DataFrame( fails_prob_together, index = thresholds, 
                                              columns = probs_agg.columns)
         
        sorted_probs = return_period(probs_agg, 'Ordered')

        return probs_agg, sorted_probs
    
    # PERMUTATION ---------- Preliminary analysis shows that precipitation is such an important variable that it alone can predict a good amount of the failures, irrespective of the other variables
    df_features_ec_season_permuted = df_features_ec_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
        
    # predictions for observed data PD
    y_pred, score_prc, probs, seasons_over_2012 = predictions(brf_model, df_features_ec_season)
    
    # Predictions for permuted
    y_pred_perm, score_prc_perm, probs_perm, seasons_over_2012_perm = predictions(brf_model,df_features_ec_season_permuted)

    # PERMUTATION ---------- Preliminary analysis shows that precipitation is such an important variable that it alone can predict a good amount of the failures, irrespective of the other variables
    df_features_ec_2C_season_permuted = df_features_ec_2C_season.apply(np.random.RandomState(seed=1).permutation, axis=0)    
    
    # predictions for observed data PD
    y_pred_2C, score_prc_2C, probs_2C, seasons_over_2012_2C = predictions(brf_model, df_features_ec_2C_season)
    
    # Predictions for permuted
    y_pred_2C_perm, score_prc_perm_2C, probs_perm_2C, seasons_over_2012_perm_2C = predictions(brf_model, df_features_ec_2C_season_permuted)
                
    # plots comparing prediction confidence for obs and perumuted
    probs_agg,sorted_probs = plot_probs_failure(probs, probs_perm)
       
     # plots comparing prediction confidence for obs and perumuted
    probs_agg_2C, sorted_probs_2C = plot_probs_failure(probs_2C, probs_perm_2C)
            
    # Return period for joint occurrence
    JO_PD_rp = 2000/df_jo_2012.iloc[0,0]
    JO_2C_rp = 2000/df_jo_2012.iloc[0,1]
    RF_PD_rp = 2000/df_jo_2012.iloc[0,0]
    JO_2C_rp = 2000/df_jo_2012.iloc[0,1]
    print(df_jo_2012.iloc[0,0],df_jo_2012.iloc[0,1], JO_PD_rp, JO_2C_rp)
    ### Plot comparing 2C and PD return periods
    plt.figure(figsize=(6,6), dpi=150)
    sns.scatterplot(data = sorted_probs, x=sorted_probs["return-years"], 
                    y=sorted_probs["Ordered"], label='PD',linewidth=0 ) 
    sns.scatterplot(data = sorted_probs_2C, x=sorted_probs_2C["return-years"], 
                    y=sorted_probs_2C["Ordered"], label = '2C',linewidth=0 )
    
    # closest_2012 = np.searchsorted(sorted_probs_2C['Ordered'], y_pred_2012, side='left')
    # closest_2012 = sorted_probs_2C[sorted_probs_2C['rank']==closest_2012]
    
    plt.scatter(x=JO_PD_rp, y=y_pred_2012, label = 'JO 2012 PD', linewidth=1 )
    plt.scatter(x=JO_2C_rp, y=y_pred_2012, label = 'JO 2012 2C', linewidth=1 )

    if y_pred_2012 is not None:
        plt.axhline(y = y_pred_2012, linestyle = '--', color = 'k', label='2012')
    plt.xscale('log')
    plt.legend(loc="lower right")
    plt.xlabel('years')
    plt.ylabel('Failure probability')
    plt.title("Confidence event and return period for PD and 2C")
    plt.savefig('paper_figures/return_period_storyline.png', format='png', dpi=150)
    plt.show()
    
    
    ### Graph comparing ensemble PD for 100 years against 100 years CRU
    
    # Reshape to have 100 years each column
    def return_year_ensemble(probs):
        probs_ec_ensemble = np.reshape(probs[:,1], (100,20))
        probs_ec_ensemble = pd.DataFrame(probs_ec_ensemble)
        
        df_ordered = np.empty([100,20])
        df_return_year = np.empty([100,20])
        
        for i in list(probs_ec_ensemble.columns):
            sorted_probs_ec_ensemble = return_period(probs_ec_ensemble, [i])
            df_ordered[:,i] = sorted_probs_ec_ensemble[i]
            df_return_year[:,i] = sorted_probs_ec_ensemble["return-years"]
        
        df_return_year = pd.DataFrame(df_return_year)
        df_ordered = pd.DataFrame(df_ordered)
        return df_ordered, df_return_year
    
    df_ordered_PD, df_return_year_PD = return_year_ensemble(probs)
    df_ordered_2C, df_return_year_2C = return_year_ensemble(probs_2C)   
    

    # Statistics ensemble // If each obs needs to be ploted: # plt.scatter( x=df_return_year, y=df_ordered)
    
    ord_mean = np.median(df_ordered_PD, axis=1)
    ord_min = np.percentile(df_ordered_PD, 5, axis=1)
    ord_max = np.percentile(df_ordered_PD, 95, axis=1)
    
    
    ord_2C_min = np.percentile(df_ordered_2C, 5, axis=1)
    ord_2C_max = np.percentile(df_ordered_2C, 95, axis=1)
    ord_2C_mean = np.mean(df_ordered_2C, axis=1)
    
    
    plt.figure(figsize=(6,6), dpi=150)
    
    plt.fill_between(df_return_year_PD[0], ord_min, ord_max,
                     facecolor="blue", # The fill color
                     color='blue',       # The outline color
                     alpha=0.2, label= '5-95prc Ensemble PD')   
    
    plt.fill_between(df_return_year_2C[0], ord_2C_min, ord_2C_max,
                     facecolor="orange", # The fill color
                     color='orange',       # The outline color
                     alpha=0.2, label= '5-95prc Ensemble 2C')        # Transparency of the fill
    plt.scatter( x=df_return_year_PD[0], y=ord_mean,label = 'Median ensemble PD',)
    plt.scatter( x=df_return_year_2C[0], y=ord_2C_mean,label = 'Median ensemble 2C',)
    
    plt.xscale('log')
    plt.legend(loc="lower right")
    plt.xlabel('years')
    plt.ylabel('Failure probability')
    plt.title("Confidence event and return period for PD and 2C")
    plt.savefig('paper_figures/return_period_ensemble_2C_PD.png', format='png', dpi=150)
    plt.show()
    
    
    