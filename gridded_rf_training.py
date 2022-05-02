# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 08:55:02 2021

@author: morenodu
"""


def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


DS_y_det_grid = xr.DataArray( detrend_dim(DS_y["yield"], 'time') + DS_y["yield"].mean('time'), name= DS_y["yield"].name, attrs = DS_y["yield"].attrs)
DS_y_det_grid = DS_y_det_grid.transpose("lat", "lon","time")
DS_y_det_grid = DS_y_det_grid.reindex(lat=DS_y_det_grid.lat[::-1])


df_epic_det_grid = DS_y_det_grid.to_dataframe().dropna()

# Check if it works:
DS_y['yield'].groupby('time').mean(...).plot()
DS_y_det_grid.groupby('time').mean(...).plot()



# Detrend Dataset
def detrend_dataset(DS):
    px= DS.polyfit(dim='time', deg=1)
    fitx = xr.polyval(DS['time'], px)
    dict_name = dict(zip(list(fitx.keys()), list(DS.keys())))
    fitx = fitx.rename(dict_name)
    DS_det  = (DS - fitx) + DS.mean('time')
    return DS_det

# Select data according to months
def is_month(month, ref_in, ref_out):
    return (month >= ref_in) & (month <= ref_out)

DS_cli_us_det = DS_cli_us # detrend_dataset(DS_cli_us)


# Check if it works:
DS_cli_us.dtr.groupby('time').mean(...).plot()
DS_cli_us_det.dtr.groupby('time').mean(...).plot()


# Select months
DS_cli_us_det_season = DS_cli_us_det.sel(time=is_month(DS_cli_us_det['time.month'], 7,8))

DS_cli_us_det_season = DS_cli_us_det_season.groupby('time.year').mean('time')
DS_cli_us_det_season = DS_cli_us_det_season.rename({'year':'time'})
DS_cli_us_det_season = DS_cli_us_det_season.reindex(lat=DS_cli_us_det_season.lat[::-1])


DS_cli_us_det_season_det2 = detrend_dataset(DS_cli_us_det_season)

DS_cli_us_det_season.dtr.groupby('time').mean(...).plot()
df_clim_agg_chosen['dtr_7_8'].plot()
DS_cli_us_det_season_det2.dtr.groupby('time').mean(...).plot()

df_cli_us_det_season_grid = DS_cli_us_det_season_det2.to_dataframe().dropna()



# ML feature exploration, importance and selection if needed
plot_feat_imp_grid = feature_importance_selection(df_cli_us_det_season_grid, df_epic_det_grid)

# Main algorithm for training the ML model
df_cli_us_det_season_grid.columns = ['Temperature (°C)', 'Precipitation (mm/month)','DTR (°C)']
brf_model_us_grid, fig_dep_us_grid, rf_scores_us_grid = failure_probability(df_cli_us_det_season_grid, df_epic_det_grid, show_partial_plots= True, model_choice = 'conservative')



