import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from mask_shape_border import mask_shape_border

# Crop space to either US or soy states
usa = gpd.read_file('gadm36_USA_1.shp', crs="epsg:4326") 
us1_shapes = list(shpreader.Reader('gadm36_USA_1.shp').geometries())
state_names = ['Iowa','Illinois','Minnesota','Indiana','Nebraska','Ohio','South Dakota','North Dakota','Missouri','Arkansas']
soy_us_states = usa[usa['NAME_1'].isin( state_names)]

# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_br_names = ['Mato Grosso','Goiás']
soy_br_states = bra[bra['NAME_1'].isin(state_br_names)]

# Crop space to either BR or soy states
bra = gpd.read_file('gadm36_BRA_1.shp', crs="epsg:4326") 
br1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
state_br_names = ['Rio Grande do Sul','Paraná']
soy_br_states_2 = bra[bra['NAME_1'].isin(state_br_names)]

# Crop space to either ARG or soy states
arg = gpd.read_file('gadm36_ARG_1.shp', crs="epsg:4326") 
ar1_shapes = list(shpreader.Reader('gadm36_ARG_1.shp').geometries())
state_ar_names = ['Buenos Aires','Santa Fe', 'Córdoba'] #'Rio Grande do Sul','Paraná' #'Mato Grosso','Goiás' 
soy_ar_states = arg[arg['NAME_1'].isin(state_ar_names)]

#%% yield model - WOFOST
DS_plant=xr.open_dataset("cgms-wofost_wfdei_nobc_hist_default_noirr_plant-day_soy_global_annual_1979_2012.nc",decode_times=False).sel(time=slice(1,31))
DS_maty=xr.open_dataset("cgms-wofost_wfdei_nobc_hist_default_noirr_maty-day_soy_global_annual_1979_2012.nc",decode_times=False).sel(time=slice(1,31))
DS_biom=xr.open_dataset("cgms-wofost_wfdei_nobc_hist_default_noirr_biom_soy_global_annual_1979_2012.nc",decode_times=False).sel(time=slice(1,31))
DS_anth=xr.open_dataset("cgms-wofost_wfdei_nobc_hist_default_noirr_anth-day_soy_global_annual_1979_2012.nc",decode_times=False).sel(time=slice(1,31))
DS_aet=xr.open_dataset("cgms-wofost_wfdei_nobc_hist_default_noirr_aet_soy_global_annual_1979_2012.nc",decode_times=False).sel(time=slice(1,31))


#%% function
def clean_unique_values(DS_plant, mask):
    DS_plant['time'] = pd.to_datetime(list(range(1980, 2011)), format='%Y').year
    DS_plant = mask_shape_border(DS_plant,mask) #US_shape
    DS_plant = DS_plant.dropna(dim = 'lon', how='all')
    DS_plant = DS_plant.dropna(dim = 'lat', how='all')
    da_plant = DS_plant[list(DS_plant.keys())[0]]    # automatize the var selection inside the dataset (one variable only)
    da_plant = da_plant/30.5
    plt.figure(figsize=(20,10)) #plot clusters
    ax=plt.axes(projection=ccrs.Mercator())
    da_plant.mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=False,cbar_kwargs={'label': 'Planting date'})
    if mask.iloc[0]['NAME_0'] == 'Brazil':
        shape = br1_shapes
        borders = [-75,-32,-35,2]
    elif mask.iloc[0]['NAME_0'] == 'United States':
        shape = us1_shapes
        borders =[-125,-67,24,50]
    elif mask.iloc[0]['NAME_0'] == 'Argentina':
        shape = ar1_shapes
        borders =[-75,-55,-20,-55]
    ax.add_geometries(shape, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent(borders, ccrs.PlateCarree())
    plt.show()
    
    planting_dates = np.unique(da_plant.mean('time').values)
    planting_dates = (planting_dates[planting_dates>0])
    return(planting_dates)

#%%
planting_dates=clean_unique_values(DS_plant,soy_us_states)
anthesis_dates=clean_unique_values(DS_anth,soy_us_states)
maturity_dates=clean_unique_values(DS_maty,soy_us_states)
anth_us =(planting_dates + anthesis_dates.mean()).mean()
matu_us = (planting_dates + maturity_dates.mean()).mean()
print(planting_dates)
print("Anthesis date is: ",anth_us)
print("Maturity date is: ",matu_us)
print("duration sub-season: ", matu_us-anth_us)

planting_dates_br=clean_unique_values(DS_plant,soy_br_states)
anthesis_dates_br=clean_unique_values(DS_anth,soy_br_states)
maturity_dates_br=clean_unique_values(DS_maty,soy_br_states)
anth_br1= (planting_dates_br + anthesis_dates_br.mean()).mean()
matu_br1 = (planting_dates_br + maturity_dates_br.mean()).mean()-12
print(planting_dates_br)
print("Anthesis date is: ",anth_br1)
print("Maturity date is: ",matu_br1)
print("duration sub-season: ",matu_br1+12-anth_br1)


planting_dates_br_2=clean_unique_values(DS_plant,soy_br_states_2)
anthesis_dates_br_2=clean_unique_values(DS_anth,soy_br_states_2)
maturity_dates_br_2=clean_unique_values(DS_maty,soy_br_states_2)
anth_br2= (planting_dates_br_2 + anthesis_dates_br_2.mean()).mean()
matu_br2 = (planting_dates_br_2 + maturity_dates_br_2.mean()).mean()-12
print(planting_dates_br_2)
print("Anthesis date is: ",anth_br2)
print("Maturity date is: ",matu_br2)
print("duration sub-season: ",matu_br2+12-anth_br2)

planting_dates_ar=clean_unique_values(DS_plant,soy_ar_states)
anthesis_dates_ar=clean_unique_values(DS_anth,soy_ar_states)
maturity_dates_ar=clean_unique_values(DS_maty,soy_ar_states)
anth_ar= (planting_dates_ar + anthesis_dates_ar.mean()).mean()
matu_ar =(planting_dates_ar + maturity_dates_ar.mean()).mean()-12
print(planting_dates_ar)
print("Anthesis date is: ",anth_ar)
print("Maturity date is: ",matu_ar)
print("duration sub-season: ",matu_ar+12-anth_ar)



# actual_growing_season_dates=clean_unique_values(DS_aet)
# biomass=clean_unique_values(DS_biom)
## months: November, December, Jan, Fev, Mar