import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, Point, mapping
from mask_shape_border import mask_shape_border
#%% Adding state-level contours for Brazil to be plotted with the rasters
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
'''
# Import brazilian national border and limit all results to it # lon_br, lat_br = g[0].boundary[811].xy ##Brazilian mainland border
# br_shape = gpd.read_file('gadm36_BRA_0.shp')
# g = [i for i in br_shape.geometry]
# df = pd.DataFrame(data=[lon_br,lat_br],index= ['Lon','Lat'])
# df = df.T
# df.to_csv('Brazil_borders.csv')
# df = pd.read_csv('Brazil_borders.csv',index_col=0)
 '''
#%% adding the netcdf files of temperature, precipitation and evaporation (ERA5)
DS_cli=xr.open_dataset("temp_evp_prec_era5_monthly.nc").sel(time=slice('1979-09-01','2012-03-31'),longitude=slice(-61,-44),latitude=slice(-5,-33))
clipped= mask_shape_border(DS_cli,'gadm36_BRA_0.shp' )
#%%  Add growing year and month with respect to the harvest season of soybean 
DS_cli=clipped # NEW DS THAT IS CLIPPED
DS_cli['t2m']=DS_cli.t2m -273.15
DS_cli['tp']=DS_cli.tp * 1000
DS_cli['e']=DS_cli.e * 1000
DS_cli.t2m.attrs = {'units': 'Celcius degree', 'long_name': '2 metre temperature'}
DS_cli.tp.attrs = {'units': 'mm', 'long_name': 'Total precipitation'}
DS_cli.e.attrs = {'units': 'mm of water equivalent', 'long_name': 'Evaporation', 'standard_name': 'lwe_thickness_of_water_evaporation_amount'}

DS_cli['growing_month'] = DS_cli["time.month"]
DS_cli['growing_year'] = DS_cli["time.year"]
for i in range(len(DS_cli["time"].values)):
    if DS_cli["time.month"].values[i] < 9:
        DS_cli['growing_year'][i] = DS_cli["time.year"][i]
    else:
        DS_cli["growing_year"][i] = DS_cli["time.year"][i]+1    
DS_cli['growing_month'].values = np.array(list(np.arange(1,8))*(DS_cli['time.year'].values[-1]-DS_cli['time.year'].values[0]))

DS_cli_year = DS_cli.groupby(DS_cli['growing_year']).mean(keep_attrs=True)

#%% Boxplot: difference between the 2012 season and the historical collection of data
df_cli= DS_cli.to_dataframe().groupby(['time']).mean()
year = 2005
df_cli_h=df_cli.loc[df_cli.growing_year != year ]
df_cli_2012=df_cli.loc[df_cli.growing_year == year]

for i in ["t2m","tp","e"]:
    ax = df_cli_h.boxplot(column=i, by="growing_month" , widths = 0.3, patch_artist = False, figsize=(12,8), color = "blue")
    ax = plt.scatter(data=df_cli_2012, x="growing_month", y=i , c="red")
    red_patch = mpatches.Patch(color='red', label=f'{year} season')
    blue_patch = mpatches.Patch(color='blue', label='Historical data')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

#%% Mean and standard deviation
map_proj3 = ccrs.PlateCarree()
plt.figure(figsize=(30, 30))
delta= DS_cli.sel(time=slice('1979-09-01','2011-03-31')).groupby('growing_month').mean(keep_attrs=True)

for i in ('t2m','tp','e'):
    dym=getattr(delta,i)
    fig=dym.isel(growing_month=slice(0,7,1)).plot(x='longitude', y='latitude', col='growing_month', col_wrap=7, subplot_kws={'projection': map_proj3},robust=True, cbar_kwargs={'label': dym.attrs['units']},cmap='Reds')
    for ax in fig.axes.flat:
        ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
        ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())      
    fig.fig.tight_layout()
    
map_proj3 = ccrs.PlateCarree()
# plt.figure(figsize=(30, 30))
delta= DS_cli.sel(time=slice('1979-09-01','2011-03-31')).groupby('growing_month').std(keep_attrs=True)

for i in ('t2m','tp','e'):
    dym=getattr(delta,i)
    fig=dym.isel(growing_month=slice(0,7,1)).plot(x='longitude', y='latitude', col='growing_month', col_wrap=7, subplot_kws={'projection': map_proj3},robust=True, cbar_kwargs={'label': dym.attrs['units']},cmap='RdPu')
    for ax in fig.axes.flat:
        ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
        ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())      
    fig.fig.tight_layout()



#%% 2012 x history - cell-level resolution (for each cell in the grid and for each month, compose a boxplot along years and then compare with the 2012 season )
    
result= DS_cli.sel(time=slice('2011-09-01','2012-03-31')).groupby('growing_month').mean() - DS_cli.sel(time=slice('1979-09-01','2011-03-31')).groupby('growing_month').mean()
map_proj3 = ccrs.PlateCarree()

plt.figure(figsize=(30, 30))

fig = result.t2m.isel(growing_month=slice(0,7,1)).plot(x='longitude', y='latitude', col='growing_month', col_wrap=7, subplot_kws={'projection': map_proj3},robust=True,cbar_kwargs={'label': 'Celcius degree'}, cmap='RdBu_r')
for ax in fig.axes.flat:
    ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-60,-44,-33,-8], ccrs.PlateCarree())
    
fig = result.tp.isel(growing_month=slice(0,7,1)).plot(x='longitude', y='latitude', col='growing_month', col_wrap=7, subplot_kws={'projection': map_proj3},robust=True,cbar_kwargs={'label': 'mm/day'}, cmap='RdBu')
for ax in fig.axes.flat:
    ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
    
fig = result.e.isel(growing_month=slice(0,7,1)).plot(x='longitude', y='latitude', col='growing_month', col_wrap=7, subplot_kws={'projection': map_proj3},robust=True,cbar_kwargs={'label': 'mm of equivalent water/day'},cmap='RdBu')
for ax in fig.axes.flat:
    ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
    ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
    
# #or Minimized version
# map_proj3 = ccrs.PlateCarree()
# # plt.figure(figsize=(30, 30))
# delta= DS_cli.sel(time=slice('2011-09-01','2012-03-31')).groupby('growing_month').mean() - DS_cli.sel(time=slice('1979-09-01','2011-03-31')).groupby('growing_month').mean()
# delta.t2m.attrs = {'units': 'Celcius degree', 'long_name': '2 metre temperature'}
# delta.tp.attrs = {'units': 'mm/day', 'long_name': 'Total precipitation'}
# delta.e.attrs = {'units': 'mm of water equivalent / day', 'long_name': 'Evaporation', 'standard_name': 'lwe_thickness_of_water_evaporation_amount'}

# for i in ('t2m','tp','e'):
#     dym=getattr(delta,i)
#     fig=dym.isel(growing_month=slice(0,7,1)).plot(x='longitude', y='latitude', col='growing_month', col_wrap=7, subplot_kws={'projection': map_proj3},robust=True, cbar_kwargs={'label': dym.attrs['units']})
#     for ax in fig.axes.flat:
#         ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
#                       edgecolor='black', facecolor=(0,1,0,0.0))
#         ax.set_extent([-62,-44,-33,-8], ccrs.PlateCarree())
        
#     fig.fig.tight_layout()