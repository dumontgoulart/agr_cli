import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import seaborn as sns

#%% Import netCDF data

DS_spei_1=xr.open_dataset("spei01.nc")
DS_spei_2=xr.open_dataset("spei02.nc")

#%% for 1 month SPEI

# dataset limitnig to Brasil 2000-2015
ds_spei = DS_spei_1.sel(time=slice('2000-01-01', '2015-12-31'),lon=slice(-61.63,-44.2),lat=slice(-32.86,-7.35))

# DataArray for the west and south Brasil 2000-2015 isolating the SPEI index
da_spei = DS_spei_1.spei.sel(time=slice('2000-01-01', '2015-12-31'),lon=slice(-61.63,-44.2),lat=slice(-32.86,-7.35))

# Create a pandas dataframe by mean valuer per year
da_year=da_spei.groupby('time.year').mean()
df = da_year.to_dataframe()


#%% PLOT FIGURES
#plot scatterplot along time (dataset)
ds_spei.plot.scatter(y="spei",x="time", aspect=2, size=8)
plt.savefig('f1.png', bbox_inches='tight')

#lines for (roughly) mato grosso, parana and RS.
da_spei.sel(time=slice('2011-04-01', '2013-12-31')).isel( lon=-17, lat=[-10,-30,-40]).plot.line(x='time',aspect=2, size=8)
plt.savefig('f2.png', bbox_inches='tight')

# Boxplot assessing values per year for the whole region
boxplot = df.boxplot(by="year" , widths = 0.6, patch_artist = True, figsize=(12,8))
plt.savefig('f3.png', bbox_inches='tight')

# Illustration of region in the world
fig = plt.figure(figsize=(12, 9))
ax=plt.axes(projection=ccrs.Orthographic(-62,-20))
p=da_spei.isel(time=143).plot(transform=ccrs.PlateCarree(),robust=True,cmap='RdBu')
ax.set_global(); ax.coastlines();
fig.savefig('f4.png', bbox_inches='tight')

# Ensemble of SPEI indicators for the region along the year 2011/12 - Attention to the soybeans cycle 11->03
p=da_spei.isel(time=slice(140,152,1)).plot(x='lon', y='lat', col='time', col_wrap=4,robust=True,cmap='RdBu')
plt.savefig('f5.png', bbox_inches='tight')

# Reference year
p_ref=da_spei.isel(time=slice(110,122,1)).plot(x='lon', y='lat', col='time', col_wrap=4,robust=True,cmap='RdBu')
plt.savefig('f6.png', bbox_inches='tight')

 
