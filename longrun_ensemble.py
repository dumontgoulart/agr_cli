# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:45:04 2021

@author: morenodu
"""
import xclim as xc
DS_tmx_cru_longrun = xr.open_mfdataset(glob.glob('EC_earth_longruns/tasmax_Amon_ECEARTH23_rcp85_186001-210012_*.nc'), concat_dim = 'realization', combine='nested', decode_times=False)   
DS_tmx_cru_longrun.tasmax.plot(hue='realization')
plt.show()


#xr.open_dataset("EC_earth_longruns/tasmax_Amon_ECEARTH23_rcp85_186001-210012_01.nc", decode_times=False) 
units, reference_date = DS_tmx_cru_longrun.time.attrs['units'].split('since')
DS_tmx_cru_longrun['time'] = pd.date_range(start=reference_date, periods=DS_tmx_cru_longrun.sizes['time'], freq='MS')
DS_tmx_cru_longrun.coords['lon']= (DS_tmx_cru_longrun.coords['lon'] + 180) % 360 - 180
DS_tmx_cru_longrun = DS_tmx_cru_longrun.sortby(DS_tmx_cru_longrun.lon)
DS_tmx_cru_longrun = DS_tmx_cru_longrun.sel(lat=slice(50,10), lon=slice(-150,-30)) #.sel(time=slice('31-12-1944', '31-12-2045')

DS_tmx_cru_longrun = DS_tmx_cru_longrun.reindex(lat=DS_tmx_cru_longrun.lat[::-1])
DS_tmx_cru_longrun = DS_tmx_cru_longrun.reindex(lat=list(reversed(DS_tmx_cru_longrun.lat)))
DS_tmx_cru_longrun_us = DS_tmx_cru_longrun.where(DS_y['yield'].mean('time') > -0.1 )

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_tmx_cru_longrun_us['tasmax'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()


plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_cli_us['tmx'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-110,-65,25,50], ccrs.PlateCarree())
plt.show()

df_longrun_us = DS_tmx_cru_longrun_us.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_longrun_us_8 = df_longrun_us[df_longrun_us.index.month == 8]
df_longrun_us_8 = df_longrun_us_8[(df_longrun_us_8.index > '1944-12-01') & (df_longrun_us_8.index <= '2045-12-12')]
df_longrun_us_8 = df_longrun_us_8 - 273.15


df_ec_ensemble_us = DS_ec_earth_PD_us.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_ec_ensemble_us_8 = df_ec_ensemble_us[df_ec_ensemble_us.index.month == 8] #[(df_cru_us.index > '1920-12-01') & (df_cru_us.index <= '2020-12-12')]


df_cru_us = DS_cli_us.to_dataframe().groupby(['time']).mean() # pandas because not spatially variable anymore
df_cru_us_8 = df_cru_us[df_cru_us.index.month == 8] #[(df_cru_us.index > '1920-12-01') & (df_cru_us.index <= '2020-12-12')]


plt.plot(df_longrun_us_8, label = 'Longrun ensemble')
plt.plot(df_cru_us_8.tmx, label = "CRU")
plt.legend()
plt.show()

sns.kdeplot(data = df_longrun_us_8, x = df_longrun_us_8.tasmax, fill=True, alpha=.5,linewidth=0, label = 'Longrun EC')
sns.kdeplot(data = df_cru_us_8, x = df_cru_us_8.tmx, fill=True, alpha=.5, linewidth=0, label = "CRU")
sns.kdeplot(data = df_ec_ensemble_us_8, x = df_ec_ensemble_us_8.tmx, fill=True, alpha=.5, linewidth=0, label = "Ensemble EC")
plt.legend()
plt.show()





