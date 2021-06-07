# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:28:28 2021

@author: morenodu
"""
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
 
DS_test2 = xr.open_dataset("EC_earth_2C/pr_m_ECEarth_2C_ensemble_2062-4062.nc",decode_times=True)

def open_regularize_3(address_file, reference_file):    
    DS_ec = xr.open_dataset(address_file,decode_times=True)
    DS_ec['lat']=DS_test2.lat
    if list(DS_ec.keys())[1] == 'tasmax':
        da_ec = DS_ec[list(DS_ec.keys())[1]] - 273.15
    elif list(DS_ec.keys())[1] == 'pr':
        da_ec = DS_ec[list(DS_ec.keys())[1]] * 1000 
    else:
        da_ec = DS_ec[list(DS_ec.keys())[1]]
    DS_ec = da_ec.to_dataset() 
    DS_ec_crop = DS_ec.where(reference_file.mean('time') > -300 )
    return DS_ec_crop


DS_tmx_ec_un = open_regularize("EC_earth_PD/tasmax_m_ECEarth_PD_ensemble_2035-4035.nc", DS_pre_cru_us['pre'])

DS_tmx_ec = open_regularize("EC_earth_PD/tasmax_m_ECEarth_PD_ensemble_2035-4035.nc", DS_pre_cru_us['pre']).groupby('time.year').mean('time')

DS_tmx_ec_2C = open_regularize("EC_earth_2C/tasmax_m_ECEarth_2C_ensemble_2062-4062.nc", DS_pre_cru_us['pre']).groupby('time.year').mean('time') 

DS_tmx_ec_3C = open_regularize_3("EC_earth_3C/tasmax_d_ECEarth_3C_ensemble_2082-4082.nc", DS_pre_cru_us['pre']).groupby('time.year').mean('time') 


DS_tmx_ec_mean = DS_tmx_ec.to_dataframe().groupby(['year']).mean()
DS_tmx_ec_mean.index = range((2000))

DS_tmx_ec_2C_mean = DS_tmx_ec_2C.to_dataframe().groupby(['year']).mean()
DS_tmx_ec_2C_mean.index = range((2000))

DS_tmx_ec_3C_mean = DS_tmx_ec_3C.to_dataframe().groupby(['year']).mean()
DS_tmx_ec_3C_mean.index = range((2000))

DS_tmx_ec_mean.plot()


fig1 = plt.figure(figsize=(10,5))
plt.hist(DS_tmx_ec_mean, bins=50, label = 'Present Day')
plt.hist(DS_tmx_ec_2C_mean,bins=50, label = '2C')
plt.hist(DS_tmx_ec_3C_mean,bins=50, label = '3C')
plt.legend(loc="upper left")
plt.title('Histogram of EC-Earth ensembles')
plt.ylabel('Frequency')
plt.xlabel('Mean yearly temperature (°C)')
fig1.savefig('paper_figures/ec_earth_moving_average.png', format='png', dpi=250)
plt.show()

plt.figure(figsize=(20,10)) #plot clusters
ax=plt.axes(projection=ccrs.Mercator())
DS_tmx_ec_un['tasmax'].mean('time').plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True)
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
# ax.set_extent([-105,-25,-50,50], ccrs.PlateCarree())
ax.set_title('Spatial variability of bias between CRU and EC-earth')
plt.show()