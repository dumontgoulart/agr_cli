# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:38:22 2020

@author: morenodu
"""


import xarray as xr 
import numpy as np 
def multi_linregress_3D(x1, x2, y, lagx=0, lagy=0):
    """
    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon). 
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated for y with respect to x.
    Output: Covariance, correlation, regression slope and intercept, p-value, and standard error on regression between the two datasets along their aligned time dimension.  
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x1,y = xr.align(x1,y)
    x2,y = xr.align(x2,y)
 
    #3. Compute data length, mean and standard deviation along time axis for further use: 
    n = y.notnull().sum(dim='time')
    x1mean=np.nanmean(x1,axis=0)
    x2mean=np.nanmean(x2,axis=0)
    ymean=np.nanmean(y,axis=0)
    x1std=np.nanstd(x1,axis=0)
    x2std=np.nanstd(x2,axis=0)
    ystd=np.nanstd(y,axis=0)
    
    #4. Compute covariance along time axis
    #cov   =  np.sum((x1 - x1mean)*(x2 - x2mean)*(y - ymean), axis=0) / (n)
    
    #5. Compute correlation along time axis
    #cor   = cov/(x1std*x2std*ystd)
    #6. Compute regression slope and intercept:
    slopex1    = ( np.sum((x2)**2, axis=0)*np.sum((x1)*(y), axis=0) - np.sum((x1)*(x2), axis=0)*np.sum((x2)*(y), axis=0) )  / ( np.sum((x1)*(x1), axis=0) * np.sum((x2)*(x2), axis=0) - (np.sum((x1)*(x2), axis=0))**2)
    slopex2    = ( np.sum((x1)**2, axis=0)*np.sum((x2)*(y), axis=0) - np.sum((x1)*(x2), axis=0)*np.sum((x1)*(y), axis=0) )  / ( np.sum((x1)*(x1), axis=0) * np.sum((x2)*(x2), axis=0) - (np.sum((x1)*(x2), axis=0))**2)
    intercept = ymean - x1mean*slopex1  - x2mean*slopex2
    
    y_pred = intercept + x1*slopex1 + x2*slopex2
    
    rss =  np.nansum((y-y_pred)**2, axis=0)
    ss_tot = np.nansum((y-ymean)**2, axis=0)
    r_2 = 1 - (rss/ss_tot)
    r_2 = xr.DataArray(r_2, dims=intercept.dims, coords=intercept.coords)
    
    #7. Compute P-value and standard error
    #Compute t-statistics
    
    # tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    # stderr = slope/tstats
    # from scipy.stats import t
    # pval   = t.sf(tstats, n-2)*2
    # pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return slopex1, slopex2,intercept, y_pred, r_2 #,pval,stderr

slope1, slope2 ,intercept, y_pred, r_2 = multi_linregress_3D(da_cli_tmax8,da_cli_wet7, DS_y_us['yield'], lagx=0, lagy=0)


plt.figure(figsize=(20,10)) 
ax=plt.axes(projection=ccrs.Mercator())
adm1_shapes = list(shpreader.Reader('gadm36_BRA_1.shp').geometries())
r_2.plot(x='lon', y='lat',transform=ccrs.PlateCarree(), robust=True,cbar_kwargs={'label': 'Correlation temperature and evaporation'},  cmap='RdBu', levels=15)
loc = plticker.MultipleLocator(base=2.0) # this locator puts ticks at regular intervals
ax.set_xticks(ax.get_xticks()[::1]); ax.set_yticks(ax.get_yticks()[::1])
ax.add_geometries(us1_shapes, ccrs.PlateCarree(),edgecolor='black', facecolor=(0,1,0,0.0))
ax.set_extent([-125,-67,24,50], ccrs.PlateCarree())
plt.show()

