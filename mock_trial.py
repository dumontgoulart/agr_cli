# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:48:01 2021

@author: morenodu
"""
import pandas as pd, numpy as np, xarray as xr

ds = xr.Dataset({
    var: xr.DataArray(
        np.random.random((4, 3, 6)),
        dims=['time', 'lat', 'lon'],
        coords=[pd.date_range('2010-01-01', periods=4, freq='Q'),
            np.arange(-60, 90, 60),
            np.arange(-180, 180, 60)])
    for var in ['tmp']})

ds.tmp[0,0,0] = 0
ds.tmp[1,2,2] = 0

ds = ds.where(ds.tmp > 0)

