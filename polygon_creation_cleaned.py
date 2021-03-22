import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from shapely.geometry import Polygon, Point
from netCDF4 import Dataset 
import wrf

wrf_file = Dataset('/Volumes/Seagate Backup /wrf_file/wrfout_d02_2016-07-27')
lats2 = wrf.getvar(wrf_file,'XLAT',timeidx=0)
lons2 = wrf.getvar(wrf_file,'XLONG',timeidx=0)
df = pd.read_csv('../../Northern_CV_file.csv',index_col=0)

lats = df.Lat.values 
lons = df.Lon.values
points = []
for i in range(len(lats)):
	_= [lons[i],lats[i]]
	points.append(_)

# print(x)
poly_proj = Polygon(points)
x = Point(lons2[0][0], lats2[0][0]) 
print(x.within(poly_proj))

mask = np.zeros_like(lats2)                                             
print(mask.shape)

for i in range(lats2.shape[0]):                                                      
    for j in range(lats2.shape[1]):                                                  
        grid_point = Point(lons2[i][j], lats2[i][j])                  
        if grid_point.within(poly_proj):  
        	print('worked')
        	mask[i][j] = 1

bool_final = mask
np.save('./central_Northern_CV_file',bool_final)

xxx
