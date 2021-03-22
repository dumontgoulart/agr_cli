from fastkml import kml
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader

doc = open("gadm36_BRA_0.kml").read().encode('utf-8')
k = kml.KML()
k.from_string(doc)
features = list(k.features())
for f in k.features():
    print(f.name)
print(k.to_string(prettyprint=True))
f2 = list(features[0].features())
x = f2[0].geometry


lon,lats = x.exterior.coords.xy

df = pd.DataFrame(data=[lon,lats],index= ['Lon','Lat'])
df = df.T
df.to_csv('Northern_CV_file.csv')


#%%

fname2 = 'gadm36_BRA_0.shp'
adm2_shapes = list(shpreader.Reader(fname2).geometries())

br_shape = gpd.read_file('gadm36_BRA_0.shp')

print(adm2_shapes)
print(br_shape.geometry)