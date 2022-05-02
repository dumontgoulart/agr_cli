# Function for masking the dataset with a country's border. Everything out of it returns a Nan. Can provide better analysis.
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping

def mask_shape_border (DS,shape_file_address):
    """
    This function takes a netcdf file and a shape and delivers the netcdf cropped
    according to the shape. Assumption of EPSG: 4326 required.
    
    Parameters: 
    DS: Dataset with multiple climatic features (.nc);
    shape_file_address: a directory to the shapefile location or the shapefile itself.
        
    Returns: 
    DS_clipped: The datset (.nc) cropped according to the shape file.
        
    Created on Wed Jun 21 17:19:09 2020 
    by @HenriqueGoulart
    """

    def conv_lat(DS):
       if 'lat' in DS:
           DS.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
       elif 'latitude' in DS:
           DS.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
       else:
           print("Error: check latitude and longitude variable name.")
       return DS
    
    # Define coordinates from DS
    DS = conv_lat(DS)
    
    # Define CRS
    DS.rio.write_crs("epsg:4326", inplace=True)
    
    # Load shape according to type of input file
    if type(shape_file_address) == str:         
        mask = gpd.read_file(shape_file_address, crs="epsg:4326")         
    else:         
        mask = shape_file_address    
    
    # Clipping and dropping extra dimension
    DS_clipped = DS.rio.clip(mask.geometry.apply(mapping), mask.crs, drop=False)
    DS_clipped = DS_clipped.drop('spatial_ref')

    if 'spatial_ref' in list(DS_clipped.coords):
        DS_clipped=DS_clipped.drop('spatial_ref')
    
    
    return(DS_clipped)