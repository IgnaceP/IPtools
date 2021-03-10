import numpy as np
from geotiff2Np import *
from shp2mpol import *
from pol2pol import pol2Pol

def burnValuesIntoRaster(raster, polygon, raster_fn):
    """
    Function to burn values of polygon properties into a raster

    Args:
    - raster: (Required) geotiff directory path to the raster to burn the values into
    - polygon: (Required) shapefile directory path of the polygons with one attribute
    - raster_fn: (Required) geotiff directory path to save the new raster
    """

    # load raster and mpol
    arr, epsg_ras, TL_x, TL_y, res = geotiff2Np(raster, return_TL_coors = True, return_resolution = True, return_projection= True)
    mpol, epsg_mpol, fields  = shp2Mpol(polygon, get_fields=[0], return_coordinate_system = True)


    # if polygon has a different coordinate system than the raster, reproject the polygon
    if epsg_ras != epsg_mpol:
        mpol = pol2Pol(mpol, epsg_mpol, epsg_ras)



ras_fn = '/media/ignace/LaCie/TELEMAC/Gulf_setup/input/gis/Delta_Rios_Cleaned_25m_edited.tif'
pol_fn = '/media/ignace/LaCie/TELEMAC/Gulf_setup/input/gis/scratch.shp'
burnValuesIntoRaster(ras_fn, pol_fn, ' ')
