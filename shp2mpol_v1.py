""" Shapefile 2 Shapely MultiPolygon

This module loads a shapefile to a shapefile MultiPolygon

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""


from osgeo import ogr
import json
from shapely.geometry import Polygon as shPol
from shapely.geometry import MultiPolygon as shMPol
import utm
import numpy as np
from pyproj import CRS
from pol2pol import project, pol2Pol
from pprint import pprint

def shp2Mpol(fn, return_coordinate_system = False, print_coordinate_system = False):
    """
    Function to load and (re)project one or multiple polygons from a ESRI shapefile
    ! the shapefile should only have one polygon per feature !

    author: Ignace Pelckmans

    Args:
        fn: (Required) string of .shp file directory
        return_coordinate_system: (Optional, defaults to False) True/False to return the original EPSG code
        print_coordinate_system: (Optional, defaults to False) True/False to print the original EPSG code

    Returns:
        a shapely (Multi)Polygon
        (Optional) an int representing the original EPSG code of the coordinate system
    """

    # load shapefile with the ogr toolbox of osgeo
    file = ogr.Open(fn)
    shape = file.GetLayer(0)


    epsg = int(shape.GetSpatialRef().ExportToPrettyWkt().splitlines()[-1].split('"')[3])
    crs = CRS.from_epsg(epsg)
    if print_coordinate_system:
        print("The EPSG code of the coordinate system is: %d" % (crs.to_epsg()))
    # get number of polygons in shapefile
    n_features = shape.GetFeatureCount()

    # initialize new polygon list
    pols = []

    # loop over all polygons
    for i in range(n_features):
        # get feature object
        feature = shape.GetFeature(i)
        print(dir(feature))
        # export to JS objects
        feature_JSON = feature.ExportToJson()
        # loads as JS object array
        feature_JSON = json.loads(feature_JSON)


        # extract coordinate attribute from JS object
        # coor is a list of all rings, first one is the outer ring, further elements are coordinate pair lists of the inner rings
        coor = feature_JSON['geometry']['coordinates']
        ex = coor[0]; inner = coor[1:]

        # create a shapely polygon
        if len(inner) > 0: pol = shPol(ex, inner)
        else: pol = shPol(ex)
        pols.append(pol)

    # create a shapely MultiPolygon
    mpol = shMPol(pols)

    if return_coordinate_system:
        if len(pols)==1:
            return pol, crs.to_epsg()
        else:
            return mpol, crs.to_epsg()
    else:
        if len(pols)==1:
            return pol
        else:
            return mpol
