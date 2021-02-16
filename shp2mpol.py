""" Shapefile 2 Shapely MultiPolygon

This module loads a shapefile to a shapefile MultiPolygon

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""


from osgeo import ogr
import os
import json
from shapely.geometry import Polygon as shPol
from shapely.geometry import MultiPolygon as shMPol
from shapely.geometry import shape as shapshape
import utm
import numpy as np
from pyproj import CRS
from pol2pol import project, pol2Pol
from pprint import pprint

def shp2Mpol(fn, get_fields = False ,return_coordinate_system = False, print_coordinate_system = False, print_n_features = False):
    """
    Function to load and (re)project one or multiple polygons from a ESRI shapefile
    ! the shapefile should only have one polygon per feature !

    author: Ignace Pelckmans

    Args:
        fn: (Required) string of .shp file directory
        get_fields: (Optional, defaults to False) list of integers or False to return a field or fields
        return_coordinate_system: (Optional, defaults to False) True/False to return the original EPSG code
        print_coordinate_system: (Optional, defaults to False) True/False to print the original EPSG code
        print_n_features: (Optional, defaults to False) True/False to print the number of features in the polygon

    Returns:
        a shapely (Multi)Polygon
        (Optional) an int representing the original EPSG code of the coordinate system
    """

    # check if file exists
    if not os.path.isfile(fn):
        raise ValueError('Wrong inputfile!')

    # load shapefile with the ogr toolbox of osgeo
    file = ogr.Open(fn)
    shape = file.GetLayer(0)

    epsg = int(shape.GetSpatialRef().ExportToPrettyWkt().splitlines()[-1].split('"')[3])
    crs = CRS.from_epsg(epsg)
    if print_coordinate_system:
        print("The EPSG code of the coordinate system is: %d" % (crs.to_epsg()))
    # get number of polygons in shapefile
    n_features = shape.GetFeatureCount()

    if print_n_features:
        print('There are %d features in the shapefile.' % n_features)

    # initialize new polygon list
    pols = []
    fields = []

    # loop over all polygons
    for i in range(n_features):
        # get feature object
        feature = shape.GetFeature(i)
        # export to JS objects
        feature_JSON = feature.ExportToJson()
        feature_JSON = json.loads(feature_JSON)

        if get_fields:
            f = [feature.GetField(j) for j in get_fields]
            fields.append(f)

        # create a shapely polygon
        shp_geom = shapshape(feature_JSON["geometry"])
        if type(shp_geom) == shMPol:
            pol = shp_geom[0]
        elif type(shp_geom) == shPol:
            pol = shp_geom
        else:
            raise ValueError('Invalid input shapefile.')
        pols.append(pol)

    # create a shapely MultiPolygon
    mpol = shMPol(pols)

    if return_coordinate_system:
        if len(pols)==1:
            if get_fields:
                return pol, crs.to_epsg(), fields
            else:
                return pol, crs.to_epsg()
        else:
            if get_fields:
                return mpol, crs.to_epsg(), fields
            else:
                return mpol, crs.to_epsg()
    else:
        if len(pols)==1:
            if get_fields:
                return pol, fields
            else:
                return pol
        else:
            if get_fields:
                return mpol, fields
            else:
                return mpol
