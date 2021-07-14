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

import numpy as np
from pyproj import CRS
from pol2pol import project, pol2Pol
from pprint import pprint
import pandas as pd

def shp2Mpol(fn, get_fields = False ,return_coordinate_system = False, print_coordinate_system = False, print_n_features = False, include_centroid = False):
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

    if get_fields:
        fieldnames = []
        for n in range(shape.GetLayerDefn().GetFieldCount()):
            if get_fields == 'All': fieldnames.append(shape.GetLayerDefn().GetFieldDefn(n).name)
            else:
                if n in get_fields: fieldnames.append(shape.GetLayerDefn().GetFieldDefn(n).name)
        if get_fields == 'All': get_fields = list(np.arange(len(fieldnames), dtype = int))
        if include_centroid: fieldnames.append('Centroid')


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


        # create a shapely polygon
        if feature_JSON["geometry"] != None:
            shp_geom = shapshape(feature_JSON["geometry"])
            if type(shp_geom) == shMPol:
                pol = shp_geom[0]
            elif type(shp_geom) == shPol:
                pol = shp_geom
            else:
                raise ValueError('Invalid input shapefile.')
            pols.append(pol)

            if get_fields:
                f = [feature.GetField(int(j)) for j in get_fields]
                centroid_xy = np.asarray([arr[0] for arr in pol.centroid.xy])
                if include_centroid: f.append(centroid_xy)
                fields.append(f)

    # create a shapely MultiPolygon
    mpol = shMPol(pols)

    # create a dataframe to store the fields
    fields = np.asarray(fields)
    df = {}

    if get_fields:
        for i in range(len(fieldnames)):
            df[fieldnames[i]] = fields[:,i]
        fields = pd.DataFrame(df)

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
