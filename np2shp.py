import numpy as np
import osgeo.ogr as ogr
import osgeo.osr as osr
import os
import json
import pandas as pd
from time import sleep

def npy2Shp(arr, fn_output, field_names = [], field_types = ['real'], x = 0, y = 1, epsg = 4326):
    """
    Method to translate a numpy array to a shapefile containing points.

    Args:
    - arr: (Required) numpy array where the first two columns are the X and Y
    - fn_output: (Required) path string of the shapefile to store the points in
    - field_names: (Optional, defaults to []) list of field names.
    - x: (Optional, defaults to 0) int indicating the column where the x-coordinate is stored
    - y: (Optional, defaults to 1) int indicating the column where the y-coordinate is stored
    - epsg: (Optional, defaults to 4326) int indicating the epgs code of the desired projection
    """

    rows, cols = np.shape(arr)

    if len(field_names) == 0:
        field_names = ['x','y']
    elif len(field_names) != cols:
        field_names = ['x','y'] + field_names

    X = arr[:,x]
    Y = arr[:,y]

    fields = {}
    if cols > 2:
        for i in range(2,cols):
            fields[field_names[i]] = arr[:,i]

    # set up the shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the data source
    data_source = driver.CreateDataSource(fn_output)
    # create the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    # create the layer
    layer = data_source.CreateLayer('', srs, ogr.wkbPoint)

    # add fields
    layer.CreateField(ogr.FieldDefn("X", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("Y", ogr.OFTReal))

    t = 0
    for f in fields:
        if len(field_types) == 1: t = 0
        if field_types[t].startswith('real'): fieldtype = ogr.OFTReal
        elif field_types[t].startswith('string'): fieldtype = ogr.OFTString
        elif field_types[t].startswith('int'): fieldtype = ogr.OFTInteger64
        t += 1
        layer.CreateField(ogr.FieldDefn(f, fieldtype))

    # add points
    for i in range(rows):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        for f in fields: feature.SetField(f, float(fields[f][i]))
        feature.SetField('X', X[i])
        feature.SetField('Y', Y[i])

        # create the WKT for the feature using Python string formatting
        wkt = "POINT(%f %f)" %  (X[i] , Y[i])
        # Create the point from the Well Known Txt
        point = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(point)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    # Save and close the data source
    data_source = None
