import numpy as np
import osgeo.ogr as ogr
import osgeo.osr as osr
import os
import json
import pandas as pd
from time import sleep

def csv2shp(fn, fn_output, field_names = [], x = 0, y = 1, epsg = 4326):
    """
    Method to translate a csv to a shapefile containing points.

    Args:
    - fn: (Required) path string of the input csv file
    - fn_output: (Required) path string of the shapefile to store the points in
    - field_names: (Optional, defaults to []) list of field names.
    - x: (Optional, defaults to 0) int indicating the column where the x-coordinate is stored
    - y: (Optional, defaults to 1) int indicating the column where the y-coordinate is stored
    - epsg: (Optional, defaults to 4326) int indicating the epgs code of the desired projection
    """

    df = pd.read_csv(fn)
    if len(field_names) == 0:
        field_names = df.columns
    elif len(field_names) != len(df.columns):
        field_names = ['x','y'] + field_names

    X = df[df.columns[x]]
    Y = df[df.columns[y]]
    n = len(X)

    fields = {}
    if len(df.columns) > 2:
        for i in range(2,len(df.columns)):
            fields[field_names[i]] = df[df.columns[i]].values

    # set up the shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the data source
    data_source = driver.CreateDataSource(fn_output)
    # create the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    # create the layer
    layer = data_source.CreateLayer(fn_output.split('.')[-2], srs, ogr.wkbPoint)

    # add fields
    layer.CreateField(ogr.FieldDefn("X", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("Y", ogr.OFTReal))

    for f in fields:
        if isinstance(fields[f][0],float): fieldtype = ogr.OFTReal
        elif isinstance(fields[f][0],int): fieldtype = ogr.OFTInteger
        else: fieldtype = ogr.OFTString
        fieldtype = ogr.OFTReal
        layer.CreateField(ogr.FieldDefn(f, fieldtype))

    # add points
    for i in range(n):
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        for f in fields: feature.SetField(f, float(fields[f][i]))
        feature.SetField('X', float(X[i]))
        feature.SetField('Y', float(Y[i]))

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
