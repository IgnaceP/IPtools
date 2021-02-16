import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPolygon, Polygon
import osgeo.ogr as ogr
import osgeo.osr as osr
import os
from pol2pol import project, pol2Pol



def pol2Shp(pols, fn, epsg = 4326, fields = [], field_names = [], field_types = ['real']):
    """
    Method to translate a shapely MultiPolygon to a shapefile
    """

    # set right format
    if type(pols) == Polygon: pols = [pols]

    # number of polygons in MultiPolygon
    n = len(pols)

    attr_fields = {}
    if len(fields) > 0:
        for i in range(len(fields)):
            attr_fields[field_names[i]] = fields[i]

    # set up the shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the data source
    data_source = driver.CreateDataSource(fn)
    # create the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    # create the layer
    layer = data_source.CreateLayer(fn.split('.')[-2], srs, ogr.wkbPolygon)

    # create fields
    t = 0
    for f in attr_fields:
        if len(field_types) == 1: t = 0
        if field_types[t].startswith('real'): fieldtype = ogr.OFTReal
        elif field_types[t].startswith('string'): fieldtype = ogr.OFTString
        elif field_types[t].startswith('int'): fieldtype = ogr.OFTInteger64
        t += 1
        layer.CreateField(ogr.FieldDefn(f, fieldtype))

    # add polygons
    for i in range(n):
        pol = pols[i]
        if not pol.is_valid: pol = pol.buffer(0)
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        for f in attr_fields: feature.SetField(f, attr_fields[f][i])

        # create the WKT for the feature using Python string formatting
        wkt = "POLYGON(("

        for xy in np.rot90(pol.exterior.xy):
            x, y = xy
            wkt += "%f %f" % (x, y)
            if x != np.rot90(pol.exterior.xy)[0,0] or y != np.rot90(pol.exterior.xy)[0,1]:
                wkt += ","
        wkt += ")"
        for interior in pol.interiors:
            wkt += ",("
            for xy in np.rot90(interior.xy):
                x, y = xy
                if x < 0 and y < 0: xy = project(xy, 4326, 32717)
                wkt += "%f %f" % (x, y)
                if x != np.rot90(interior.xy)[0,0] or y != np.rot90(interior.xy)[0,1]:
                    wkt += ","
            wkt += ")"
        wkt += ")"

        # Create the polygon from the Well Known Txt
        polygon = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(polygon)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    # Save and close the data source
    data_source = None
