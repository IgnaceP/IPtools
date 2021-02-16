from osgeo import gdal, osr

def geotiff2Np(fn, return_projection = False, return_TL_coors = False, return_resolution = False):
    """
    Function to read a geotiff file and convert it to a Numpy array

    Args:
        fn: (Required) directory path string to indicate the filename of the raster
        return_projection: (Optional, defaults to False) True/False to indicate returning the epsg code
        return_TL_coors: (Optional, defaults to False) True/False to indicate returning the coordinates of the Top left corner
        return_resolution: (Optional, defaults to False) True/False to indicate returning the reoslution
    """

    # read file
    ds = gdal.Open(fn)
    # retreive epsg code
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)
    # retreive the coordinates of TL corner and resolution
    TL_x, res, _, TL_y, _, _ = ds.GetGeoTransform()
    # assume there is only one band
    band = ds.GetRasterBand(1)
    # read band as Numpy array
    arr = band.ReadAsArray()

    if return_projection == False and return_TL_coors == False and return_resolution == False:
        return arr
    else:
        ret = [arr]
        if return_projection: ret.append(epsg)
        if return_TL_coors: ret.append(TL_x);ret.append(TL_y)
        if return_resolution: ret.append(res)

        return ret
