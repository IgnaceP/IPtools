import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import mapping


def sampleRasterWithPolygons(polygons_fn, band_names, band_files):
    # load polygons
    gdf = gpd.read_file(polygons_fn)
    gdf['type_code'] = pd.factorize(gdf['type'])[0] + 1
    gdf['id'] = gdf.index

    # open a band raster
    with rasterio.open(band_files[0]) as src:
        # load band
        band_img = src.read(1)

        # rasterize training polygons with id as value
        pol_arr = rasterize_gdf(gdf, src, attribute = 'id')

        # flatten 2D arrays
        band1_flat = band_img.flatten()
        pol_flat = pol_arr.flatten()

        # create mask to only keep pixels inside of polygons
        pol_mask = (pol_flat > 0)

        print(f'Band {band_names[0]} sampled!')

    training_pixels = pd.DataFrame({'id': pol_flat[pol_mask], band_names[0]: band1_flat[pol_mask]})
    # open all bands
    for bandname, band_fn in zip(band_names[1:], band_files[1:]):
        with rasterio.open(band_fn) as src:
            # load band
            band_img = src.read(1)

            # flatten 2D arrays
            band_flat = band_img.flatten()

            # add to dataframe
            training_pixels[bandname] = band_flat[pol_mask]

        print(f'Band {bandname} sampled!')


    return training_pixels



def rasterize_gdf(gdf, src, attribute=None, fill=0, dtype="int32"):
    """
    Rasterize a GeoDataFrame into the grid of a rasterio dataset.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Vector geometries to rasterize.
    src : rasterio.io.DatasetReader
        Open rasterio dataset to provide metadata (transform, shape, crs).
    attribute : str, optional
        Column in gdf to burn into the raster. If None, burn value=1.
    fill : int or float
        Fill value for areas outside geometries.
    dtype : str
        Data type of output raster.

    Returns
    -------
    numpy.ndarray
        Rasterized array with same height/width as src.
    """
    if attribute:
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))
    else:
        shapes = ((geom, 1) for geom in gdf.geometry)

    out_arr = rasterize(
        shapes=shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=fill,
        dtype=dtype
    )
    return out_arr


def bands2DF(band_names, band_files):

    bands_flat = {}

    for band_name, band_fn in zip(band_names, band_files):
        # open a band raster
        with rasterio.open(band_fn) as src:
            # load band
            band_img = src.read(1)

            band_img = band_img[6000:15000,10000:20000]

            # flatten 2D arrays
            band_flat = band_img.flatten()

            bands_flat[band_name] = band_flat

            print(f'Band {band_name} flattened!')

    bands_flat = pd.DataFrame(bands_flat)

    return bands_flat
