"""
Script with function to translate a shapely (Multi)Polygon object to a Numpy raster

author: Ignace Pelckmans
                (University of Antwerp, Belgium)
"""

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

def pol2Numpy(mpol, inner = True):
    """
    Function to translate a shapely (Multi)Polygon object to a 2D Numpy array holding the coordinates

    Args:
        mpol: (Required) Shapely (MultiPolygon)
        inner: (Optional, defaults to True) True/False boolean to add the coordinate pairs of the interior rings

    Returns:
        a Numpy array of dimensions n x 2 representing the coordinates of the exterior ring and interior rings
    """

    # if the type of the mpol variable is a single Polygon, store it as a single item in a list
    if type(mpol) == Polygon: mpol = [mpol]

    # loop over the list of polygons or MultiPolygon
    for pol in mpol:
        # store the exterior ring in Numpy array
            xy = np.rot90(pol.exterior.xy)
            # if inner are indicated
            if inner:
                # append the coordinate pairs of the interiors to the exteriors
                for i in pol.interiors:
                    xy_i = np.rot90(i.xy)
                    xy = np.vstack((xy, xy_i))

    # make sure it are only 2D coordinate pairs
    xy = xy[:,:2]

    # remove duplicates
    xy = np.unique(xy, axis = 0)

    return xy
