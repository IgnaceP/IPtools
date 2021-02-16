""" Rasterize

Module to rasterize a shapely polygon to a Numpy Array

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point

def rasterize(mpol, res = 1, return_minmaxs = False):
    """
    Function to Rasterize

    author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        mpol: (Required) Shapely Multi(Polygon) to rasterize
        res: (Optional, defaults to 1) float/integer representing the cell size
        return_minmaxs: (Optional, defaults to False) True/False boolean to return the coordinates of the min and max of polygon and the coordinates of the top left corner


    Returns:
        Numpy array with dimensions n x m with 1 on cells covering the original polygons
        Coordinate pair with the coordinates of the left top corner of the raster
    """

    # make sure it can handle both a Polygon and a MultiPolygon
    if type(mpol) == Polygon:
        mpol = MultiPolygon([mpol])

    # initialize limits
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')

    # search min and max coordinates along polygons to determine graph limits
    for pol in mpol:
        xy = np.rot90(pol.exterior.xy)

        if np.min(xy[:, 0]) < xmin: xmin = np.min(xy[:, 0])
        if np.max(xy[:, 0]) > xmax: xmax = np.max(xy[:, 0])
        if np.min(xy[:, 1]) < ymin: ymin = np.min(xy[:, 1])
        if np.max(xy[:, 1]) > ymax: ymax = np.max(xy[:, 1])

    # raster dimensions
    rows = int(np.ceil((ymax - ymin + 1)/res))
    cols = int(np.ceil((xmax - xmin +1)/res))
    TL = [xmin - res/2,ymax + res/2] # coordinates of the top left corner of the top left corner cell
    height = res*rows
    width = res*cols

    # initialize array to represent raster
    arr = np.zeros([rows, cols])

    # nested loop over the array
    t = 0 # counter
    mesh = np.array(np.meshgrid(np.arange(rows), np.arange(cols)))
    combinations = mesh.T.reshape(-1, 2)
    for n in range(np.shape(combinations)[0]):
        i, j = combinations[n,:]
        #print("\r We are at %.2f per." % (n/(rows*cols)*100), end = "" )
        # create a shapely point at the centroid of the selected raster cell
        p = Point(xmin + res/2 + j*res, ymin + res/2 + (rows-i)*res)
        # does that point fall in the polygon? assign value 1 to that rastercell
        if mpol.contains(p):
            arr[i,j] = 1

    if return_minmaxs:
        return arr, xmin, ymin, xmax, ymax, TL
    else:
        return arr
