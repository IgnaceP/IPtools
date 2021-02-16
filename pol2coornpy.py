import numpy as np
from shapely.geometry import Polygon, MultiPolygon

def pol2CoorNpy(pols):
    """
    Method to create a n x 2 numpy list of all coordinates.

    Args:
    - pols: (Required) shapely (Multi)Polygon

    Returns:
    - xy: numpy array with column for x and column for y coordinates
    """

    if type(pols) != MultiPolygon:
        pols = [pols]

    xy_total = np.zeros([1,2])

    for pol in pols:
        xy = np.rot90(np.asarray(pol.exterior.xy))
        for interior in pol.interiors:
          xy = np.vstack((xy, np.rot90(np.asarray(interior.xy))))
        xy_total = np.vstack((xy_total, xy))

    return xy_total[1:,:]
