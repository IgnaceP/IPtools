import numpy as np
import warnings
warnings.filterwarnings("ignore")
from shapely.geometry import Point, LineString, MultiLineString, Polygon, \
                             MultiPolygon


def removeVertices(xy, geotype = 'pol', sens = 0, print_results = False):
    """ Function to remove excessive vertices of line/polygon
    :param xy: vertices list of a polygon (list of pairs or nx2 numpy array with x- and y-coordinates of the polygon's vertices)
    :param geotype: 'pol' or 'line'
    :param sens: maximum allowed change of slope
    :return: vertices list without excessive points (nx2 numpy array with x- and y-coordinates of the polygon's vertices)
    """

    # transform to np array
    xy = np.asarray(xy)
    if np.shape(xy)[1] != 2: xy = np.rot90(xy)

    # get number of vertices
    n = np.shape(xy)[0]

    # initialize an empty numpy array to store the items to be removed
    rem = np.zeros(n)

    # In case of a line, never remove the first or last vertex, in case of polygon that is possible
    if geotype ==  'pol': start_loop, end_loop = 0, n
    else: start_loop, end_loop = 1, n-1

    # loop over all vertices
    for i in range(start_loop, end_loop):
        # get left and right neighbor indices
        if i == 0: l, r = -1,1
        elif i == n-1: l, r = n-2, 0
        else: l, r = i-1, i+1

        # check slope of straights
        x, y = xy[i,:]
        lx, ly = xy[l,:]
        rx, ry = xy[r,:]
        la = (ly - y)/(lx - x)
        ra = (ry - y)/(rx - x)

        # if slopes are equal, the vertex is not essential
        if abs(la - ra) <= sens:
            rem[i] = 1

    # mask to remove all excessive vertices
    xy_upd = xy[rem==0]
    n_upd = np.shape(xy_upd)[0]

    if print_results:
        print('Reduced %d vertices to %d vertices.' % (n,n_upd))

    return xy_upd

def removeVerticesMpol(Mpol, print_results = False):
    """ Function to remove excessive vertices from a shapely Multipolygon
    :param Mpol: Shapely MultiPolygon
    :return: simplified Multipolygon
    """
    pols = []
    for pol in Mpol:
        xy = pol.exterior.xy
        xy = removeVertices(xy, print_results = print_results)

        xy_i = []
        for i in pol.interiors:
            xy_i.append(removeVertices(i.xy, print_results = print_results))

        pol_upd = Polygon(xy, xy_i)
        pols.append(pol_upd)


    return MultiPolygon(pols)

def removeVerticesPol(pol, print_results = False):
    """ Function to remove excessive vertices from a shapely Polygon
    :param pol: Shapely Polygon
    :return: simplified Polygon
    """

    xy = pol.exterior.xy
    xy = removeVertices(xy, print_results = print_results)

    xy_i = []
    for i in pol.interiors:
        xy_i.append(removeVertices(i.xy), print_results = print_results)

    pol_upd = Polygon(xy, xy_i)

    return pol_upd
