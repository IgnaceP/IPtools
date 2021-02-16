import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, Polygon, \
                             MultiPolygon
from snap_polygon_funs_py import *
from progressbar import printProgressBar

def snapPol2Pol(Polygon2Snap, Polygon2Snap2, Snap_Sensitivity, print_progressbar = False):
    """ Function to snap a polygon to another polygon
    :param polygon2Snap: Polygon which has to be edited to snap to a reference polygon (shapely (Multi)Polygon)
    :param polygon2Snap2: Reference Polygon
    :param Snap_Sensitivity: Max distance to snap
    :param print_progressbar: Boolean to print a Progressbar or not
    :return: Shapely (Multi)Polygon """

    # make sure that the rest of this script handles as it is a MultiPolygon
    if type(Polygon2Snap) == Polygon: Polygon2Snap = [Polygon2Snap]
    if type(Polygon2Snap2) == Polygon: Polygon2Snap2 = [Polygon2Snap2]

    # put all vertices of the polygon2snap2 in one 2d numpy array
    vert = np.zeros([1,2])
    for pol in Polygon2Snap2:
        xy = np.rot90(pol.exterior.xy)
        vert = np.vstack((vert, xy))

        interiors = pol.interiors
        for i in interiors:
            xy = np.rot90(i.xy)
            vert = np.vstack((vert,xy))

    polygons = []
    t = 0
    for pol in Polygon2Snap:
        t += 1
        xy = np.rot90(pol.exterior.xy)
        xy_snapped = xy.copy()
        if print_progressbar: print('Snapping outer boundaries of feature %d.' % t)
        for i in range(np.shape(xy)[0]):
            if print_progressbar: printProgressBar(i, np.shape(xy)[0])
            neigh = getNeighbor(xy[i,:], vert, tresh = Snap_Sensitivity)
            if type(neigh) == np.ndarray:
                xy_snapped[i,0] = neigh[0]
                xy_snapped[i,1] = neigh[1]

        xy_snapped_ext = xy_snapped.copy()
        del xy
        del xy_snapped

        interiors_snapped = []

        if print_progressbar: print('Snapping inner boundaries of feature %d.' % t)
        for i in pol.interiors:
            xy = np.rot90(i.xy)
            xy_snapped = xy.copy()

            for i in range(np.shape(xy)[0]):
                if print_progressbar: printProgressBar(i, np.shape(xy)[0])
                neigh = getNeighbor(xy[i,:], vert, tresh = Snap_Sensitivity)
                if type(neigh) == np.ndarray:
                    xy_snapped[i,:] = neigh

            interiors_snapped.append(xy_snapped)

        polygon = Polygon(xy_snapped_ext, interiors_snapped)
        polygons.append(polygon)

    return MultiPolygon(polygons)
