import numpy as np
from shapely.geometry import Polygon, MultiPolygon

def addMiddlePoints(p1, p2, n ):
    """ Function to calculate the middle point between two points

    Args:
        p1: (Required) set of coordinates of first point
        p2: (Required) set of coordinates of second point
        n: (Required) number of divisions
        """

    x_1 = p1[0]
    y_1 = p1[1]

    x_2 = p2[0]
    y_2 = p2[1]

    dx = (x_2 - x_1)/n
    dy = (y_2 - y_1)/n

    ps = [[x_1,y_1]]

    for i in range(1,n):
        x = x_1 + i*dx
        y = y_1 + i*dy

        ps.append([x,y])

    ps.append([x_2,y_2])

    return ps

def addVertices(xy,dist_param, xytype = 'polygon'):
    """ Function to add vertices on linear segments which are further apart than the defined distance parameter

    Args:
        xy: (Required) Numpy array with dimensions n x 2, shapely Polygon or MultiPolygon representing the x- and y-coordinates of the polygon's vertices
        xytype: (Optional, defaults to 'polygon') string 'polygon' of 'line' to indicate whether the feature to measure the distance to is a polygon or a line (only applies if xy is a numpy array)
        dist_param  - Required: distance treshold from which extra vertices will be added

    """

    # make sure it can handle pol being a list of coors, shapely MultiPolygon or a shapely Polygon
    # interior rings can be added as a regular coordinate pairs to the array
    if type(xy) == Polygon:
        shapely = True
        xytype = 'polygon'
        xy_np = np.rot90(XY.exterior.xy)
        xy_inner = [np.rot90(i.xy) for i in XY.interiors]

        xy = xy_np
        for i in xy_inner:
            xy = np.vstack((xy, i))

    elif type(xy) == MultiPolygon:
        shapely = True
        xytype = 'polygon'
        xy = np.zeros([1,2])
        for xy in XY:
            xy_np = np.rot90(xy.exterior.xy)
            xy = np.vstack((xy, xy_np))
            xy_inner = [i.xy for i in xy.interiors]
            for i in xy_inner:
                xy = np.vstack((xy, i))
        xy = xy[1:,:]
    else:
        shapely = False
        pass

    n = np.shape(xy)[0]
    pol_updates = []

    for i in range(n - 1):

        x_1 = xy[i, 0]
        y_1 = xy[i, 1]

        x_2 = xy[i + 1, 0]
        y_2 = xy[i + 1, 1]

        pol_updates.append((x_1, y_1))

        dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

        if dist > dist_param:
            n_inter = int(dist // (dist_param / 2))
            inter = addMiddlePoints((x_1, y_1), (x_2, y_2), n_inter)
            for i in inter:
                pol_updates.append(i)
        else:
            pol_updates.append((x_2, y_2))

    pol_updates.append((xy[-1, 0], xy[-1, 1]))
    exterior = np.asarray(pol_updates)

    if shapely:
        # Interiors
        interiors = []

        for interior in pol.interiors:
            xy = interior.coords.xy
            xy = np.asarray(xy)
            xy = np.rot90(xy)

            n = np.shape(xy)[0]
            pol_updates = []

            for i in range(n-1):

                x_1 = xy[i, 0]
                y_1 = xy[i, 1]

                x_2 = xy[i+1, 0]
                y_2 = xy[i+1, 1]

                pol_updates.append((x_1, y_1))

                dist = np.sqrt((x_1-x_2)**2+(y_1-y_2)**2)

                if dist > dist_param:
                    n_inter = int(dist//(dist_param/2))
                    inter = addMiddlePoints((x_1,y_1),(x_2,y_2), n_inter)
                    for i in inter:
                        pol_updates.append(i)
                else:
                    pol_updates.append((x_2, y_2))

            pol_updates.append((xy[-1, 0], xy[-1, 1]))
            interiors.append(pol_updates)


        pol_updated = shPol(shell = exterior, holes = interiors)

        return pol_updated
    else:
        return exterior
