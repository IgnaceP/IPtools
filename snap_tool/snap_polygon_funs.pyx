import numpy as np
cimport numpy as cnp

cpdef getNeighbor(cnp.ndarray[cnp.double_t, ndim = 1] p,
                  cnp.ndarray[cnp.double_t, ndim = 2] pol,
                  double kernel_fract = 0.01, double tresh = float('inf')):
    """
    Function to find the closest vertex of polgyon to a POI
    :param p: POI (1x2 numpy array with x- and y-coordinate in a metric system)
    :param pol: vertices list of a polygon (nx2 numpy array with x- and y-coordinates of the polygon's vertices)
    :param kernel_fract: stepsize of the kernel size (% of the total width/height of the polygon)
    :param tresh: treshold of what can be considered a neighbor. If the distance of the closest coordinate pair is larger than this treshold, the function returns 0
    :return: coordinate pair (1x2 numpy array with x- and y-coordinate pair) of the closest vertex to the POI or zero if there is no neighbor
    """
    # get width and height of channel polygon
    cdef double width = np.max(pol[:, 0]) - np.min(pol[:, 0])
    cdef double height = np.max(pol[:, 1]) - np.min(pol[:, 1])

    # parse to separate x- and y-coordinate
    cdef double px = p[0]
    cdef double py = p[1]

    # stepsize to increase kernel
    cdef double ks = kernel_fract* max(width, height)

    # initialize a flag to feed the while loop and start a counter
    cdef double neigh_flag = False
    cdef int t = 0
    # run while loop as long as there are no neighbouring vertices selected
    while neigh_flag == False:

        # count
        t += 1

        # create a square mask to only calculate the distances between the POI and nearby polygon vertices
        neigh_mask = (pol[:,0] > px - t*ks) * (pol[:,0] < px + t*ks) \
                * (pol[:,1] > py - t*ks) * (pol[:,1] < py + t*ks)
        # mask on all vertices
        neigh = pol[neigh_mask]

        # calculate the distance between the POI en all vertices after masking
        dist = (np.sum((neigh - p) ** 2, axis=1)) ** 0.5
        # mask a circle within the square kernel
        circle_mask = (dist <= t*ks/2)
        neigh = neigh[circle_mask]
        dist = dist[circle_mask]

        # terminate loop if there is at least one vertices in the neighbouring circle, otherwise increase the kernel size
        if np.shape(neigh)[0] > 0:
            neigh_flag = True

    # get the neigbor coordinates
    if np.min(dist)> tresh: neighbor = 0
    else: neighbor = neigh[np.argmin(dist),:]

    return neighbor
