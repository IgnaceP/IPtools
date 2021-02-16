import numpy as np

arr = np.asarray([[1,2], [2,3],[2,3],[2,3],[1,2],[5,5]])
arr_z = np.asarray([10,5,6,11,8,10])

def generalizePoints(xy, z, keep = 'highest'):
    """
    Method to remove duplicate points
    """
    z = np.asarray(list(z))
    xy = np.asarray(xy)

    if keep == 'highest':
        sort_i = np.flip(np.argsort(z))
    elif keep == 'lowest':
        sort_i = np.argsort(z)
    z = z[sort_i]
    xy = xy[sort_i]
    xy, i = np.unique(xy, axis = 0, return_index = True)
    z = z[i]

    return xy, z
