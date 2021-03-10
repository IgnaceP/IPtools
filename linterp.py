import numpy as np
from scipy.interpolate import griddata

def linterp(xy_known, z_known, xy_unknown, outside = 'nearest'):
    """
    Method to interpolate. Points within the hull will be interpolated linearly. Points outside the hull will be interpolated according to:
        1. nearest: a nearest neighbour interpolation
        2. 2neigh: a weighted average of the two nearest known points

    Args:
        xy_known: (Required) n x 2 numpy array with coordinate pairs of the known points
        z_known: (Required) n x 1 numpy array with z values of known coordinate points
        xy_unknown: (Required) m x 2 numpy array with coordinate pairs of the points to interpolate to
        outside: (Optional, defaults to nearest) string (nearest/twoneigh) method to handle values outside the hull of known points

    Returns:
        z_unknown: m x 1 numpy array with interpolated values on the location so xy_unknown
    """

    # interpolate within the hull
    z_unknown = griddata(xy_known, z_known, xy_unknown, method = 'linear', fill_value = -99999)

    # outside the hull
    outside_mask = (z_unknown == -99999)
    if outside == 'nearest':
        z_unknown_outside = griddata(xy_known, z_known, xy_unknown, method = 'nearest')
        z_unknown[outside_mask] = z_unknown_outside[outside_mask]
    else:
        for i in range(len(z_unknown)):
            if outside_mask[i]:
                dist = ((xy_known[:,0]-xy_unknown[i,0])**2 + (xy_known[:,1]-xy_unknown[i,1])**2)**0.5
                neighs_xy = np.sort(dist)[:2]
                neighs_z = z_known[np.argsort(dist)[:2]]
                new_z = np.sum(np.flip(neighs_xy/np.sum(neighs_xy))*neighs_z)
                z_unknown[i] = new_z

    return z_unknown

#############################################################################################""
# test

xy = np.array([[0,0],[0,10],[10,10]])
z = np.array([5, 10, 15])

xy_ti = np.asarray([[3,5],[10,5], [2,1]])
z_ti = linterp(xy, z, xy_ti, outside = 'twoneigh')

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(xy[:,0], xy[:,1], c = 'blue')
for i in range(len(z)):
    ax.annotate('%.2f' % z[i], xy[i])

ax.scatter(xy_ti[:,0], xy_ti[:,1], c = 'orange')
for i in range(len(z_ti)):
    ax.annotate('%.2f' % z_ti[i], xy_ti[i])

fig.savefig('test.png')
