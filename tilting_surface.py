#!/usr/bin/evn python

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# !!! copyright to https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

def estimateTiltingPlane(points, points2interp = [], bounds = [], res = 100, order = 1, return_RMSE = False):
    """
    Method to estimate a tilting plane based on a set of points

    Args:
        points: (Required) n x 3 numpy array with the x, y and z coordinates of known points
        points2interp: (Optional, defaults to []) n x 2 numpy array with points to interpolate
        bounds: (Optional, defaults to []) list with xmin, xmax, ymin and ymax
        resolution: (Optional, defaults to 100) resolution of the output points
        order: (Optional, defaults to 1) 1 or 2 indicating whether to fit a linear or quadratic plane
        return_RMSE: (Optional, defaults to False) True/False to return the RMSE as well
    """

    if len(points2interp) == 0:
        if len(bounds) == 4:
            xmin, xmax, ymin, ymax = bounds
        else:
            xmin = np.min(points[:,0])
            xmax = np.max(points[:,0])
            ymin = np.min(points[:,1])
            ymax = np.max(points[:,1])

        # regular grid covering the domain of the data
        X,Y = np.meshgrid(np.arange(xmin, xmax, res), np.arange(ymin, ymax, res))
        XX = X.flatten()
        YY = Y.flatten()
    else:
        X, Y = points2interp[:,0], points2interp[:,1]
        XX, YY = points2interp[:,0], points2interp[:,1]


    data = points

    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,residues,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients

        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,residues,_,_ = scipy.linalg.lstsq(A, data[:,2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

    if return_RMSE:
        # calculate root mean square error
        RMSE = residues
        return [XX, YY, Z.flatten()], RMSE
    else:
        return [XX, YY, Z.flatten()]
