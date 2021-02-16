from myselafin import Selafin
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap


def loadMeshFromSLF(fn, output_fn):
    """
    Function to load a msh from a selafin object and save it is as a plot)
    """

    if output_fn[-4] == '.': output_fn = output_fn[:-4]
    name = output_fn

    fn_mesh = output_fn+'_Mesh.png'
    fn_box = output_fn+'_Mesh.npy'

    print('Loading selafin...')
    slf = Selafin(fn)
    slf.import_header()
    data = slf.import_data(step = None)
    data = np.asarray(data,dtype = np.float32)
    slf.close()
    print('Selafin loaded!')

    print('Calculating mean waterheight')
    h = np.mean(data[2], axis = 1)

    print('Saving box...')
    xmin, xmax = np.min(slf.x), np.max(slf.x)
    ymin, ymax = np.min(slf.y), np.max(slf.y)
    xrange = xmax - xmin; yrange = ymax - ymin
    np.save(fn_box, np.array([xmin -0.05*xrange, xmax +0.05*xrange, ymin-0.05*yrange, ymax+0.05*yrange]))

    print('Saving time steps...')
    np.save(output_fn+'_t.npy', slf.times)
    print('Saving data numpy array...')
    np.save(output_fn+'_data.npy', data)
    print('Saving x and y...')
    np.save(output_fn+'_x.npy', np.array(slf.x))
    np.save(output_fn+'_y.npy', np.array(slf.y))
    np.save(output_fn+'_ikle.npy', np.array(slf.ikle))
    np.save(output_fn+'_times.npy', np.array(slf.times))

    # ------------------------------------------------------------------------------ #
    # Plot the Mesh

    print('Plotting...')
    f, a = plt.subplots(figsize=(25,25))

    tc = a.tripcolor(slf.x, slf.y, slf.ikle-1, np.zeros([np.shape(data)[1]]), cmap = 'gray')
    tc.set_edgecolors('white')
    a.axis('off')
    a.set_xlim(xmin - 0.05*xrange, xmax + 0.05*xrange)
    a.set_ylim(ymin - 0.05*yrange, ymax + 0.05*yrange)
    #a.set_aspect('equal')
    f.savefig(fn_mesh, bbox_inches='tight', transparent=True)
