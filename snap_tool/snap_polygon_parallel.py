from shp2mpol import *
import Functions as fun
from snap_polygon import *
import Functions as fun
from polplot import *
from mpol2shp import *
import multiprocessing as mp
from shapely.geometry import MultiPolygon

def snapPol2PolParallel(pol2plot, pol2plot2, snap_sensitivity = 1, cpus = 6):
    """
    Function to snap a MultiPolygon to another MultiPolygon in parallel
    :param polygon2Snap: Polygon which has to be edited to snap to a reference polygon (shapely (Multi)Polygon)
    :param polygon2Snap2: Reference Polygon
    :param snap_sensitivity: Max distance to snap
    :param cpus: number of cpu's to include in the process
    :return: Shapely (Multi)Polygon
    """

    if cpus > mp.cpu_count():
        print("There are no %d cpu's available, instead the programm will all %d cpu's" % (cpus, mp.cpu_count()))
        cpus = mp.cpu_count()


    pool = mp.Pool(cpus)
    pols = pool.starmap(snapPol2Pol, [(pol, pol2plot2, snap_sensitivity, False) for pol in pol2plot])
    pool.close

    pols = [p[0] for p in pols]
    pol_snapped = MultiPolygon(pols)

    return pol_snapped
