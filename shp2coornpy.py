from pol2coornpy import *
from shp2mpol import *
from shapely.geometry import shape as shapshape
from shapely.geometry import Polygon as shPol
from shapely.geometry import MultiPolygon as shMPol
from shapely.geometry import MultiPoint as shMPoint

def shp2CoorNpy(fn, shptype = 'polygon', print_coordinate_system = False, print_n_features = False, get_fields = False):
    """
    Method to translate a shapefile to a numpy array with all coordinates of the shapefile.s

    Args:
        fn: (Required) path string to shapefile
    Returns:
        xy: numpy array with column for x and column for y coordinates
    """

    # check if file exists
    if not os.path.isfile(fn):
        raise ValueError('Wrong inputfile!')

    if shptype == 'polygon':
        if get_fields:
            mpol, fields = shp2Mpol(fn, print_coordinate_system = print_coordinate_system, print_n_features = print_n_features, get_fields = True)
        else:
            mpol = shp2Mpol(fn, print_coordinate_system = print_coordinate_system, print_n_features = print_n_features)

        xy = pol2CoorNpy(mpol)

    elif shptype.startswith('point'):

        # load shapefile with the ogr toolbox of osgeo
        file = ogr.Open(fn)
        shape = file.GetLayer(0)

        epsg = int(shape.GetSpatialRef().ExportToPrettyWkt().splitlines()[-1].split('"')[3])
        crs = CRS.from_epsg(epsg)
        if print_coordinate_system:
            print("The EPSG code of the coordinate system is: %d" % (crs.to_epsg()))
        # get number of polygons in shapefile
        n_features = shape.GetFeatureCount()

        if print_n_features:
            print('There are %d features in the shapefile.' % n_features)

        # initialize new polygon list
        points = np.zeros([n_features, 2])
        fields = []

        # loop over all polygons
        for i in range(n_features):
            print('Progress: %.2f %%' % (i/n_features * 100), end="\r", flush=True)
            # get feature object
            feature = shape.GetFeature(i)
            # export to JS objects
            feature_JSON = feature.ExportToJson()
            feature_JSON = json.loads(feature_JSON)

            if get_fields:
                f = [feature.GetField(j) for j in get_fields]
                f = np.asarray(f)
                fields.append(f)

            # get the point coordinates
            p = shapshape(feature_JSON["geometry"])
            if type(p) == shMPoint:
                if len(p) == 1: p = p[0]
            points[i,:] = p.x, p.y

        xy = points

    if get_fields: return xy, fields
    else: return xy
