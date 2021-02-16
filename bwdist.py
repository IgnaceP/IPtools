from scipy.ndimage import morphology

def bwdist(arr):
    """ Function to perform the Matlab bwdist function to a binary Numpy array """
    return morphology.distance_transform_edt(arr==0)
