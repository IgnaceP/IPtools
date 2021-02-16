import pandas as pd
import numpy as np

def csv2npy(fn, fn_to_save_to, col_names):
    """
    Function to translate a csv (e.g. from QGIS) to a headless Numpy array

    """

    df = pd.read_csv(fn)

    arr = np.zeros([len(df), len(col_names)])

    t = 0
    for col in col_names:
        c = df[col].values
        arr[:,t] = c
        t += 1

    arr = arr[~np.isnan(arr).any(axis=1)]

    if fn_to_save_to[-4:] != '.npy': fn_to_save_to += '.npy'
    np.save(fn_to_save_to, arr)
    print('File created in %s' % fn_to_save_to)
