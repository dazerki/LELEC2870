import numpy as np


def remove_outliers(X, y, below=0.0, above=0.0):
    sorted_y = np.sort(y.copy())
    thresh_below = sorted_y[int(below*len(y))]
    thresh_above = sorted_y[-int(above*len(y))-1]
    mask = np.logical_and(y >= thresh_below, y <= thresh_above)  # which data we have to keep is between the thresholds

    return X[mask, :], y[mask]
