import numpy as np


class RemoveOutliers:

    def __init__(self):
        pass

    def transform(self, X, y):
        X_ = X.copy()
        y_ = y.copy()
        X_, y_ = self.__remove_outliers(X_, y_)

        return X_, y_

    # Make the function private with __ in front of it
    def __remove_outliers(self, X, y):
        mean = np.mean(y)
        std = np.std(y)
        mask = np.bitwise_and(mean - std <= y, y <= mean + std)[:, 0]

        return X[mask, :], y[mask]
