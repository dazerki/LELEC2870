import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin


class MutualInfoSelection(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=20):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        y_ = y.copy()
        X_ = self.preProcessMutualInf(X_, y_, n_components=self.n_components)

        return X_

    def preProcessMutualInf(self, X, Y, n_components=20):
        muInf = mutual_info_regression(X, np.ravel(Y))
        if n_components is not None:
            ind = np.argsort(muInf)
            muInf_ind = ind[-n_components:]
        else:
            mean_muInf = np.mean(np.abs(muInf))
            muInf_ind = np.where(muInf > mean_muInf)[0]  # numpy.where returns array of the result => take the result

        return X[:, muInf_ind]
