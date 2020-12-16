import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MRMR(BaseEstimator, TransformerMixin):

    def __init__(self, feature_correlations, target_correlations, n_components=20, thresh=1.0):
        n = len(target_correlations)
        self.n_components = n_components
        self.thresh = thresh
        self.feature_correlations = feature_correlations
        self.target_correlations = target_correlations

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        Xmr, corr, target_corr = self.maximumRelevance(X_, self.n_components)
        Xmrmr = self.minimumRedundancy(Xmr, np.abs(corr), np.abs(target_corr), thresh=self.thresh)

        return Xmrmr

    def maximumRelevance(self, X, n_components=None):
        target_corr = self.target_correlations.copy()
        features_corr = self.feature_correlations.copy()
        if n_components is not None:
            ind = np.argsort(target_corr)
            target_corr_ind = ind[-n_components:]
        else:
            mean_corr = np.mean(np.abs(target_corr))
            # numpy.where returns array of the result => take the result
            target_corr_ind = np.where(np.abs(target_corr) > mean_corr)[0]

        return X[:, target_corr_ind], features_corr[target_corr_ind, :][:, target_corr_ind], target_corr

    def minimumRedundancy(self, X, corr, target_corr, thresh=1.0):
        size = corr.shape[0]  # used to keep at least one feature
        toKeep = np.ones(corr.shape, dtype=bool)
        for i in range(len(corr)):
            corr_i = corr[i, :]
            inds = np.flip(np.argsort(corr_i))
            j = 0
            while j < len(inds) and corr_i[inds[j]] >= thresh:
                if i != inds[j] and size-1 > 0:
                    size -= 1
                    # Keep most relevant feature and remove the other to avoid redundancy
                    if target_corr[i] >= target_corr[inds[j]]:
                        toKeep[inds[j], :] = False
                        toKeep[:, inds[j]] = False
                    else:
                        toKeep[i, :] = False
                        toKeep[:, i] = False
                j += 1
        mask = np.argmax(sum(toKeep))  # find a line where we have at least one true to extract the indices to keep
        non_redundant_ind = np.where(toKeep[mask])[0]
        return X[:, non_redundant_ind]