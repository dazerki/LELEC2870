import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin


class Selector(BaseEstimator, SelectorMixin):

    def __init__(self, score_func, labels, info_vector, k=20, random_state=1998):
        self.score_func = score_func
        self.k = k
        self.random_state = random_state
        self.info_vector = info_vector
        self.labels = labels

    def fit(self, X, y):

        self.scores_ = self.info_vector

        return self

    def _get_support_mask(self):

        if self.k <= 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = self.scores_
            mask = np.zeros(scores.shape, dtype=bool)
            mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1
            return mask
