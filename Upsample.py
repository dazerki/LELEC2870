import numpy as np

from imblearn.base import BaseSampler
from imblearn.over_sampling import RandomOverSampler

class UpSampling(BaseSampler):
    def __init__(self, random_state=None, **kwargs):
        super().__init__()
        self.random_state = random_state
        self.sampler = RandomOverSampler(random_state=self.random_state)
        self._sampling_type = self.sampler._sampling_type

    def _fit_resample(self, X, y):
        Xmod = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        ymod = np.zeros(y.shape)
        ymod[np.where(y >= 500)] = 1
        ymod[np.where(y >= 1400)] = 2
        ymod[np.where(y >= 5000)] = 3
        ymod[np.where(y >= 10000)] = 4

        X_down, y_down = self.sampler.fit_resample(Xmod, ymod)
        return X_down[:, :-1], X_down[:, -1]
