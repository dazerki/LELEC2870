import numpy as np

from imblearn.base import BaseSampler
from imblearn.under_sampling import RandomUnderSampler

class DownSampling(BaseSampler):
    def __init__(self, random_state=None, **kwargs):
        super().__init__()
        self.random_state = random_state
        self.sampler = RandomUnderSampler(random_state=self.random_state)
        self._sampling_type = self.sampler._sampling_type

    def _fit_resample(self, X, y):
        X_new = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        y_new = np.zeros(y.shape)
        y_new[np.where(y >= 10000)] = 4
        y_new[np.where(y >= 5000)] = 3
        y_new[np.where(y >= 1400)] = 2
        y_new[np.where(y >= 500)] = 1
        y_new[np.where(y < 500)] = 0

        X_res_down, y_res_down = self.sampler.fit_resample(X_new, y_new)
        return X_res_down[:, :-1], X_res_down[:, -1]
