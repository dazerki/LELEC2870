import numpy as np
import random
import sklearn
from sklearn.utils import resample

from imblearn.base import BaseSampler
from imblearn.under_sampling import RandomUnderSampler


# class DownSample:
#
#     def __init__(self):
#         pass
#
#     def transform(self, X, y, seed):
#         X_ = X.copy()
#         y_ = y.copy()
#         X_, y_ = self.__downsample(X_, y_, seed)
#
#         return X_, y_
#
#     # Make the function private with __ in front of it
#     def __downsample(self, X, y, seed):
#
#         random.seed(seed)
#
#         data = np.concatenate((X, y), axis=1)
#
#         flop = data[(y < 500)[:, 0]]
#         mild_success = data[(np.logical_and(500 <= y, y < 1400))[:, 0]]
#         success = data[(np.logical_and(1400 <= y, y < 5000))[:, 0]]
#         great_success = data[(np.logical_and(5000 <= y, y < 10000))[:, 0]]
#         viral = data[(y >= 10000)[:, 0]]
#
#         mini = np.min([len(flop), len(mild_success), len(success), len(great_success), len(viral)])
#         if len(flop) != mini:
#             #flop = random.choices(flop, k=mini)
#             flop = resample(flop, n_samples=mini, random_state=0, stratify=flop)
#         if len(mild_success) != mini:
#             mild_success = resample(mild_success, n_samples=mini, random_state=0, stratify=mild_success)
#         if len(success) != mini:
#             success = resample(success, n_samples=mini, random_state=0, stratify=success)
#         if len(great_success) != mini:
#             great_success = resample(great_success, n_samples=mini, random_state=0, stratify=great_success)
#         if len(viral) != mini:
#             viral = resample(viral, n_samples=mini, random_state=0, stratify=viral)
#
#         data = np.concatenate((flop, mild_success, success, great_success, viral), axis=0)
#         random.shuffle(data)
#
#         return data[:, :-1], data[:, -1]

class DownSampling(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__()
        self.sampler = RandomUnderSampler()
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
