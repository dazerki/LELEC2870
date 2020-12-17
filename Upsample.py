import numpy as np
import random
import sklearn
from sklearn.utils import resample

from imblearn.base import BaseSampler
from imblearn.over_sampling import RandomOverSampler


# class UpSample:
#
#     def __init__(self):
#         pass
#
#     def transform(self, X, y, seed):
#         X_ = X.copy()
#         y_ = y.copy()
#         X_, y_ = self.__upsample(X_, y_, seed)
#
#         return X_, y_
#
#     # Make the function private with __ in front of it
#     def __upsample(self, X, y, seed):
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
#         maxi = np.max([len(flop), len(mild_success), len(success), len(great_success), len(viral)])
#         if len(flop) != maxi:
#             flop = np.concatenate((flop, random.choices(flop, k=maxi-len(flop))), axis=0)
#         if len(mild_success) != maxi:
#             mild_success = np.concatenate((mild_success, random.choices(mild_success, k=maxi-len(mild_success))), axis=0)
#         if len(success) != maxi:
#             success = np.concatenate((success, random.choices(success, k=maxi-len(success))), axis=0)
#         if len(great_success) != maxi:
#             great_success = np.concatenate((great_success, random.choices(great_success, k=maxi-len(great_success))), axis=0)
#         if len(viral) != maxi:
#             viral = np.concatenate((viral, random.choices(viral, k=maxi-len(viral))), axis=0)
#
#         data = np.concatenate((flop, mild_success, success, great_success, viral), axis=0)
#         random.shuffle(data)
#
#         return data[:, :-1], data[:, -1]

class UpSampling(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__()
        self.sampler = RandomOverSampler()
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
