import numpy as np
import random


class DownSample:

    def __init__(self):
        pass

    def transform(self, X, y, seed):
        X_ = X.copy()
        y_ = y.copy()
        X_, y_ = self.__downsample(X_, y_, seed)

        return X_, y_

    # Make the function private with __ in front of it
    def __downsample(self, X, y, seed):

        random.seed(seed)

        data = np.concatenate((X, y), axis=1)

        flop = data[(y < 500)[:, 0]]
        mild_success = data[(np.logical_and(500 <= y, y < 1400))[:, 0]]
        success = data[(np.logical_and(1400 <= y, y < 5000))[:, 0]]
        great_success = data[(np.logical_and(5000 <= y, y < 10000))[:, 0]]
        viral = data[(y >= 10000)[:, 0]]

        mini = np.min([len(flop), len(mild_success), len(success), len(great_success), len(viral)])
        if len(flop) != mini:
            flop = random.choices(flop, k=mini)
        if len(mild_success) != mini:
            mild_success = random.choices(mild_success, k=mini)
        if len(success) != mini:
            success = random.choices(success, k=mini)
        if len(great_success) != mini:
            great_success = random.choices(great_success, k=mini)
        if len(viral) != mini:
            viral = random.choices(viral, k=mini)

        data = np.concatenate((flop, mild_success, success, great_success, viral), axis=0)
        random.shuffle(data)

        return data[:, :-1], data[:, -1]
