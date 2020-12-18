import numpy as np
import matplotlib.pyplot as plt


def feature_distributions(data_, keys_, visual=False):

    if not visual:
        return

    binaries = np.zeros(data_.shape[1])
    for i in range(data_.shape[1]):
        if len(np.unique(data_[:, i])) == 2:
            binaries[i] = 1
        else:
            binaries[i] = 0
    binaries = binaries.astype(bool)

    keys = keys_[np.logical_not(binaries)]
    data = data_[:, np.logical_not(binaries)]

    print(data.shape[1])

    for i in range(data.shape[1]):
        values = data[:, i]
        values.sort()
        counts = []
        uniques = []
        toEqual = values[0]
        count = 1
        for j in range(1, len(values)):
            if toEqual == values[j]:
                count += 1
            else:
                counts.append(count)
                uniques.append(toEqual)
                count = 1
                toEqual = values[j]
        y = np.linspace(0.0, 1.0, len(values))
        plt.figure()
        plt.scatter(values, y)
        plt.title('cdf of feature : {}'.format(keys[i]))
        plt.figure()
        plt.scatter(np.log1p(values), y)
        plt.title('log_cdf of feature : {}'.format(keys[i]))
        plt.figure()
        plt.scatter(np.array(uniques), np.array(counts)/len(values))
        plt.title('pdf of feature : {}'.format(keys[i]))
        plt.show()


def visualize(data_, target_, keys_, binary_fig=True, continuous_fig=True):

    binaries = np.zeros(data_.shape[1])
    for i in range(data_.shape[1]):
        if len(np.unique(data_[:, i])) == 2:
            binaries[i] = 1
        else:
            binaries[i] = 0
    binaries = binaries.astype(bool)

    target = target_[target_ < 20000]
    data = data_[(target_ < 20000).T[0], :]

    flop = target < 500
    mild_success = np.logical_and(500 <= target, target < 1400)
    success = np.logical_and(1400 <= target, target < 5000)
    great_success = np.logical_and(5000 <= target, target < 10000)
    viral = target >= 10000

    if binary_fig:

        data_binary = data[:, binaries]
        keys_binary = keys_[binaries]
        data_channel = data_binary[:, :6]
        keys_channel = [key[16:] for key in keys_binary[:6]]
        data_day = data_binary[:, 6:-1]
        keys_day = [key[11:] for key in keys_binary[6:-1]]
        data_weekend = data_binary[:, -1]

        all = np.array([flop, mild_success, success, great_success, viral])
        labels = np.array(['flops', 'milds', 'succs', 'greats', 'virals'])
        labels_pourc = np.array(['% flops/day', '% milds/day', '% succs/day', '% greats/day', '% virals/day'])

        for i in range(5):
            plt.figure(figsize=(5, 4))
            first = sum(np.logical_and(data_channel[:, 0], all[i])) / sum(all[i])
            second = sum(np.logical_and(data_channel[:, 1], all[i])) / sum(all[i])
            third = sum(np.logical_and(data_channel[:, 2], all[i])) / sum(all[i])
            fourth = sum(np.logical_and(data_channel[:, 3], all[i])) / sum(all[i])
            fifth = sum(np.logical_and(data_channel[:, 4], all[i])) / sum(all[i])
            sixth = sum(np.logical_and(data_channel[:, 5], all[i])) / sum(all[i])
            numbers = np.array([first, second, third, fourth, fifth, sixth])
            plt.bar(range(0, 6), numbers, tick_label=keys_channel)
            plt.xticks(range(0, 6), keys_channel, rotation=45)
            plt.title(labels[i])
        plt.show()

        for i in range(5):
            plt.figure(figsize=(5, 4))
            first = sum(np.logical_and(data_day[:, 0], all[i])) / sum(all[i])
            second = sum(np.logical_and(data_day[:, 1], all[i])) / sum(all[i])
            third = sum(np.logical_and(data_day[:, 2], all[i])) / sum(all[i])
            fourth = sum(np.logical_and(data_day[:, 3], all[i])) / sum(all[i])
            fifth = sum(np.logical_and(data_day[:, 4], all[i])) / sum(all[i])
            sixth = sum(np.logical_and(data_day[:, 5], all[i])) / sum(all[i])
            seventh = sum(np.logical_and(data_day[:, 6], all[i])) / sum(all[i])
            numbers = np.array([first, second, third, fourth, fifth, sixth, seventh])
            plt.bar(range(0, 7), numbers)
            plt.xticks(range(0, 7), keys_day, rotation=45)
            plt.title(labels[i])
        plt.show()

        for i in range(5):
            plt.figure(figsize=(5, 4))
            first = sum(
                np.logical_or(
                    np.logical_or(
                        np.logical_or(
                            np.logical_or(
                                np.logical_and(data_day[:, 0], all[i]),
                                np.logical_and(data_day[:, 1], all[i])
                            ),
                            np.logical_and(data_day[:, 2], all[i])
                        ),
                        np.logical_and(data_day[:, 3], all[i])
                    ),
                    np.logical_and(data_day[:, 4], all[i])
                )
            ) / sum(all[i]) / 5
            second = sum(np.logical_and(data_weekend, all[i])) / sum(all[i]) / 2
            numbers = np.array([first, second])
            plt.bar(range(0, 2), numbers)
            plt.xticks(range(0, 2), ['weekday', 'weekend'], rotation=45)
            plt.title(labels_pourc[i])
        plt.show()

    if continuous_fig:

        data = data[:, np.logical_not(binaries)]
        keys = keys_[np.logical_not(binaries)]

        colors = np.full(len(target), "yellow")
        colors[flop] = "blue"
        colors[mild_success] = "green"
        colors[success] = "yellow"
        colors[great_success] = "orange"
        colors[viral] = "red"

        plt.rcParams["figure.figsize"] = (10, 10)

        # each non binary feature wrt the target
        nb_feat = len(data[0])
        nrows = 2
        ncols = 2
        size = 4
        for i in range(int(np.ceil(nb_feat / size))):
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
            for j in range(min(size, nb_feat - size * i)):
                axs[j // ncols, j % ncols].scatter(target, data[:, i * size + j], c=colors)
                axs[j // ncols, j % ncols].set_title(keys[i * size + j])
            plt.show()

        # each feature PER CLASS wrt the target
        """
        flops = data[flop, :]
        mild_successes = data[mild_success, :]
        successes = data[success, :]
        great_successes = data[great_success, :]
        virals = data[viral, :]

        flops_target = target[flop]
        mild_successes_target = target[mild_success]
        successes_target = target[success]
        great_successes_target = target[great_success]
        virals_target = target[viral]

        labels = ['flop', 'mild', 'success', 'great', 'viral']
        colors2 = ["blue", "green", "yellow", "orange", "red"]
        data_per_label = np.array([flops, mild_successes, successes, great_successes, virals])
        target_per_label = np.array(
            [flops_target, mild_successes_target, successes_target, great_successes_target, virals_target])        

        for i in range(len(data[0])):
            fig, axs = plt.subplots(nrows=2, ncols=3)
            fig.suptitle("Feature n° {} : {}".format(i+1, keys_[i]))
            for j in range(5):
                col = np.tile(colors2[j], len(target_per_label[j][:]))
                axs[j//3, j%3].scatter(target_per_label[j][:], data_per_label[j][:, i], c=col)
                axs[j//3, j%3].set_title(labels[j])
            plt.show()
        """