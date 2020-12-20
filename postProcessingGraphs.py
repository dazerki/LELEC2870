import numpy as np
import matplotlib.pyplot as plt


def output_results(results, visual=False):

    if not visual:
        return

    params = results['params'][5:len(results['params']):6]
    mean_test_score = results['mean_test_score'][5:len(results['params']):6]
    labels = [str(params[i]['mrmr__n_components']) for i in range(len(params))]

    plt.figure()
    plt.scatter(np.linspace(1, len(params), len(params)), mean_test_score)
    plt.xticks(np.linspace(1, len(params), len(params)), labels, rotation='vertical')
    plt.title("F1-score wrt the n_components to keep for threshold = 1.0")
    plt.xlabel("Maximum relevance n_components to keep")
    plt.ylabel("F1-score")
    plt.show()

    """
    params = results['params'][36:42]
    mean_test_score = results['mean_test_score'][36:42]
    labels = [str(params[i]['mrmr__thresh']) for i in range(len(params))]

    plt.figure()
    plt.scatter(np.linspace(1, len(params), len(params)), mean_test_score)
    plt.xticks(np.linspace(1, len(params), len(params)), labels, rotation='vertical')
    plt.title("F1-score wrt the correlation threshold for n_components = 14")
    plt.xlabel("Minimum redundancy correlation threshold")
    plt.ylabel("F1-score")
    plt.show()
    """
