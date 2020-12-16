# general imports
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    pass
import pandas as pd
import sklearn

# sklearn imports
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
# from sklearn.pipeline import Pipeline

# custom imports
from MRMR import MRMR
from Mutual_Info_Selection import MutualInfoSelection
from Remove_outliers import RemoveOutliers
from Upsample import UpSample
from Downsample import DownSampling


def scoref1(ytrue, ypred, th):
    return sklearn.metrics.f1_score(ytrue > th, ypred > th, zero_division=0)


def scoreregression(ytrue, ypred):
    scores = [
    scoref1(ytrue, ypred, th=th) for th in [500, 1400, 5000, 10000]
    ]
    return np.mean(scores)


def classRepartition(target):
    flop = sum(target < 500)[0]
    mild_success = sum(np.logical_and(500 <= target, target < 1400))[0]
    success = sum(np.logical_and(1400 <= target, target < 5000))[0]
    great_success = sum(np.logical_and(5000 <= target, target < 10000))[0]
    viral = sum(target >= 10000)[0]
    print("Number of flop articles : {}".format(flop))
    print("Number of mild success articles : {}".format(mild_success))
    print("Number of success articles : {}".format(success))
    print("Number of great success articles : {}".format(great_success))
    print("Number of viral articles : {}".format(viral))


class ModelTrainer:

    def __init__(self, data, target, keys, modelType, preProcessingList, scoringFunction, feature_correlations,
                 target_correlations, test_ratio=0.3, seed=1998):

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_ratio, random_state=seed)

        self.data = X_train
        self.target = y_train
        self.keys = keys
        self.preProcessingList = preProcessingList
        self.scoringFunction = scoringFunction
        self.modelType = modelType
        self.evalData = X_test
        self.evalTarget = y_test
        self.test_ratio = test_ratio
        self.seed = seed
        self.feature_correlations = feature_correlations
        self.target_correlations = target_correlations
        self.training_score = -1
        self.best_model = None
        self.trained_params = None

        if modelType == 'linear':
            self.model = linear_model.LinearRegression()
        elif modelType == 'KNN':
            self.model = KNeighborsRegressor()
        elif modelType == 'MLP':
            self.model = MLPRegressor(random_state=self.seed, max_iter=500)

    def outputClassRepartition(self):
        output = self.model.predict(self.evalData)
        target = self.evalTarget

        if len(output.shape) > 1:
            output = output[:, 0]
        if len(target.shape) > 1:
            target = target[:, 0]

        flop_hit = np.size(np.where(np.logical_and(target < 500, output < 500)))
        mild_success_hit = np.size(np.where(np.logical_and(np.logical_and(500 <= target, target < 1400),np.logical_and(500 <= output, output < 1400))))
        success_hit = np.size(np.where(np.logical_and(np.logical_and(1400 <= target, target < 5000),np.logical_and(1400 <= output, output < 5000))))
        great_success_hit = np.size(np.where(np.logical_and(np.logical_and(5000 <= target, target < 10000),np.logical_and(5000 <= output, output < 10000))))
        viral_hit = np.size(np.where(np.logical_and(target >= 10000, output >= 10000)))

        flop = sum(output < 500)
        mild_success = sum(np.logical_and(500 <= output, output < 1400))
        success = sum(np.logical_and(1400 <= output, output < 5000))
        great_success = sum(np.logical_and(5000 <= output, output < 10000))
        viral = sum(output >= 10000)

        flop_target = sum(target < 500)
        mild_success_target = sum(np.logical_and(500 <= target, target < 1400))
        success_target = sum(np.logical_and(1400 <= target, target < 5000))
        great_success_target = sum(np.logical_and(5000 <= target, target < 10000))
        viral_target = sum(target >= 10000)

        print("Number of flop articles :\n\t"
              "- In output : {}\n\t"
              "- In target : {}".format(flop, flop_target))
        print("flop hit / flop total: {} / {}".format(flop_hit,flop_target))
        print("Number of mild success articles :\n\t"
              "- In output : {}\n\t"
              "- In target : {}".format(mild_success, mild_success_target))
        print("mild success hit / mild success total: {} / {}".format(mild_success_hit,mild_success_target))
        print("Number of success articles :\n\t"
              "- In output : {}\n\t"
              "- In target : {}".format(success, success_target))
        print("success hit / success total: {} / {}".format(success_hit,success_target))
        print("Number of great success articles :\n\t"
              "- In output : {}\n\t"
              "- In target : {}".format(great_success, great_success_target))
        print("great success hit / great success total: {} / {}".format(great_success_hit,great_success_target))
        print("Number of viral articles :\n\t"
              "- In output : {}\n\t"
              "- In target : {}".format(viral, viral_target))
        print("viral hit / viral total: {} / {}".format(viral_hit,viral_target))

    def visualize(self):

        target = self.target[self.target < 20000]
        data = self.data[(self.target < 20000).T[0], :]

        flop = target < 500
        mild_success = np.logical_and(500 <= target, target < 1400)
        success = np.logical_and(1400 <= target, target < 5000)
        great_success = np.logical_and(5000 <= target, target < 10000)
        viral = target >= 10000

        colors = np.full(len(target), "yellow")
        colors[flop] = "blue"
        colors[mild_success] = "green"
        colors[success] = "yellow"
        colors[great_success] = "orange"
        colors[viral] = "red"

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

        plt.rcParams["figure.figsize"] = (10, 10)

        # each feature wrt the target
        # """
        for i in range(int(np.ceil(len(data[0]) / 4))):
            fig, axs = plt.subplots(nrows=2, ncols=2)
            for j in range(min(4, 58 - 4 * i)):
                axs[j // 2, j % 2].scatter(target, data[:, i * 4 + j], c=colors)
                axs[j // 2, j % 2].set_title(self.keys[i * 4 + j])
            plt.show()
        # """

        # each feature PER CLASS wrt the target
        """
        for i in range(len(data[0])):
            fig, axs = plt.subplots(nrows=2, ncols=3)
            fig.suptitle("Feature n° {} : {}".format(i+1, self.keys[i]))
            for j in range(5):
                col = np.tile(colors2[j], len(target_per_label[j][:]))
                axs[j//3, j%3].scatter(target_per_label[j][:], data_per_label[j][:, i], c=col)
                axs[j//3, j%3].set_title(labels[j])
            plt.show()
        """

    # Pre-processing of the data
    def addPreProcess(self):

        steps = []

        for preProcessMethod in self.preProcessingList:

            # MRMR args :
            # feature_correlations is the precomputed correlation matrix between the features, can't grid search on it
            # => set it once and for all
            # target_correlations is the precomputed correlation matrix between the features and the target,
            # can't grid search on it => set it once and for all
            # n_components, number of components with highest correlation with target, default to 20
            # thresh, the threshold for the correlation above which we have to remove one feature, default to 1.0
            if preProcessMethod == 'mrmr':
                feature_corr = self.feature_correlations.copy()
                target_corr = self.target_correlations.copy()
                steps.append(('mrmr', MRMR(feature_corr, target_corr)))

            # MutualInfoSelection args :
            # n_components, the number of components with the highest mutual info to keep, default to 20
            elif preProcessMethod == 'mutual':
                steps.append(('mutual', MutualInfoSelection()))

            elif preProcessMethod == 'whitening':
                continue

            # No args
            elif preProcessMethod == 'standardization':
                steps.append(('standardization', StandardScaler()))

            # PCA args :
            # n_components, number of components to identify with PCA, default to all components
            # whiten, boolean indicating if PCA has to use whitening before applying the algorithm
            # random_state, seed to have reproducible results
            elif preProcessMethod == 'PCA':
                steps.append(('pca_standardization', StandardScaler()))
                steps.append(('pca', PCA(random_state=self.seed)))

            # No args, always applied before the other methods !
            elif preProcessMethod == 'outliers':
                ro = RemoveOutliers()
                self.data, self.target = ro.transform(self.data, self.target)

            # No args, always applied before the other methods !
            elif preProcessMethod == 'upsample':
                up = UpSample()
                self.data, self.target = up.transform(self.data, self.target, self.seed)

            # No args, always applied before the other methods !
            elif preProcessMethod == 'downsample':
                continue
                # down = DownSample()
                # self.data, self.target = down.transform(self.data, self.target, self.seed)

             # other pre-processing steps ?
            else:
                continue

        steps.append(('sampling', DownSampling()))
        steps.append(('regression', self.model))
        self.model = Pipeline(steps=steps)

    # Training of the model
    def train(self, parametersGrid=None):

        if parametersGrid is not None:
            score_func = sklearn.metrics.make_scorer(self.scoringFunction)
            self.model = GridSearchCV(
                self.model,
                parametersGrid,
                verbose=0,
                cv=3,
                n_jobs=-1,
                scoring=score_func
            )

        if self.modelType == 'MLP':
            self.model.fit(self.data, np.ravel(self.target))
        else:
            self.model.fit(self.data, self.target)
            if parametersGrid is not None:
                self.training_score = self.model.best_score_
                self.best_model = self.model.best_params_
                self.trained_params = self.model.param_grid

    # Evaluation of the pipeline
    def evaluate(self):

        # Predict targets of unseen data
        predictions = self.model.predict(self.evalData)

        # Compute score
        result = self.scoringFunction(self.evalTarget, predictions)

        return self.training_score, self.trained_params, self.best_model, result


if __name__ == "__main__":

    # Use pandas to load into a DataFrame
    # Y1.csv doesn’t have a header so
    # add one when loading the file

    X1 = pd.read_csv('X1.csv')
    Y1 = pd.read_csv('Y1.csv', header=None, names=['shares'])
    X2 = pd.read_csv('X2.csv')
    keys = X1.keys()

    # To work with numpy arrays :
    X1 = X1.values
    Y1 = Y1.values
    X2 = X2.values

    # Visualization
    classRepartition(Y1)  # print number of articles per class

    # which ratio of the data set we use for testing
    test_ratio = 0.2
    # seed for reproducible results
    seed = 1998
    # pre-computation of correlation matrices between features on the whole data set and between features and target
    # for the labelled data X1
    feature_correlations = np.corrcoef(np.concatenate((X1, X2), axis=0), rowvar=False)
    target_correlations = np.corrcoef(X1, Y1, rowvar=False)[:-1, -1]

    # which methods we want to train (linear, KNN, MLP), be careful about the computation time
    # example : methods = ['linear', 'KNN', 'MLP', ...]
    methods = []
    # methods.append('linear')
    methods.append('KNN')
    # methods.append('MLP')

    # which pre-processing steps to apply for each method : one list per method to allow to specify more than one
    # pre-processing step for each method
    preProcessing = []
    #preProcessing.append([])  # dummy elements in case of no pre-processing
    # preProcessing.append(['PCA'])  # for linear regression
    preProcessing.append(['standardization'])  # for KNN
    # preProcessing.append(['whitening'])  # for MLP

    for i, method in enumerate(methods):

        print("========== TRAINING MODEL : {} ==========".format(method))

        # Build the trainer
        trainer = ModelTrainer(X1, Y1, keys, method, preProcessing[i], scoreregression, feature_correlations,
                               target_correlations, test_ratio=test_ratio, seed=seed)

        # Visualize data
        # trainer.visualize()

        # Add the pre-processing steps to the pipeline
        print("Start of pre-processing ...", end="")
        trainer.addPreProcess()
        print("End of pre-processing.")

        # Define the parameters to train, including those of the pre-processing steps you added
        if (method == 'linear' and not preProcessing[i].__contains__('mrmr') and
                not preProcessing[i].__contains__('mutual') and not preProcessing[i].__contains__('PCA')):

            grid_params = None

        elif method == 'linear':

            grid_params = {
                'pca__n_components': np.arange(1, 59, 1),
                'pca__whiten': [True, False]
            }

        elif method == 'KNN':

            grid_params = {
                'regression__n_neighbors': [24],
                'regression__weights': ['distance'], #, 'uniform'
                'regression__metric': [ 'minkowski'],
                'regression__leaf_size': [2] # 'euclidean', 'manhattan', 'chebyshev','minkowski', 'cosine','euclidean', 'manhattan', 'chebyshev'
            }
            # {'metric': 'cosine', 'n_neighbors': 10, 'weights': 'uniform'}
            # 0.5008216155210505

        else:
            grid_params = None

        # Train the model
        print("Start of training ...", end="")
        trainer.train(parametersGrid=grid_params)
        print("End of training.")

        # Evaluate the model after training
        print("Start of evaluation ...", end="")
        train_result, trained_params, best_model, result = trainer.evaluate()
        print("End of evaluation.")

        # Print results
        if train_result != -1:
            print("Best training result for the {} method : {:.3f}".format(method, train_result))
            print("Trained parameters : {}".format(trained_params))
            print("Best parameters : {}".format(best_model))
        print("Testing result for the {} method : {:.3f}".format(method, result))

        # Output class repartition
        trainer.outputClassRepartition()
