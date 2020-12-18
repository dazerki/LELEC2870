# general imports
import numpy as np
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
from sklearn.feature_selection import mutual_info_regression

# imblearn imports
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler

# custom imports
from MRMR import MRMR
from Selector import Selector
from Remove_outliers import remove_outliers
from Upsample import UpSampling
from Downsample import DownSampling
from output import printOuput
from Visualizer import visualize, feature_distributions


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
                 target_correlations, target_muInf, test_ratio=0.3, seed=1998):

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
        self.target_muInf = target_muInf
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
        printOuput(output, target)

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

            elif preProcessMethod == 'select_corr':
                steps.append(('select_corr', Selector(score_func=np.corrcoef,
                                                      info_vector=self.target_correlations,
                                                      random_state=self.seed)))

            # MutualInfoSelection args :
            # n_components, the number of components with the highest mutual info to keep, default to 20
            elif preProcessMethod == 'select_mut':
                steps.append(('select_mut', Selector(score_func=mutual_info_regression,
                                                     info_vector=self.target_muInf,
                                                     random_state=self.seed)))

            # Whitening is equivalent to applying PCA on all the variables and scaling the variables to unit variance
            # Whitening args :
            # n_components, number of components to identify with PCA, default to all components
            elif preProcessMethod == 'whitening':
                steps.append(('whitening', PCA(whiten=True, random_state=self.seed)))

            # No args
            elif preProcessMethod == 'standardization':
                steps.append(('standardization', StandardScaler()))

            # PCA args :
            # n_components, number of components to identify with PCA, default to all components
            # whiten, boolean indicating if PCA has to use whitening before applying the algorithm
            # random_state, seed to have reproducible results
            elif preProcessMethod == 'PCA':
                steps.append(('pca', PCA(random_state=self.seed)))

            # No args, always applied before the other methods !
            elif preProcessMethod == 'outliers':
                steps.append(('outliers', FunctionSampler(func=remove_outliers)))

            # No args
            elif preProcessMethod == 'upsample':
                steps.append(('upsampling', UpSampling()))

            # No args
            elif preProcessMethod == 'downsample':
                steps.append(('downsampling', DownSampling()))

             # other pre-processing steps ?
            else:
                continue

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
    # WARNING : if binary_fig is true it will output 15 figures
    #           if continuous_fig is true it will output 11 figures
    #           if both are true it will output 26 figures
    classRepartition(Y1)  # print number of articles per class
    visualize(X1, Y1, keys, binary_fig=False, continuous_fig=False)  # print some figures to visualize the data
    feature_distributions(X1, keys, visual=False)  # turn visual to true to see the figures (3 figures * 44 features)

    # which ratio of the data set we use for testing
    test_ratio = 0.2

    # seed for reproducible results
    seed = 1998

    # pre-computation of correlation matrices between features on the whole data set and between features and target
    # for the labelled data set X1 (only done if needed, see below)
    feature_correlations = None
    target_correlations = None

    # pre-computation of mutual information matrix between features and the target (only done if needed, see below)
    target_muInf = None


    # which methods we want to train (linear, KNN, MLP), be careful about the computation time
    # example : methods = ['linear', 'KNN', 'MLP', ...]
    methods = []
    methods.append('linear')
    # methods.append('KNN')
    #methods.append('MLP')

    # which pre-processing steps to apply for each method : one list per method to allow to specify more than one
    # pre-processing step for each method
    preProcessing = []
    preProcessing.append(['outliers', 'select_mut'])  # dummy elements in case of no pre-processing
    # preProcessing.append(['PCA'])  # for linear regression
    # preProcessing.append(['standardization'])  # for KNN
    #preProcessing.append(['whitening', 'upsample'])  # for MLP

    for i in range(len(preProcessing)):
        if preProcessing[i].__contains__('select_corr'):

            # pre-computation of correlation matrices between features on the whole data set and between features and
            # target for the labelled data set X1
            feature_correlations = np.corrcoef(np.concatenate((X1, X2), axis=0), rowvar=False)
            target_correlations = np.corrcoef(X1, Y1, rowvar=False)[:-1, -1]

        if preProcessing[i].__contains__('select_mut'):

            # pre-computation of mutual information matrix between features and the target
            target_muInf = mutual_info_regression(X1, np.ravel(Y1), random_state=seed)

    for i, method in enumerate(methods):

        print("\n========== TRAINING MODEL : {} ==========".format(method))

        # Build the trainer
        trainer = ModelTrainer(X1, Y1, keys, method, preProcessing[i], scoreregression, feature_correlations,
                               target_correlations, target_muInf, test_ratio=test_ratio, seed=seed)

        # Add the pre-processing steps to the pipeline
        print("Start of pre-processing ...", end="")
        trainer.addPreProcess()
        print("End of pre-processing.")

        # Define the parameters to train, including those of the pre-processing steps you added
        if method == 'linear':

            # for outliers the argument kw_args is used an need to receive dictionaries with argument names as keys
            # and values to test as values
            # example : 'outliers__kw_args': [{'below': 0.01*i, 'above': 0.01*i} for i in range(5)]
            grid_params = {
                'outliers__kw_args': [{'below': 0.01*i, 'above': 0.01*i} for i in range(51)],
                'select_mut__k': range(50, 58)
            }

        elif method == 'KNN':

            grid_params = {
                'regression__n_neighbors': np.arange(2,30),
                'regression__weights': ['distance', 'uniform'], #, 'uniform'
                'regression__metric': ['euclidean', 'manhattan', 'chebyshev','minkowski', 'cosine'],
                'regression__leaf_size': [2] # 'euclidean', 'manhattan', 'chebyshev','minkowski', 'cosine','euclidean', 'manhattan', 'chebyshev'
            }
            # Best parameters : {'regression__leaf_size': 2, 'regression__metric': 'manhattan', 'regression__n_neighbors': 48, 'regression__weights': 'distance'}
            # Testing result for the KNN method : 0.528

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
        print("Testing result for the {} method : {:.3f}\n".format(method, result))

        # Output class repartition
        trainer.outputClassRepartition()
