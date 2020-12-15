import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    pass
import pandas as pd
import sklearn
import seaborn as sms
import random

from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def scoref1(ytrue, ypred, th):
    return sklearn.metrics.f1_score(ytrue > th, ypred > th)


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


def preProcessMutualInf(X, Y, n_components=None):
    muInf = mutual_info_regression(X, np.ravel(Y))
    if n_components is not None:
        ind = np.argsort(muInf)
        muInf_ind = ind[-n_components:]
    else:
        mean_muInf = np.mean(np.abs(muInf))
        muInf_ind = np.where(muInf > mean_muInf)[0]  # numpy.where returns array of the result => take the result
    return X[:, muInf_ind]


class MRMR:

    def __init__(self, n_components, thresh):
        self.n_components = n_components
        self.thresh = thresh

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        Xmr, corr, target_corr = self.maximumRelevance(X_, y, self.n_components)
        Xmrmr = self.minimumRedundancy(Xmr, np.abs(corr), np.abs(target_corr), thresh=self.thresh)

        return Xmrmr

    def maximumRelevance(self, X, Y, n_components=None):
        corr = np.corrcoef(X, Y, rowvar=False)
        target_corr = corr[:-1, -1]  # last column/line is correlation between each feature and output
        if n_components is not None:
            ind = np.argsort(target_corr)
            target_corr_ind = ind[-n_components:]
        else:
            mean_corr = np.mean(np.abs(target_corr))
            # numpy.where returns array of the result => take the result
            target_corr_ind = np.where(np.abs(target_corr) > mean_corr)[0]

        return X[:, target_corr_ind], corr[target_corr_ind, :][:, target_corr_ind], target_corr

    def minimumRedundancy(self, X, corr, target_corr, thresh=0.4):
        toKeep = np.ones(corr.shape, dtype=bool)
        for i in range(len(corr)):
            corr_i = corr[i, :]
            inds = np.flip(np.argsort(corr_i))
            j = 0
            while corr_i[inds[j]] >= thresh:
                if i != inds[j]:
                    # Keep most relevant feature and remove the other to avoid redundancy
                    if target_corr[i] >= target_corr[inds[j]]:
                        toKeep[inds[j], :] = False
                        toKeep[:, inds[j]] = False
                    else:
                        toKeep[i, :] = False
                        toKeep[:, i] = False
                j += 1
        mask = np.argmax(sum(toKeep))  # find a line where we have at least one true to extract the indices to keep
        non_redundant_ind = np.where(toKeep[mask])[0]
        return X[:, non_redundant_ind]


class ModelTrainer:

    def __init__(self, data, target, modelType, preProcessingList, scoringFunction):
        self.data = data
        self.target = target
        self. preProcessingList = preProcessingList
        self.scoringFunction = scoringFunction
        self.modelType = modelType
        self.evalData = []
        self.evalTarget = []

        if modelType == 'linear':
            self.model = linear_model.LinearRegression()
        elif modelType == 'KNN':
            self.model = KNeighborsRegressor()
        elif modelType == 'MLP':
            self.model = MLPRegressor(random_state=1, max_iter=500)

    def outputClassRepartition(self):
        output = self.model.predict(self.evalData)
        flop = sum(output < 500)
        mild_success = sum(np.logical_and(500 <= output, output < 1400))
        success = sum(np.logical_and(1400 <= output, output < 5000))
        great_success = sum(np.logical_and(5000 <= output, output < 10000))
        viral = sum(output >= 10000)
        print("Number of flop articles : {}".format(flop))
        print("Number of mild success articles : {}".format(mild_success))
        print("Number of success articles : {}".format(success))
        print("Number of great success articles : {}".format(great_success))
        print("Number of viral articles : {}".format(viral))

    def visualize(self):

        if self.modelType == 'linear':
            plt.figure()
            plt.boxplot(self.target)
            plt.show()

    # Pre-processing of the data
    def preProcess(self, n_components=None, thresh=0.4, min=True):

        for preProcessMethod in self.preProcessingList:

            if preProcessMethod == 'mrmr':
                self.model = Pipeline(steps=[('mrmr', MRMR(n_components, thresh)), ('regression', self.model)])

            elif preProcessMethod == 'mutual':
                self.data = preProcessMutualInf(self.data, self.target, n_components=n_components)
                self.evalData = self.data.copy()
                self.evalTarget = self.target.copy()

            elif preProcessMethod == 'whitening':
                continue

            elif preProcessMethod == 'standardization':
                scaler = preprocessing.StandardScaler().fit(self.data)
                self.data = scaler.transform(self.data)
                self.evalData = self.data.copy()
                self.evalTarget = self.target.copy()

            elif preProcessMethod == 'PCA':
                pca = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=0))
                pca.fit(self.data, self.target)
                self.data = pca.transform(self.data)
                self.evalData = self.data.copy()
                self.evalTarget = self.target.copy()

            elif preProcessMethod == 'outliers':
                mean = np.mean(self.target)
                std = np.std(self.target)
                mask = np.bitwise_and(mean - std <= self.target, self.target <= mean + std)[:, 0]
                self.data = self.data[mask, :]
                self.target = self.target[mask]
                self.evalData = self.data.copy()
                self.evalTarget = self.target.copy()

            elif preProcessMethod == 'equalClassSize':
                data = np.concatenate((self.data, self.target), axis=1)
                flop = data[(self.target < 500)[:, 0]]
                mild_success = data[(np.logical_and(500 <= self.target, self.target < 1400))[:, 0]]
                success = data[(np.logical_and(1400 <= self.target, self.target < 5000))[:, 0]]
                great_success = data[(np.logical_and(5000 <= self.target, self.target < 10000))[:, 0]]
                viral = data[(self.target >= 10000)[:, 0]]

                if min:
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
                else:
                    max = np.max([len(flop), len(mild_success), len(success), len(great_success), len(viral)])
                    if max - len(flop) != 0:
                        flop = np.concatenate((flop, random.choices(flop, k=max-len(flop))), axis=0)
                    if max - len(mild_success) != 0:
                        mild_success = np.concatenate((mild_success, random.choices(mild_success, k=max - len(mild_success))), axis=0)
                    if max - len(success) != 0:
                        success = np.concatenate((success, random.choices(success, k=max - len(success))), axis=0)
                    if max - len(great_success) != 0:
                        great_success = np.concatenate((great_success, random.choices(great_success, k=max - len(great_success))), axis=0)
                    if max - len(viral) != 0:
                        viral = np.concatenate((viral, random.choices(viral, k=max - len(viral))), axis=0)

                data = np.concatenate((flop, mild_success, success, great_success, viral), axis=0)
                random.shuffle(data)
                self.data = data[:, :-1]
                self.target = data[:, -1]

             # other pre-processing steps ?
            else:
                continue

    # Training of the model
    def train(self, training_ratio=0.7, parametersGrid=None):

        # useful values : size = nb of data points, N = nb of features
        size, N = self.data.shape

        # Step 1 - Building training and validation sets

        # indices to split in training set and validation set
        ind = np.arange(size, dtype='int64')

        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target,test_size=(1-training_ratio), random_state=1998)

        # np.random.shuffle(ind)
        # training_ind, validation_ind = ind[:int(size * training_ratio)], ind[int(size * training_ratio):]

        self.training_data = X_train
        self.training_target = y_train
        self.validation_data = X_test
        self.validation_target = y_test

        # Step 2 - Train the model

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

        # We should do cross-validation here as well (possible to do it on the pre-processing line with pipeline maybe?)
        # No need to divide the data into training and validation is we can use GridSearchCV, it does it itself
        if self.modelType == 'MLP':
            self.model.fit(self.training_data, np.ravel(self.training_target))
        else:
            self.model.fit(self.training_data, self.training_target)
            print('Best score: ',self.model.best_score_)

    # Evaluation of the pipeline
    def evaluate(self, nb_iters=10, testing_ratio=0.3):

        size, N = self.validation_data.shape
        eval_size = int(size*testing_ratio)

        results = []
        ind = np.arange(size, dtype='int64')

        print("Evaluation will be done through {} iterations.".format(nb_iters))
        for iter in range(nb_iters):

            np.random.shuffle(ind)
            evaluation_data = self.validation_data[ind[:eval_size], :]
            evaluation_target = self.validation_target[ind[:eval_size]]

            # Evaluate the combination pre-processing + model
            predictions = self.model.predict(evaluation_data)

            # Compute scores
            results.append(self.scoringFunction(evaluation_target, predictions))

            if (iter+1) % 10 == 0:
                print("Ended iteration {} out of {}".format(iter+1, nb_iters))

        return np.mean(results)


if __name__ == "__main__":

    # Use pandas to load into a DataFrame
    # Y1.csv doesn’t have a header so
    # add one when loading the file

    X1 = pd.read_csv('X1.csv')
    Y1 = pd.read_csv('Y1.csv', header=None, names=['shares'])

    # To work with numpy arrays :
    X1 = X1.values
    Y1 = Y1.values

    # Visualization
    classRepartition(Y1)  # print number of articles per class

    # number of times we have to validate (10 epochs seem to be a minimal to obtain relevant performances results)
    nb_epochs = 10

    # which ratio of the data set we use for training (other data used for validation)
    training_ratio = 0.7

    # which methods we want to train (linear, KNN, MLP), be careful about the computation time
    # example : methods = ['linear', 'KNN', 'MLP', ...]
    methods = []
    #methods.append('linear')
    methods.append('KNN')
    #methods.append('MLP')

    # which pre-processing steps to apply for each method : one list per method to allow to specify more than one
    # pre-processing step for each method
    preProcessing = []
    #preProcessing.append(['standardization'])  # for linear regression
    preProcessing.append(['standardization','equalClassSize'])  # for KNN
    #preProcessing.append([])  # for MLP

    for i, method in enumerate(methods):

        print("========== TRAINING MODEL : {} ==========".format(method))

        # Build the trainer
        trainer = ModelTrainer(X1, Y1, method, preProcessing[i], scoreregression)

        # Visualize data
        # trainer.visualize()

        # Pre-process the data with the given methods
        print("Start of pre-processing ...", end="")
        # thresh is for minimum redundancy, n_components for maximum_relevance, ignored if MRMR is not used
        trainer.preProcess(thresh=1.0, n_components=20)
        print("End of pre-processing.")

        # Define the parameters to train
        if method == 'KNN':
            grid_params = {
                'n_neighbors': [24],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'cosine'],
                'leaf_size': [2] # 'euclidean', 'manhattan', 'chebyshev',
            }
            # {'metric': 'cosine', 'n_neighbors': 10, 'weights': 'uniform'}
            # 0.5008216155210505

        else:
            grid_params = None

        # Train the model
        print("Start of training ...", end="")
        trainer.train(training_ratio=training_ratio, parametersGrid=grid_params)
        print("End of training.")

        # Evaluate the model after training
        print("Start of evaluation ...", end="")
        result = trainer.evaluate(nb_iters=1, testing_ratio=1)
        print("End of evaluation.")

        # Print results
        print("Result for the {} method : {:.3f}".format(method, result))

        # Output class repartition
        trainer.outputClassRepartition()
