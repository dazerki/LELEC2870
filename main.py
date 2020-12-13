import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    pass
import pandas as pd
import sklearn

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


def scoref1(ytrue, ypred, th):
    return sklearn.metrics.f1_score(ytrue > th, ypred > th)

def scoreregression(ytrue, ypred):
    scores = [
    scoref1(ytrue, ypred, th=th) for th in [500, 1400, 5000, 10000]
    ]
    return np.mean(scores)

def preProcessCorr(X, Y):

    # corr = np.corrcoef(training_set)
    # ind = np.argpartition(corr[:][58], -4)[-4:]
    # features = np.zeros((19822, 3))
    # tracker = 0
    # for i in ind[0:-1]:
    #     features[:, tracker] = X1[:, i]
    #     tracker += 1

    corr = np.corrcoef(X, Y, rowvar=False)[:-1, -1]  # last column/line is correlation between each feature and output
    mean_corr = np.mean(np.abs(corr))
    corr_ind = np.where(np.abs(corr) > mean_corr)[0]  # numpy.where returns array of the result => take the result
    return X[:, corr_ind]


class ModelTrainer:

    def __init__(self, data, target, modelType, preProcessingList, scoringFunction):
        self.data = data
        self.target = target
        self. preProcessingList = preProcessingList
        self.scoringFunction = scoringFunction
        self.modelType = modelType

        if modelType == 'linear':
            self.model = linear_model.LinearRegression()
        elif modelType == 'KNN':
            self.model = KNeighborsRegressor()
        elif modelType == 'MLP':
            self.model = MLPRegressor(random_state=1, max_iter=500)

    # Pre-processing of the data
    def preProcess(self):

        for preProcessMethod in self.preProcessingList:

            if preProcessMethod == 'correlation':
                self.data = preProcessCorr(self.data, self.target)

            elif preProcessMethod == 'whitening':
                continue

            elif preProcessMethod == 'normalization':
                scaler = preprocessing.StandardScaler().fit(self.data)
                self.data = scaler.transform(self.data)

            elif preProcessMethod == 'PCA':
                pca = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=0))
                pca.fit(self.data, self.target)
                self.data = pca.transform(self.data)
                continue

             # other pre-processing steps ?
            else:
                continue

    # Training of the model
    def train(self, training_ratio=0.7, parametersGrid=None):

        # useful values : size = nb of data points, N = nb of features
        size, N = self.data.shape

        # Step 1 - Building training and validation sets

        # indices to split in training set and validation set
        ind = np.arange(X1.shape[0], dtype='int64')

        np.random.shuffle(ind)
        training_ind, validation_ind = ind[:int(size * training_ratio)], ind[int(size * training_ratio):]

        self.training_data = self.data[training_ind, :]
        self.training_target = self.target[training_ind]
        self.validation_data = self.data[validation_ind, :]
        self.validation_target = self.target[validation_ind]

        # Step 2 - Train the model

        if parametersGrid is not None:
            score_func = sklearn.metrics.make_scorer(self.scoringFunction)
            self.model = GridSearchCV(
                self.model,
                parametersGrid,
                verbose=1,
                cv=3,
                n_jobs=-1,
                scoring=score_func
            )

        if self.modelType == 'MLP':
            self.model.fit(self.training_data, np.ravel(self.training_target))
        else:
            self.model.fit(self.training_data, self.training_target)

    # Evaluation of the trained model
    def evaluate(self, nb_epochs):
        train_results = []
        valid_results = []

        print("Evaluation will be done through {} epochs.".format(nb_epochs))
        for epoch in range(nb_epochs):

            # Evaluate the combination pre-processing + model
            train_predictions = self.model.predict(self.training_data)
            valid_predictions = self.model.predict(self.validation_data)

            # Compute scores
            train_results.append(self.scoringFunction(self.training_target, train_predictions))
            valid_results.append(self.scoringFunction(self.validation_target, valid_predictions))
            print("Ended iteration {} out of {}".format(epoch+1, nb_epochs))

        return np.mean(train_results), np.mean(valid_results)


if __name__ == "__main__":

    # Use pandas to load into a DataFrame
    # Y1.csv doesn’t have a header so
    # add one when loading the file

    X1 = pd.read_csv('X1.csv')
    Y1 = pd.read_csv('Y1.csv', header=None, names=['shares'])

    # To work with numpy arrays :
    X1 = X1.values
    Y1 = Y1.values

    # number of times we have to validate (10 epochs seem to be a minimal to obtain relevant performances results)
    nb_epochs = 10

    # which ratio of the data set we use for training (other data used for validation)
    training_ratio = 0.7

    # which methods we want to train (linear, KNN, MLP), be careful about the computation time
    # example : methods = ['linear', 'KNN', 'MLP', ...]
    methods = []
    #methods.append('linear')
    #methods.append('KNN')
    methods.append('MLP')

    # which pre-processing steps to apply for each method : one list per method to allow to specify more than one
    # pre-processing step for each method
    preProcessing = []
    #preProcessing.append(['correlation'])  # for linear regression
    #preProcessing.append(['normalization'])  # for KNN
    preProcessing.append([])

    for i, method in enumerate(methods):

        print("========== TRAINING MODEL : {} ==========".format(method))

        # Build the trainer
        trainer = ModelTrainer(X1, Y1, method, preProcessing[i], scoreregression)

        # Pre-process the data with the given methods
        print("Start of pre-processing ...", end="")
        trainer.preProcess()
        print("End of pre-processing.")

        # Define the parameters to train
        if method == 'KNN':
            grid_params = {
                'n_neighbors': [10],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'cosine'],  # 'euclidean', 'manhattan', 'chebyshev',
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
        training_result, validation_result = trainer.evaluate(nb_epochs=nb_epochs)
        print("End of evaluation.")

        # Print results
        print("Average result for the {} method on the training set : {:.2f}".format(method, training_result))
        print("Average result for the {} method on the validation set : {:.2f}".format(method, validation_result))
