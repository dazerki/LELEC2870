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

# Use pandas t o l o a d i n t o a DataFrame
# Y1 . c s v doesn ’ t have a h e a d e r s o
#add one when l o a d i n g t h e f i l e

X1 = pd.read_csv('X1.csv')
Y1 = pd.read_csv ('Y1.csv', header=None , names=['shares'] )
# I f you p r e f e r t o work with numpy a r r a y s :
X1 = X1.values
Y1 = Y1.values

def linear_regression(X1, Y1):
    #select features
    data = np.concatenate((np.transpose(X1[0:9999, :]),np.transpose(Y1[0:9999])))
    corr = np.corrcoef(data)
    ind = np.argpartition(corr[:][58], -4)[-4:]
    features = np.zeros((19822,3))
    tracker = 0
    for i in ind[0:-1]:
        features[:,tracker] = X1[:, i]
        tracker +=1

    # Create linear regression object
    regr = linear_model.LinearRegression()

    print(features.shape)

    # Train the model using the training sets
    regr.fit(features[0:9999,:], Y1[0:9999])

    # Make predictions using the testing set
    y_pred = regr.predict(features[:][10000:19000])
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # print('Mean squared error: %.2f'
    #   % mean_squared_error(Y1[10000:19000], y_pred))
    print(scoreregression(Y1[10000:19000], y_pred))

def KNN(X1, Y1):
    X_train = X1[0:9999,:]
    y_train = Y1[0:9999]

    grid_params = {
        'n_neighbors': [10],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'cosine'],#'euclidean', 'manhattan', 'chebyshev',
    }
    # {'metric': 'cosine', 'n_neighbors': 10, 'weights': 'uniform'}
    # 0.5008216155210505



    scaler = preprocessing.StandardScaler().fit(X_train)

    # pca = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=0))
    # pca.fit(X_train, y_train)
    score_func = sklearn.metrics.make_scorer(scoreregression)

    gs = GridSearchCV(
        KNeighborsRegressor(),
        grid_params,
        verbose = 1,
        cv = 3,
        n_jobs = -1,
        scoring = score_func
        )
    gs_result = gs.fit(scaler.transform(X_train), y_train)
    print(gs_result.best_params_)
    y_pred = gs.predict(scaler.transform(X1[10000:19000]))
    # knn = KNeighborsRegressor(n_neighbors=5)
    # knn.fit(pca.transform(scaler.transform(X_train)), np.ravel(y_train))
    #
    # y_pred = knn.predict(pca.transform(scaler.transform(X1[10000:19000,:])))
    print(scoreregression(Y1[10000:19000], y_pred))

def MLP(X1, Y1):
    X_train = X1[0:9999,:]
    y_train = Y1[0:9999]

    score_func = sklearn.metrics.make_scorer(scoreregression)

    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, np.ravel(y_train))

    y_pred = regr.predict(X1[10000:19000])
    print(scoreregression(Y1[10000:19000], y_pred))


def scoref1(ytrue, ypred, th):
    return sklearn.metrics.f1_score(ytrue>th, ypred>th)
def scoreregression(ytrue, ypred):
    scores = [
    scoref1(ytrue, ypred, th=th) for th in [ 500, 1400, 5000, 10000]
    ]
    return np.mean(scores)


KNN(X1, Y1)
