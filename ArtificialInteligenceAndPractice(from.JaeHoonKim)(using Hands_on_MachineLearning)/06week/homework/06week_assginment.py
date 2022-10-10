import sys
import sklearn
import numpy as np
import os

from mlxtend.data import loadlocal_mnist
from sklearn.svm import LinearSVC, SVC
from sklearn.svm import LinearSVR, SVR
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score, f1_score
###############
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
###############
import warnings
warnings.filterwarnings("ignore")

# Where to save the figures
PROJECT_ROOT_DIR = "C:\\Users\\mycom0703\\Desktop\\Young\\20.github_shared\\01.python\\AI\handsOnMachineLearning-from_JaeHoonKim-\\ArtificialInteligenceAndPractice(from.JaeHoonKim)(using Hands_on_MachineLearning)\\06week"
MNIST_PATH = PROJECT_ROOT_DIR + "\\datasets\\mnist\\"

def get_setosa_or_versicolor(iris):
    X = iris["data"]  # petal length, petal width
    y = iris["target"]

    setosa_or_versicolor = (y == 0) | (y == 1)
    print(setosa_or_versicolor)
    return X[setosa_or_versicolor], y[setosa_or_versicolor]

def get_virginica(iris):
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris virginica
    return X, y


def performance(y_test, y_test_pred, average='binary'):
    precision = precision_score(y_test, y_test_pred, average=average)
    recall = recall_score(y_test, y_test_pred, average=average)
    f1 = f1_score(y_test, y_test_pred, average=average)
    return precision, recall, f1  

def load_mnist(path=MNIST_PATH):
    X_train, y_train = loadlocal_mnist(
            images_path= MNIST_PATH+'train-images.idx3-ubyte', 
            labels_path= MNIST_PATH+'train-labels.idx1-ubyte')
    X_test, y_test = loadlocal_mnist(
            images_path= MNIST_PATH+'t10k-images.idx3-ubyte', 
            labels_path= MNIST_PATH+'t10k-labels.idx1-ubyte')
    return X_train, X_test, y_train, y_test

def transform(dataset):
    pipeline = Pipeline([
        ('std_scaler',    StandardScaler())
    ])

    return pipeline.fit_transform(dataset)

if __name__ == '__main__':
    # Large margin classification
    iris = datasets.load_iris()
    
    # make training data for binary classifiers
    X, y = get_setosa_or_versicolor(iris)
#     print(len(X))
#     print(X)
#     print(y)
    svc = SVC()
    svc.fit(X, y)
    y_pred = svc.predict(X)
    precision, recall, f1 = performance(y, y_pred, average="micro")
    print(precision, recall, f1)
    
    X_train, X_test, y_train, y_test = load_mnist(MNIST_PATH)
    print(X_train.shape)
    print(X_test.shape)
    X_train, y_train = X_train[:5000], y_train[:5000]
    svc.get_params()
    

    print()
################################

##         LinearSVC          ##

################################
    param_grid={
        'model__C': [0.5, 1.0, 5, 10, 15, 20],
        'model__penalty':['l1', 'l2'],
    }
    

    pipe1 =[
        ('scaler', MinMaxScaler()),
        ('model', LinearSVC())
    ]
    pipe2 = [
        ('scaler', StandardScaler()),
        ('model', LinearSVC())
    ] 

    pipeline1 = Pipeline(pipe1)
    pipeline2 = Pipeline(pipe2)

    gs1_1 = GridSearchCV(estimator=pipeline1,
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    cv=5,
                    n_jobs=-1
                   )
    
    gs1_1.fit(X_train, y_train)
    y_pred = gs1_1.predict(X_train)
    print(f"accuracy:{accuracy_score(y_train, y_pred)}")
    scores_df = pd.DataFrame(gs1_1.cv_results_)
    # print(scores_df)
    print("###############################################################################")
    print("1-1. estimator: linearSVC, minMaxScaler      param: C, penalty")
    print("score: ", gs1_1.score(X_test,y_test))
    print("best_estimator: ", gs1_1.best_estimator_)
    print("best_score: ", gs1_1.best_score_)
    print("best_param: ", gs1_1.best_params_)
    print("###############################################################################")
##################################################
    gs1_2 = GridSearchCV(estimator=pipeline2,
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    cv=5,
                    n_jobs=-1
                   )
    
    gs1_2.fit(X_train, y_train)
    y_pred = gs1_2.predict(X_train)
    print(f"accuracy:{accuracy_score(y_train, y_pred)}")
    scores_df = pd.DataFrame(gs1_2.cv_results_)
    # print(scores_df)
    print("###############################################################################")
    print("1-2.  estimator: linearSVC, standartScaler()      param: C, penalty")
    print("score: ", gs1_2.score(X_test,y_test))
    print("best_estimator: ", gs1_2.best_estimator_)
    print("best_score: ", gs1_2.best_score_)
    print("best_param: ", gs1_2.best_params_)
    print("###############################################################################")
    print()
# ################################

# ##        SGDClassifier       ##

# ################################

    param_grid={
        'model__loss': ['hindge', 'log', 'perceptron'],
        'model__penalty':['l1', 'l2'],
    }
    pipe1 =[
        ('scaler', MinMaxScaler()),
        ('model', SGDClassifier())
    ]
    pipe2 = [
        ('scaler', StandardScaler()),
        ('model', SGDClassifier())
    ]
    pipeline1 = Pipeline(pipe1, verbose=True)
    pipeline2 = Pipeline(pipe2, verbose=True)
#     pipeline1 = make_pipeline(StandardScaler(), LinearSVC())
    gs2_1 = GridSearchCV(estimator=pipeline1,
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    cv=5,
                    n_jobs=-1
                   )
    gs2_2 = GridSearchCV(estimator=pipeline2,
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    cv=5,
                    n_jobs=-1
                   )
    gs2_1.fit(X_train, y_train)
    
    y_pred_1 = gs2_1.predict(X_train)
    print(f"accuracy:{accuracy_score(y_train, y_pred_1)}")
    print("###############################################################################")
    print("2-1. estimator: SGDclassfier, MinMaxScaler()      param: loss, penalty")
    print("score: ", gs2_1.score(X_test,y_test))
    print("best_estimator: ", gs1_1.best_estimator_)
    print("best_score: ", gs2_1.best_score_)
    print("best_param: ", gs2_1.best_params_)
    print("###############################################################################")
    print()

    gs2_2.fit(X_train, y_train)
    y_pred_2 = gs2_2.predict(X_train)
    print(f"accuracy:{accuracy_score(y_train, y_pred_2)}")
    print("###############################################################################")
    print("2-2. estimator: SGDclassfier, standartScaler()      param: C, penalty")
    print("score: ", gs2_2.score(X_test,y_test))
    print("best_estimator: ", gs2_2.best_estimator_)
    print("best_score: ", gs2_2.best_score_)
    print("best_param: ", gs2_2.best_params_)
    print("###############################################################################")
    print()
# ################################

# ##             SVC            ##

# ################################
# # C: 0.5, 1.0, 5, 10, 15, 20
# # kernel: linear, rbf, poly, sigmod
# # degree: 2, 3
# # gamma: scale, auto  
    param_grid={
        'model__C':[0.5, 1.0, 5, 10, 15, 20],
        'model__kernel':['linear', 'rbf', 'poly', 'sigmod'],
        'model__degree': [2, 3],
        'model__gamma':['scale', 'auto'],
    }
    base_estimator = SGDClassifier()
    pipe1 = [
        ('scaler', MinMaxScaler()),
        ('model', SVC())
    ]    
    pipe2 =[
        ('scaler', StandardScaler()),
        ('model', SVC())
    ]
    pipeline1 = Pipeline(pipe1, verbose=True)
    pipeline2 = Pipeline(pipe2, verbose=True)
#     pipeline1 = make_pipeline(StandardScaler(), LinearSVC())
    gs3_1 = GridSearchCV(estimator=pipeline1,
                    param_grid=param_grid, 
                    scoring='accuracy', 
                    cv=5,
                    n_jobs=-1
                   )
    gs3_2 = GridSearchCV(estimator=pipeline2,
                param_grid=param_grid, 
                scoring='accuracy', 
                cv=5,
                n_jobs=-1
                )

    gs3_1.fit(X_train, y_train)
    y_pred = gs3_1.predict(X_train)
    print(f"accuracy:{accuracy_score(y_train, y_pred)}")
    scores_df = pd.DataFrame(gs3_1.cv_results_)
    print("###############################################################################")
    print("3-1. estimator: SVC, MinMaxScaler()      param: C, kernel, degree, gamma")
    print("score: ", gs3_1.score(X_test,y_test))
    print("best_estimator: ", gs3_1.best_estimator_)
    print("best_score: ", gs3_1.best_score_)
    print("best_param: ", gs3_1.best_params_)
    print("###############################################################################")


    gs3_2.fit(X_train, y_train)
    y_pred = gs3_2.predict(X_train)
    print(f"accuracy:{accuracy_score(y_train, y_pred)}")
    scores_df = pd.DataFrame(gs3_2.cv_results_)
    print("###############################################################################")
    print("3-2. estimator: SVC, standartScaler()      param: C, kernel, degree, gamma")
    print("score: ", gs3_2.score(X_test,y_test))
    print("best_estimator: ", gs3_2.best_estimator_)
    print("best_score: ", gs3_2.best_score_)
    print("best_param: ", gs3_2.best_params_)
    print("###############################################################################")