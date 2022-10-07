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

import warnings
warnings.filterwarnings("ignore")

# Where to save the figures
PROJECT_ROOT_DIR = "."
MNIST_PATH = PROJECT_ROOT_DIR + "/datasets/mnist/"

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

#main program
if __name__ == '__main__':
    # Large margin classification
    iris = datasets.load_iris()
    
    # make training data for binary classifiers
    X, y = get_setosa_or_versicolor(iris)
##    print(len(X))
##    print(X)
##    print(y)
    
    
##    svm_clf = SVC(kernel="linear", C=float("inf"))
##    svm_clf.fit(X, y)
##    y_pred = svm_clf.predict(X)
##    precision, recall, f1 = performance(y, y_pred, average="micro")
##    print(precision, recall, f1)
##
##
    #Large margin vs margin violations
    #X, y = get_virginica(iris)
    X_train, X_test, y_train, y_test = load_mnist(MNIST_PATH)
    print(X_train.shape)
    print(X_test.shape)
    
    # small size
    X_train, y_train = X_train[:10000], y_train[:10000]

##    svm_clf1 = Pipeline([
##        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
##    ])
##    print("training .....")
##    svm_clf1.fit(X_train, y_train)
##    y_pred = svm_clf1.predict(X_test)
##    precision, recall, f1 = performance(y_test, y_pred, average="micro")
##    print(precision, recall, f1)
##
##    svm_clf2 = Pipeline([
##        ("scaler", MinMaxScaler()),
##        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
##    ])
##    print("training .....")
##    svm_clf2.fit(X_train, y_train)
##    y_pred = svm_clf2.predict(X_test)
##    precision, recall, f1 = performance(y_test, y_pred, average="micro")
##    print(precision, recall, f1)

##    # different regularization settings:
##    svm_clf3 = Pipeline([
##        ("scaler", MinMaxScaler()),
##        ("linear_svc", LinearSVC(C=0.1, loss="hinge", random_state=42)),
##    ])
##    print("training .....")
##    svm_clf3.fit(X_train, y_train)
##    y_pred = svm_clf3.predict(X_test)
##    precision, recall, f1 = performance(y_test, y_pred, average="micro")
##    print(precision, recall, f1)

    # Non-linear classification
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
##    print(X)
##    print(y)
##    polynomial_svm_clf = Pipeline([
##        ("poly_features", PolynomialFeatures(degree=3)),
##        ("scaler", StandardScaler()),
##        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
##    ])
##    print("training .....")
##    polynomial_svm_clf.fit(X, y)
##    y_pred = polynomial_svm_clf.predict(X)
##    precision, recall, f1 = performance(y, y_pred, average="micro")
##    print(precision, recall, f1)

    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(X, y)
    y_pred = poly_kernel_svm_clf.predict(X)
    precision, recall, f1 = performance(y, y_pred, average="micro")
    print(precision, recall, f1)

    poly100_kernel_svm_clf = Pipeline([
        ("scaler",  StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
    poly100_kernel_svm_clf.fit(X, y)

    
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
    rbf_kernel_svm_clf.fit(X, y)

    # Regression
    np.random.seed(42)
    m = 50
    X = 2 * np.random.rand(m, 1)
    y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

    svm_reg = LinearSVR(epsilon=1.5, random_state=42)
    svm_reg.fit(X, y)

    svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
    svm_poly_reg1.fit(X, y)

    svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1, gamma="scale")
    svm_poly_reg2.fit(X, y)


    

    

    
    
          
    
    
    

    



    


    



    
 

    














