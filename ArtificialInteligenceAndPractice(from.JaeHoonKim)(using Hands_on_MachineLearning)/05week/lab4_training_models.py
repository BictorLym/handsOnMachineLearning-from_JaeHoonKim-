import sys
import sklearn
# Common imports
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression

from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")

def linear_closed_form(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


#main program
if __name__ == '__main__':
##    X = 2 * np.random.rand(100, 1) # normal distribution
##    #print(X)
##    y = 4 + 3 * X + np.random.randn(100, 1)
##    b = np.ones((100, 1))
##    theta = linear_closed_form(np.c_[b, X], y)
##    print('closed form =', theta)
##        
    X1 = np.array([[0], [2]])
##    X_new = np.c_[np.ones((2, 1)), X1]
##    print(X_new)
##    
##    y_predict = X_new.dot(theta)
##    print('closed predict = ', y_predict)
##
##    lin_reg = LinearRegression()
##    lin_reg.fit(X, y)
##    print('linear reg = ', lin_reg.intercept_, lin_reg.coef_)
##    y_predict = lin_reg.predict(X1)
##    print('predict = ', y_predict)
##
##    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
##
##    print(y)
##    print(y.ravel())
##    sgd_reg.fit(X, y.ravel())
##    print('SDG = ', sgd_reg.intercept_, sgd_reg.coef_)
##
##
##    
##    m = 100
##    X = 6 * np.random.rand(m, 1) - 3
##    print(X)
##    
##    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
##    poly_features = PolynomialFeatures(degree=3, include_bias=False)
##    X_poly = poly_features.fit_transform(X)
##    print(X_poly)
##
##    lin_reg = LinearRegression()
##    lin_reg.fit(X_poly, y)
##    print('polynomial =', lin_reg.intercept_, lin_reg.coef_)
##    X1_poly = poly_features.fit_transform(X1)
##    y_predict = lin_reg.predict(X1_poly)
##    print('predict = ', y_predict)
##
##    # Regularized models
##
##    np.random.seed(42)
##    m = 20
##    X = 3 * np.random.rand(m, 1)
##    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
##    X_new = np.linspace(0, 3, 100).reshape(100, 1)
##
##
##    ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
##    ridge_reg.fit(X, y)
##    y_predict = ridge_reg.predict([[1.5]])
##    print('Ridge= ', y_predict)
##
##    lasso_reg = Lasso(alpha=0.1)
##    lasso_reg.fit(X, y)
##    y_predict = lasso_reg.predict([[1.5]])
##    print('Lasso = ', y_predict)
##
##    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
##    elastic_net.fit(X, y)
##    y_predict = elastic_net.predict([[1.5]])
##    print('ElasticNet = ', y_predict)

    # Logistic regression
    iris = datasets.load_iris()
##    print(list(iris.keys()))
##    print(iris.DESCR)
##    print(iris.data)
##    print(iris.target)
##
##    # feature : sepal length / sepal width / petal length / petal width in cm
##    # class : Iris-Setosa / Iris-Versicolour / Iris-Virginica
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0
##    print(X)
##    print(y)
    log_reg = LogisticRegression(solver="lbfgs", random_state=42)
    log_reg.fit(X, y)
    
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    print('logit = ', y_proba)

    print(y_proba[:, 1] >= 0.5)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]
    print("decition boundary = ", decision_boundary)

    y_predict = log_reg.predict([[1.7], [1.5]])
    print('logit = ', y_predict)


    X = iris["data"]  # petal length, petal width
    y = iris["target"]

    softmax_reg = LogisticRegression(multi_class="multinomial",
                                     solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(X, y)
    X_new = [[5, 2, 1, 4], [2, 3, 4, 1]]
    y_proba = softmax_reg.predict_proba(X_new)
    print('prob = ', y_proba)
    y_predict = softmax_reg.predict(X_new)
    print('predict = ', y_predict)
    

    
    
          
    
    
    

    



    


    



    
 

    














