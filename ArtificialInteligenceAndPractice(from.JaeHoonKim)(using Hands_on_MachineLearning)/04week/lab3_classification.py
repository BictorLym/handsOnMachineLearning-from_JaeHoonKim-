import os
import numpy as np

from sklearn.datasets import fetch_openml
from mlxtend.data     import loadlocal_mnist

from sklearn.linear_model import SGDClassifier
from sklearn.svm          import SVC
from sklearn.neighbors    import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import precision_score, recall_score, f1_score
from sklearn.metrics         import precision_recall_fscore_support
from sklearn.preprocessing   import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# Where to save the figures
PROJECT_ROOT_DIR = "."
# PROJECT_ROOT_DIR = "C:\\Users\\mycom0703\\Desktop\\Young\\20.github_shared\\01.python\\AI\handsOnMachineLearning-from_JaeHoonKim-\\ArtificialInteligenceAndPractice(from.JaeHoonKim)(using Hands_on_MachineLearning\\04week)"
MNIST_PATH = PROJECT_ROOT_DIR + "/datasets/mnist/"

## http://yann.lecun.com/exdb/mnist/

def load_mnist(path=MNIST_PATH):
    X_train, y_train = loadlocal_mnist(
            images_path= MNIST_PATH+'train-images.idx3-ubyte', 
            labels_path= MNIST_PATH+'train-labels.idx1-ubyte')
    X_test, y_test = loadlocal_mnist(
            images_path= MNIST_PATH+'t10k-images.idx3-ubyte', 
            labels_path= MNIST_PATH+'t10k-labels.idx1-ubyte')
    return X_train, X_test, y_train, y_test


def performance(y_test, y_test_pred, average='binary'):
    precision = precision_score(y_test, y_test_pred, average=average)
    recall    = recall_score(y_test, y_test_pred, average=average)
    f1        = f1_score(y_test, y_test_pred, average=average)
    return precision, recall, f1
    

#main program
if __name__ == '__main__':
##    # get data
##    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
##    X, y = mnist["data"], mnist["target"]
##    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
##    print(X_train.shape, y_train)
##    print(X_test.shape, y_test)
    
    # OR download https://deepai.org/dataset/mnist
    # unzip it to datasets
    X_train, X_test, y_train, y_test = load_mnist(MNIST_PATH)
    print(X_train.shape, y_train)
    print(X_test.shape, y_test)

##    # bianry classification
##    # 5인지 아닌지를 분류한다.
##    y_train_5 = (y_train == 5)
##    #y_train_5 = [1 if y == 5 else 0 for y in y_train]
##    y_test_5  = (y_test == 5)
##    print(y_train_5)
##    print(y_test_5)
##
##    sgd_clf = SGDClassifier(max_iter=100, tol=1e-3, random_state=42)
##    sgd_clf.fit(X_train, y_train_5) # training
##    y_test_pred = sgd_clf.predict(X_test) # test
##
##    cm = confusion_matrix(y_test_5, y_test_pred)
##    print("교차검증 혼동행렬 :\n", cm)
##
##    precision, recall, f1 = performance(y_test_5, y_test_pred)
##    print(precision, recall, f1)
##    # 교재 92 페이지 참조
##    print("정밀도 :", cm[1, 1] / (cm[0, 1] + cm[1, 1]))
##    print("재현율 :", cm[1, 1] / (cm[1, 0] + cm[1, 1]))


##    # Multiclass classification
##    svm_clf = SVC(gamma="auto", max_iter=100, random_state=42)
##    svm_clf.fit(X_train, y_train)
##    y_test_pred = svm_clf.predict(X_test)
##    cm = confusion_matrix(y_test, y_test_pred)
##    print("교차검증 혼동행렬 :\n", cm)
##
##    precision, recall, f1 = performance(y_test, y_test_pred, average="micro")
##    print(precision, recall, f1)


##    scaler = StandardScaler()
##    X_train = scaler.fit_transform(X_train.astype(np.float64))
##    X_test  = scaler.fit_transform(X_test.astype(np.float64))
 

##    svm_clf.fit(X_train, y_train) # y_train, not y_train_5
##    y_test_pred = svm_clf.predict(X_test)
##    cm = confusion_matrix(y_test, y_test_pred)
##    print("교차검증 혼동행렬 :\n", cm)
##
##    precision, recall, f1 = performance(y_test, y_test_pred, average="micro")
##    print(precision, recall, f1)


##    # Multilabel classification
##    y_train_large = (y_train >= 7)
##    y_train_odd   = (y_train % 2 == 1)
##    y_multilabel = np.c_[y_train_large, y_train_odd]
####    print(y_multilabel)
##
##    knn_clf = KNeighborsClassifier()
##    knn_clf.fit(X_train, y_multilabel)
##
##    y_test_large = (y_test >= 7)
##    y_test_odd   = (y_test % 2 == 1)
##    y_test_multilabel = np.c_[y_test_large, y_test_odd]
##
##    y_test_pred = knn_clf.predict(X_test)
##    precision, recall, f1 = performance(y_test_multilabel, y_test_pred, average="micro")
##    print(precision, recall, f1)


    # Multioutput classification
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise

    y_train_mod = X_train
    y_test_mod = X_test

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict(X_test_mod)

    print(X_test_mod[0])
    print(clean_digit[0])

    
    
          
    
    
    

    



    


    



    
 

    














