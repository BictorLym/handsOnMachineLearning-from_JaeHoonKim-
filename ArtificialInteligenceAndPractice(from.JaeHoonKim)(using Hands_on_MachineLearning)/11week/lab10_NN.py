import sys
import sklearn
import numpy as np
import os
import gzip
from pprint import pprint

import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import load_iris
from mlxtend.data import loadlocal_mnist

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

def load_mnist(path):
    X_train, y_train = loadlocal_mnist(
            images_path= path+'train-images.idx3-ubyte', 
            labels_path= path+'train-labels.idx1-ubyte')
    X_test, y_test = loadlocal_mnist(
            images_path= path+'t10k-images.idx3-ubyte', 
            labels_path= path+'t10k-labels.idx1-ubyte')
    return X_train, X_test, y_train, y_test

def load_fashion_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,"{}-labels-idx1-ubyte.gz".format(kind))
    images_path = os.path.join(path,"{}-images-idx3-ubyte.gz".format(kind))

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

def load_next_batch(X, batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]

#Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output
   
#main program
if __name__ == '__main__':
    data = load_iris()
    X, y = data.data, data.target

    per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    per_clf.fit(X, y)
    y_pred = per_clf.predict(X)

    print(y)
    print(y_pred)
    print(accuracy_score(y, y_pred))
    print(per_clf.coef_, per_clf.intercept_)
##

##    # Building an Image Classifier
##    # download dataset(fashion_mnist)
##    #          from https://github.com/zalandoresearch/fashion-mnist
##    MNIST_PATH = './datasets/fashion_mnist/'
##    X_train, y_train = load_fashion_mnist(MNIST_PATH, 'train')
##    X_test,  y_test  = load_fashion_mnist(MNIST_PATH, 't10k')
##    print(X_train.shape)
##    X_train = X_train.reshape(-1, 28, 28)
##    X_test  = X_test.reshape(-1, 28, 28)
##    print(X_train.shape)
##    print(y_train)
##    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
##                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
##
##    X_train = X_train / 255
##    X_test  = X_test  / 255
##    X_train = X_train[:10000]
##    y_train = y_train[:10000]
##    X_valid, X_train = X_train[:1000], X_train[1000:]
##    y_valid, y_train = y_train[:1000], y_train[1000:]
##    
##    model = keras.models.Sequential()
##    model.add(keras.layers.Flatten(input_shape=[28, 28]))
##    model.add(keras.layers.Dense(300, activation="relu"))
##    model.add(keras.layers.Dense(100, activation="relu"))
##    model.add(keras.layers.Dense(10, activation="softmax"))
##
##    keras.backend.clear_session()
##    np.random.seed(42)
##    tf.random.set_seed(42)
####
####    model = keras.models.Sequential([
####        keras.layers.Flatten(input_shape=[28, 28]),
####        keras.layers.Dense(300, activation="relu"),
####        keras.layers.Dense(100, activation="relu"),
####        keras.layers.Dense(10, activation="softmax")
####    ])
##    print(model.summary())
##
##    model.compile(loss="sparse_categorical_crossentropy",
##              optimizer="sgd",
##              metrics=["accuracy"])
##
##    history = model.fit(X_train, y_train, epochs=20,
##                    validation_data=(X_valid, y_valid), verbose=2)
##    print(history)
##    _, acc = model.evaluate(X_test, y_test, verbose=2)
##    print(acc)
##
##
##    # Regression MLP
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test \
        = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid \
        = train_test_split(X_train_full, y_train_full, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test  = scaler.transform(X_test)

##    model = keras.models.Sequential([
##        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
##        keras.layers.Dense(1)
##    ])
##    model.compile(loss="mean_squared_error",
##                  optimizer=keras.optimizers.SGD(lr=1e-3))
##    history = model.fit(X_train, y_train,
##                        epochs=2,
##                        validation_data=(X_valid, y_valid), verbose=2)
##    mse_test = model.evaluate(X_test, y_test, verbose=2)
##    print(mse_test)
##    
##
    # Functional API
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.models.Model(inputs=[input_], outputs=[output])

    #keras.utils.plot_model(model, "housing.png", show_shapes=True)
    model.compile(loss="mse",
                  optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train,
                  epochs=2,
                  validation_data=(X_valid, y_valid),
                  verbose=2)
    mse_test = model.evaluate(X_test, y_test, verbose=2)
    print(mse_test)
##    
##    # The subclassing API
##    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
##    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
##    X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
##    X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
##    
##    model = WideAndDeepModel(30, activation="relu")
##    # keras.utils.plot_model(model, "housing.png", show_shapes=True)
##    model.compile(loss="mse",
##                  loss_weights=[0.9, 0.1],
##                  optimizer=keras.optimizers.SGD(lr=1e-3))
##    history = model.fit((X_train_A, X_train_B),
##                        (y_train, y_train),
##                        epochs=2,
##                        validation_data=((X_valid_A, X_valid_B),
##                                         (y_valid, y_valid)),
##                        verbose=2)
##    total_loss, main_loss, aux_loss = model.evaluate((X_test_A, X_test_B),
##                                                     (y_test, y_test),
##                                                     verbose=2)
##    y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
##    print(y_pred_main)
##    print(y_pred_aux)
##
##    
    # Saving and Restoring
    model.save_weights("my_keras_model.h5")

    model.load_weights("my_keras_model.h5")
    
    model.save("models")
    model = keras.models.load_model("models")

    y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
    print(y_pred_main)
##    print(y_pred_aux)
##    
    
          
    
    
    

    



    


    



    
 

    














