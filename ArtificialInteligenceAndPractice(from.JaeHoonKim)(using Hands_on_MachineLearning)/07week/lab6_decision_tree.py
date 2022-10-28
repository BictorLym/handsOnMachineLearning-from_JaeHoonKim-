import sys
import sklearn
import numpy as np
import os
import pydot

import warnings
warnings.filterwarnings("ignore")


from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_moons

from graphviz import Source, render
from sklearn.tree import export_graphviz

# Where to save the figures
def performance(y_test, y_test_pred, average='binary'):
    precision = precision_score(y_test, y_test_pred, average=average)
    recall = recall_score(y_test, y_test_pred, average=average)
    f1 = f1_score(y_test, y_test_pred, average=average)
    return precision, recall, f1


# IMAGES_PATH = './datasets/'
IMAGES_PATH = "C:\\Users\\mycom0703\\Desktop\\Young\\20.github_shared\\01.python\\AI\handsOnMachineLearning-from_JaeHoonKim-\\ArtificialInteligenceAndPractice(from.JaeHoonKim)(using Hands_on_MachineLearning)\\07week"
#main program
if __name__ == '__main__':
    # first data
    #https://archive.ics.uci.edu/ml/datasets/iris
    iris = load_iris()
    X = iris.data # petal length and width
    y = iris.target
    print('iris.feature_names: ', iris.feature_names)
    print('iris.target_names: ', iris.target_names)
##
    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf.fit(X, y)

    y_test = [[1.0, 2.0, 5, 1.5]]

    y_prob = tree_clf.predict_proba(y_test)
    # print(f'prediction probability = {y_prob}')
    y_pred = tree_clf.predict(y_test)
    # print(f'prediction classed     = {y_pred}={iris.target_names[y_pred]}')

    # export_graphviz(
    #     tree_clf,
    #     out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
    #     feature_names=iris.feature_names,
    #     class_names=iris.target_names,
    #     rounded=True,
    #     filled=True
    # )
    # source = os.path.join(IMAGES_PATH, "iris_tree.dot")
    # destination = os.path.join(IMAGES_PATH, "iris_tree.png")
                            
    # (graph,) = pydot.graph_from_dot_file(source)
    # graph.write_png(destination)


##    # 2nd data
##    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
##    Xm, ym = make_moons(n_samples=1000, random_state=53)
##    print(Xm)
##    print(ym)
##    tree_clf = DecisionTreeClassifier(random_state=42)
##    tree_clf.fit(Xm, ym)
##    y_pred = tree_clf.predict(Xm)
##
##    precision, recall, f1 = performance(ym, y_pred, average="micro")
##    print(f'default DT          = {precision}, {recall}, {f1}')
##
##    tree_clf = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
##    tree_clf.fit(Xm, ym)
##    y_pred = tree_clf.predict(Xm)
##
##    precision, recall, f1 = performance(ym, y_pred, average="micro")
##    print(f'default prunning DT = {precision}, {recall}, {f1}')
##
##    angle = np.pi / 180 * 20
##    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
##    Xr = Xm.dot(rotation_matrix)
##
##    tree_clf = DecisionTreeClassifier(random_state=42)
##    tree_clf.fit(Xr, ym)
##    y_pred = tree_clf.predict(Xr)
##
##    precision, recall, f1 = performance(ym, y_pred, average="micro")
##    print(f'default nosized DT  = {precision}, {recall}, {f1}')


    # # Regression trees
    # # Quadratic training set + noise
    # np.random.seed(42)
    # m = 200
    # X = np.random.rand(m, 1)
    # y = 4 * (X - 0.5) ** 2
    # y = y + np.random.randn(m, 1) / 10
    # print(X)
    # print(y)

    # tree_reg = DecisionTreeRegressor(random_state=42)
    # tree_reg.fit(X, y)

    # export_graphviz(
    #     tree_reg,
    #     out_file=os.path.join(IMAGES_PATH, "regression_tree.dot"),
    #     feature_names=["x1"],
    #     rounded=True,
    #     filled=True
    # )
    # source = os.path.join(IMAGES_PATH, "regression_tree.dot")
    # destination = os.path.join(IMAGES_PATH, "regression_tree.png")
                          
    # (graph,) = pydot.graph_from_dot_file(source)
    # graph.write_png(destination)


    
    


    

    

    
    
          
    
    
    

    



    


    



    
 

    














