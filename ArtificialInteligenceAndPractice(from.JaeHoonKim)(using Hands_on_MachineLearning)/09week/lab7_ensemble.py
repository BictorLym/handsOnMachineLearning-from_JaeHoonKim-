import sys
import sklearn
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

def performance(y_test, y_test_pred, average='binary'):
    precision = precision_score(y_test, y_test_pred, average=average)
    recall = recall_score(y_test, y_test_pred, average=average)
    f1 = f1_score(y_test, y_test_pred, average=average)
    return precision, recall, f1

def transform(dataset):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),  # value이 없을 경우 median으로 변환
        ('std_scaler',    StandardScaler()),            # z = (x - u) / s 변환           
    ])

    return num_pipeline.fit_transform(dataset)

#main program
if __name__ == '__main__':
    target_col = "SeriousDlqin2yrs"
    ##"C:\\Users\\mycom0703\\Desktop\\Young\\20.github_shared\\01.python\\AI\handsOnMachineLearning-from_JaeHoonKim-\\ArtificialInteligenceAndPractice(from.JaeHoonKim)(using Hands_on_MachineLearning)\\07week"
    # data = pd.read_csv("cs-training.csv")
    file_path = "C:\\Users\\mycom0703\\Desktop\\Young\\20.github_shared\\01.python\\AI\handsOnMachineLearning-from_JaeHoonKim-\\ArtificialInteligenceAndPractice(from.JaeHoonKim)(using Hands_on_MachineLearning)\\09week\\cs-training.csv"
    data = pd.read_csv(file_path)
    print(data)
    
    data.drop("Unnamed: 0", axis=1, inplace=True)
    X, y = data.iloc[:,1:], data.iloc[:,0]
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)
    X_train = transform(X_train)
    X_test = transform(X_test)
    print("data shape :", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

##    #Voting classifiers
    log_clf = LogisticRegression(solver="lbfgs", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(gamma="scale", random_state=42)
##
    classifiers = (log_clf, rnd_clf, svm_clf)
##
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), 
                    ('rf', rnd_clf), 
                    ('svc', svm_clf)],
        voting='hard')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    print(voting_clf.__class__.__name__, accuracy_score(y_test, y_pred))
##    
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
##
    #Bagging ensembles
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, random_state=42)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    print('bagging :', accuracy_score(y_test, y_pred))
##
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    print('Decision Tree:', accuracy_score(y_test, y_pred))
##    
    # Random Forests
    # rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
    rnd_clf.fit(X_train, y_train)
    y_pred = rnd_clf.predict(X_test)
    print('Random Forests:', accuracy_score(y_test, y_pred))
##
    #Out-of-Bag evaluation
    bag_clf = BaggingClassifier(
        log_clf, n_estimators=500,
        bootstrap=True, oob_score=True, random_state=40)
    bag_clf.fit(X_train, y_train)
    print("Out-of-Bag evaluation:", bag_clf.oob_score_)

    y_pred = bag_clf.predict(X_test)
    print('OOB:', accuracy_score(y_test, y_pred))


    # AdaBoost
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=5), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    print('AdaBoost:', accuracy_score(y_test, y_pred))

    # XGBoost
    xgb_clf = XGBClassifier(random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)
    print('XGBoost:', accuracy_score(y_test, y_pred))
    
    
    


    

    

    
    
          
    
    
    

    



    


    



    
 

    














