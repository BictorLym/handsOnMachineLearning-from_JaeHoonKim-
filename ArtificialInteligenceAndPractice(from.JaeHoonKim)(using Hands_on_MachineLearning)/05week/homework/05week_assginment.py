# pip install mlxtend

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import classification_report
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")
# print("test success")

# print(pd.show_versions())

def load_data(fileName ):
    # Where to save the figures
    # PROJECT_ROOT_DIR = "."
    PROJECT_ROOT_DIR = "C:\\Users\\mycom0703\\Desktop\\Young\\20.github_shared\\01.python\\AI\handsOnMachineLearning-from_JaeHoonKim-\\ArtificialInteligenceAndPractice(from.JaeHoonKim)(using Hands_on_MachineLearning)\\05week\\homework"
    DATA_PATH = PROJECT_ROOT_DIR + "\\datasets\\"
    data = pd.read_csv(DATA_PATH + fileName)
    return data

def delete_data(data, column):
    data.pop(column)
    
def get_age_category(x):
    #AgeCategory
    #18-24:0,  25-29:1,  30-34:2,  35-39:3,  40-44:4,  45-49:5,
    #50-54:6,  55-59:7,  60-64:8,  65-69:9,  70-74:10, 75-79:11,
    #80 or older: 12
    if(x=='18-24'):
        return 0;
    elif(x=='18-24'): return 0;
    elif(x=='25-29'): return 1;
    elif(x=='30-34'): return 2;
    elif(x=='35-39'): return 3;
    elif(x=='40-44'): return 4;
    elif(x=='45-49'): return 5;
    elif(x=='50-54'): return 6;
    elif(x=='55-59'): return 7;
    elif(x=='60-64'): return 8;
    elif(x=='65-69'): return 9;
    elif(x=='70-74'): return 10;
    elif(x=='75-79'): return 11;
    elif(x=='80 or older'): return 12;

def get_race_category(x):
    #American Indian/Alaskan Native:0, Asian:1, Black:2
    #Hispanic:3, Other:4, White:5
    if(x=='American Indian/Alaskan Native'): return 0;
    elif(x=='Asian'): return 1;
    elif(x=='Black'): return 2;
    elif(x=='Hispanic'): return 3;
    elif(x=='Other'): return 4;
    elif(x=='White'): return 5;

def get_gen_health_category(x):
    #Poor:0, Fair:1, Good:2
    #Very Good:3, Excellent:4
    if(x=='Poor'): return 0;
    elif(x=='Fair'): return 1;
    elif(x=='Good'): return 2;
    elif(x=='Very good'): return 3;
    elif(x=='Excellent'): return 4;
#     else: return 5;

def numberizingColumn1(data):
    #Yes == 1, No==0
    data['HeartDisease'] = data['HeartDisease'].apply(lambda x:0 if x=="No" else 1)
    data['Smoking'] = data['Smoking'].apply(lambda x:0 if x=='No' else 1)
    data['AlcoholDrinking'] = data['AlcoholDrinking'].apply(lambda x:0 if x=='No' else 1)
    data['Stroke'] = data['Stroke'].apply(lambda x:0 if x=='No' else 1)
    data['DiffWalking'] = data['DiffWalking'].apply(lambda x:0 if x=='No' else 1)
    data['Diabetic'] = data['Diabetic'].apply(lambda x:0 if x=='No' else 1)
    data['PhysicalActivity'] = data['PhysicalActivity'].apply(lambda x:0 if x=='No' else 1)
    data['Asthma'] = data['Asthma'].apply(lambda x:0 if x=='No' else 1)
    data['KidneyDisease'] = data['KidneyDisease'].apply(lambda x:0 if x=='No' else 1)
    data['SkinCancer'] = data['SkinCancer'].apply(lambda x:0 if x=='No' else 1)
    
    #female=0 male=1
    data['Sex'] = data['Sex'].apply(lambda x:0 if x=='Female' else 1)

def numberizingColumn2(data):
    #get_age_category
    #18-24:0,  25-29:1,  30-34:2,  35-39:3,  40-44:4,  45-49:5,
    #50-54:6,  55-59:7,  60-64:8,  65-69:9,  70-74:10, 75-79:11,
    #80 or older: 12
    data['AgeCategory'] = data['AgeCategory'].apply(get_age_category)
    
    #get_race_category
    #American Indian/Alaskan Native:0, Asian:1, Black:2
    #Hispanic:3, Other:4, White:5
    data['Race'] = data['Race'].apply(get_race_category)
    
    #get_gen_health_category
    #Poor:0, Fair:1, Good:2
    #Very Good:3, Excellent:4
    data['GenHealth'] = data['GenHealth'].apply(get_gen_health_category)

def EDA1(data_set):
    df = data_set
    numberizingColumn2(df)
    print('test')
    ##Sex
    female_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Sex']=='Female')])
    num_female = len(df[df['Sex']=='Female'])
    male_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Sex']=='Male')])
    num_male = len(df[df['Sex']=='Male'])
    print('Sex')
    print('남자가 심장병에 걸릴확률: ', male_with_heart_disease/num_male)
    print('여자가 심장병에 걸릴확률: ', female_with_heart_disease/num_female)
    print()

    ##Smoking
    smoking_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Smoking']=='Yes')])
    num_smoking = len(df[df['Smoking']=='Yes'])
    no_smoking_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Smoking']=='No')]) 
    num_no_smoking = len(df[df['Smoking']=='No'])
    print('Smoking')
    print('흡연자가 심장병에 걸릴확률: ', smoking_with_heart_disease/num_smoking)
    print('비흡연자가 심장병에 걸릴확률: ', no_smoking_with_heart_disease/num_no_smoking)
    print()
    
    ##AlcoholDrinking
    alchol_drinking_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['AlcoholDrinking']=='Yes')])
    num_alchol_drinking = len(df[df['AlcoholDrinking']=='Yes'])
    no_alchol_drinking_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['AlcoholDrinking']=='No')])
    num_no_alchol_drinking = len(df[df['AlcoholDrinking']=='No'])
    print('AlcoholDrinking')
    print('음주자가 심장병에 걸릴확률: ', alchol_drinking_with_heart_disease/num_alchol_drinking)
    print('비음주자가 심장병에 걸릴확률: ', no_alchol_drinking_with_heart_disease/num_no_alchol_drinking)
    print()
    
    ##Stroke
    stroke_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Stroke']=='Yes')])
    num_stroke = len(df[df['Stroke']=='Yes'])
    no_stroke_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Stroke']=='No')])
    num_no_stroke = len(df[df['Stroke']=='No'])
    print('Stroke')
    print('뇌졸중자가 심장병에 걸릴확률: ', stroke_with_heart_disease/num_stroke)
    print('비뇌졸중자가 심장병에 걸릴확률: ', no_stroke_with_heart_disease/num_no_stroke)
    print()
    ##DiffWalking
    diffwalking_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['DiffWalking']=='Yes')])
    num_diffwalking = len(df[df['DiffWalking']=='Yes'])
    no_diffwalking_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['DiffWalking']=='No')])
    num_no_diffwalking = len(df[df['DiffWalking']=='No'])
    print('DiffWalking')
    print('걷기 어려운자가 심장병에 걸릴확률: ', diffwalking_with_heart_disease/num_diffwalking)
    print('비걷기 어려운자가 심장병에 걸릴확률: ', no_diffwalking_with_heart_disease/num_no_diffwalking)
    print()
    ##Diabetic
    diabetic_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Diabetic']=='Yes')])
    num_diabetic = len(df[df['Diabetic']=='Yes'])
    no_diabetic_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Diabetic']=='No')])
    num_no_diabetic = len(df[df['Diabetic']=='No'])
    print('Diabetic')
    print('당뇨병자가 심장병에 걸릴확률: ', diabetic_with_heart_disease/num_diabetic)
    print('비당뇨병자가 심장병에 걸릴확률: ', no_diabetic_with_heart_disease/num_no_diabetic)
    print()
    ##PhysicalActivity
    physical_activity_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['PhysicalActivity']=='Yes')])
    num_physical_activity = len(df[df['PhysicalActivity']=='Yes'])
    no_physical_activity_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['PhysicalActivity']=='No')])
    num_no_physical_activity = len(df[df['PhysicalActivity']=='No'])
    print('PhysicalActivity')
    print('물리적활동이 가능한자가 심장병에 걸릴확률: ', physical_activity_with_heart_disease/num_physical_activity)
    print('물리적활동이 불가능한자에 걸릴확률: ', no_physical_activity_with_heart_disease/num_no_physical_activity)
    print()
    ##Asthma
    asthma_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Asthma']=='Yes')])
    num_asthma = len(df[df['Asthma']=='Yes'])
    no_asthma_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['Asthma']=='No')])
    num_no_asthma = len(df[df['Asthma']=='No'])
    print('PhysicalActivity')
    print('물리적활동이 가능한자가 심장병에 걸릴확률: ', asthma_with_heart_disease/num_asthma)
    print('물리적활동이 불가능한자가 심장병에 걸릴확률: ', no_asthma_with_heart_disease/num_no_asthma)
    print()
    ##KidneyDisease
    kidney_disease_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['KidneyDisease']=='Yes')])
    num_kidney_disease = len(df[df['KidneyDisease']=='Yes'])
    no_kidney_disease_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['KidneyDisease']=='No')])
    num_no_kidney_disease = len(df[df['KidneyDisease']=='No'])
    print('KidneyDisease')
    print('신장병자가 심장병에 걸릴확률: ', kidney_disease_with_heart_disease/num_kidney_disease)
    print('비신장병자가 심장병에 걸릴확률: ', no_kidney_disease_with_heart_disease/num_no_kidney_disease)
    print()
    ##SkinCancer
    skincancer_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['SkinCancer']=='Yes')])
    num_skincancer = len(df[df['SkinCancer']=='Yes'])
    no_skincancer_with_heart_disease = len(df[(df['HeartDisease']=='Yes') & (df['SkinCancer']=='No')])
    num_no_skincancer = len(df[df['SkinCancer']=='No'])
    print('SkinCancer')
    print('피부암환자가 심장병에 걸릴확률: ', skincancer_with_heart_disease/num_skincancer)
    print('비피부암환자가 심장병에 걸릴확률: ', no_skincancer_with_heart_disease/num_no_skincancer)
    print()
    
    ##########################################################################################
    
    ##BMI
    print('BMI')
    fix, ax = plt.subplots( figsize = (13,5))

    sns.kdeplot(df[df["HeartDisease"] == 'Yes']["BMI"], ax = ax)
    sns.kdeplot(df[df["HeartDisease"] == 'No']["BMI"], ax = ax)
    
    ax.set_xlabel("BMI")
    ax.set_ylabel("HeartDisease")    
    plt.legend(["HeartDisease", "NonHeartDisease"])
    plt.show()
    print()
    
    
    ##PhysicalHealth
    print('PhysicalHealth')
    fix, ax = plt.subplots(1, 1, figsize = (13,5))

    sns.kdeplot(df[df["HeartDisease"] == 'Yes']["PhysicalHealth"], ax = ax)
    sns.kdeplot(df[df["HeartDisease"] == 'No']["PhysicalHealth"], ax = ax)

    plt.legend(["HeartDisease", "NonHeartDisease"])
    plt.show()
    print()
    
    
    ##MentalHealth
    print('MentalHealth')
    fix, ax = plt.subplots(1, 1, figsize = (13,5))

    sns.kdeplot(df[df["HeartDisease"] == 'Yes']["MentalHealth"], ax = ax)
    sns.kdeplot(df[df["HeartDisease"] == 'No']["MentalHealth"], ax = ax)

    plt.legend(["HeartDisease", "NonHeartDisease"])
    plt.show()
    print()
    
    ##########################################################################################
    
    #AgeCategory
    print('AgeCategory')
    fix, ax = plt.subplots( figsize = (14,6))

    sns.kdeplot(df[df["HeartDisease"] == 'Yes']["AgeCategory"], ax = ax)
    sns.kdeplot(df[df["HeartDisease"] == 'No']["AgeCategory"], ax = ax)
    
    ax.set_xlabel("AgeCategory")
    ax.set_ylabel("HeartDisease")    
    plt.legend(["HeartDisease", "NonHeartDisease"])
    plt.show()
    print()


    #Race
    print('Race')
    fix, ax = plt.subplots( figsize = (14,6))

    sns.kdeplot(df[df["HeartDisease"] == 'Yes']["Race"], ax = ax)
    sns.kdeplot(df[df["HeartDisease"] == 'No']["Race"], ax = ax)
    
    ax.set_xlabel("Race")
    ax.set_ylabel("HeartDisease")    
    plt.legend(["HeartDisease", "NonHeartDisease"])
    plt.show()    
    print()
    
    #GenHealth
    print('GenHealth')
    fix, ax = plt.subplots( figsize = (14,6))

    sns.kdeplot(df[df["HeartDisease"] == 'Yes']["GenHealth"], ax = ax)
    sns.kdeplot(df[df["HeartDisease"] == 'No']["GenHealth"], ax = ax)
    
    ax.set_xlabel("GenHealth")
    ax.set_ylabel("HeartDisease")    
    plt.legend(["HeartDisease", "NonHeartDisease"])
    plt.show()    
    print()
    
    ##SleepTime
    print('SleepTime')
    fix, ax = plt.subplots(1, 1, figsize = (13,5))

    sns.kdeplot(df[df["HeartDisease"] == 'Yes']["SleepTime"], ax = ax)
    sns.kdeplot(df[df["HeartDisease"] == 'No']["SleepTime"], ax = ax)

    ax.set_xlabel("SleepTime")
    ax.set_ylabel("HeartDisease")    
    plt.legend(["HeartDisease", "NonHeartDisease"])
    plt.show()
    print()

if __name__ == '__main__':
    data = load_data('heart_2020_cleaned.csv')
    
    # print(data)
    ###########################################################
    ##2.숫자가 아닌 값들은 모두 숫자로 변환한다.
    ###########################################################
    numberizingColumn1(data)
    numberizingColumn2(data)
    # print(data)


    ###########################################################
    ##4.성능에 방해가 되는 자질이 있다면 제거한다. 제거했다면 주석을 통해서 왜 제거했는지를 설명해야 한다.
    ###########################################################
    # print(data.corr())
    #############################
    #HeartDisease와 상관관계 분석시 0.09미만의 데이터를 지우기로 결정하였다
    #BMI:0.051
    #AlcoholDrinking:-0.032
    #MentalHealth:0.028
    #Sex:0.070
    #Race:0.034
    #SleepTime:0.008
    #Asthma:0.041
    #############################
    columns = ['BMI', 'AlcoholDrinking', 'MentalHealth','Sex', 'Race', 'SleepTime','Asthma']
    for i in range(len(columns)):
        delete_data(data, column=columns[i])
        
    # print(data)


    ###########################################################
    ##3.적절히 정규화(scaling)를 수행한다.
    ###########################################################
    scaler = MinMaxScaler()
    data[:] = scaler.fit_transform(data[:])
    # print(data)
    


    ###########################################################
    ##1. 학습자료(training data)와 평가자료(testing data)를 8:2로 나눈다.
    ###########################################################
    train_data = data.drop("HeartDisease", axis = 1).values
    train_target = data["HeartDisease"].values
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_target, test_size=0.2)
    # print(X_train)
    # print(y_train)
    # for i in range(100):
    #     print(y_train[i])

    ###########################################################
    ##5.분류기는 아래의 classifier를 구현하고 10-fold cross-validation(CV)으로 성능을 측정한다. 각 분류기마다 아래와 같은 표을 작성하여 제출하세요. SCV의 경우에는 sampling 개념이 들어 있어서 절대 같은 값이 나올 수 없음을 주의하시기 바랍니다
    ##  1).LogisticRegression (LR)
    ##  2).SGDClassifier (SGD)
    ###########################################################
    LR = LogisticRegression(max_iter=50,
                           warm_start=True
                            )
    SGD = SGDClassifier(max_iter=50, 
                            tol=-np.infty, 
                            warm_start=True,                       
                            penalty=None, 
                            learning_rate="constant", 
                            eta0=0.4, random_state=42)
    ###########################################################
    ##  1).LogisticRegression (LR)
    kfold = KFold(n_splits=10, shuffle=True)
    X = train_data
    y = train_target
    print("Logistic Regression")
    print("      "+"LR precision 점수   " + "  LR recall 점수    " + "     LR f1-score 점수")
    for epoch in range(10):
        kfold = KFold(n_splits=10, shuffle=True)
        LR.fit(X_train, y_train)
        
        scoring1 = ['precision', 'recall', 'f1', 'accuracy',
                         'precision_micro', 'precision_macro', 
                         'recall_micro', 'recall_macro',
                        'f1_micro', 'f1_macro']
        scores = cross_validate(LR, X, y, scoring=scoring1 ,cv=kfold)
        sorted(scores.keys())

        print(epoch+1,"   ", scores['test_precision'].mean(),"  ", scores['test_recall'].mean(),"  ", scores['test_f1'].mean())
        

        if(epoch >=9):
            print("mic-avg", end=""); print(scores['test_precision_micro'].mean(), scores['test_recall_micro'].mean(), scores['test_f1_micro'].mean());
            print("mac-avg", end=""); print(scores['test_precision_macro'].mean(), scores['test_recall_macro'].mean(), scores['test_f1_macro'].mean())
    ###########################################################

    ###########################################################
    ##  2).SGDClassifier (SGD)
    kfold = KFold(n_splits=10, shuffle=True)
    X = train_data
    y = train_target
    print("SGD Classfication")
    print("      "+"SGD precision 점수   " + "  SGD recall 점수    " + "     SGD f1-score 점수")
    for epoch in range(10):
        kfold = KFold(n_splits=10, shuffle=True)
        SGD.fit(X_train, y_train)
        
        scoring1 = ['precision', 'recall', 'f1', 
                         'precision_micro', 'precision_macro', 
                         'recall_micro', 'recall_macro',
                        'f1_micro', 'f1_macro']
        scores = cross_validate(SGD, X, y, scoring=scoring1 ,cv=kfold)
        sorted(scores.keys())
        print(epoch+1,"   ", scores['test_precision'].mean(),"  ", scores['test_recall'].mean(),"  ", scores['test_f1'].mean())
        
#         y_val_predict = SGD.predict(X_val)
#         y_train_predict = SGD.predict(X_train)
#         print(accuracy_score(y_train, y_train_predict))
        if(epoch >=9):
            print("mic-avg", end=""); print(scores['test_precision_micro'].mean(), scores['test_recall_micro'].mean(), scores['test_f1_micro'].mean());
            print("mac-avg", end=""); print(scores['test_precision_macro'].mean(), scores['test_recall_macro'].mean(), scores['test_f1_macro'].mean())
            
    ###########################################################