import sys
import sklearn
import numpy as np
import os

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy import stats
from sklearn.metrics import accuracy_score
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import silhouette_score

from matplotlib.image import imread
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline



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

def load_next_batch(X, batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]
    
#main program
if __name__ == '__main__':
    data = load_iris()
    X, y = data.data, data.target

    gm = GaussianMixture(n_components=3, random_state=42)
    gm.fit(X)
    y_pred = gm.predict(X)
    print(y)
    print(y_pred)

    mapping = {}
    for class_id in np.unique(y):
        mode, _ = stats.mode(y_pred[y==class_id])
        mapping[mode[0]] = class_id
    print(mapping)

    y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])
    print(accuracy_score(y, y_pred))

    blob_centers = np.array(
       [[ 0.2,  2.3],
        [-1.5 ,  2.3],
        [-2.8,  1.8],
        [-2.8,  2.8],
        [-2.8,  1.3]])
    blob_std =np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers,
                 cluster_std=blob_std, random_state=7)

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    print(blob_centers)
    print(kmeans.cluster_centers_)
    
    print(y_pred)
    print(kmeans.labels_)


   # 새로운 data에 대한 cluster 예측
    X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
    print(kmeans.predict(X_new))

   # 새로운 data와 centeriod와의 거리 측정
    X_dist = kmeans.transform(X_new)
    print(X_dist)
##    
   # 모든 data와 centeriod와의 거리의 합
   # 이것이 가장 작은 것이 가장 좋은 모델임.
    print(kmeans.inertia_)
##
##    # 따라서 kmeans의 score는 inertia를 이용한다.
##    # -inertia : why score가 높은 모델이 좋은 모델임.
    print(kmeans.score(X))
##
##    # Mini-Batch K-Means
    minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
    minibatch_kmeans.fit(X)
    print(minibatch_kmeans.inertia_)
##
    MNIST_PATH = './datasets/mnist/'
    X_train, X_test, y_train, y_test = load_mnist(MNIST_PATH)
    batch_size = 1000
    minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=batch_size,
                                        random_state=42)

    n_iterations = 10  
    for _ in range(n_iterations):
        for _ in range(len(X_train)//batch_size):
            X_init = load_next_batch(X_train, batch_size)
            minibatch_kmeans.partial_fit(X_init)
##   
    print(minibatch_kmeans.inertia_)
##
    print(silhouette_score(X_train, minibatch_kmeans.labels_))

    ## 이건 이미지를 찾아야함##########
    #Using clustering for image segmentation
    # filename = "ladybug.png"
    # images_path = "images/unsupervised_learning"

    # image = imread(os.path.join(images_path, filename))
    # print(image.shape)
    # X = image.reshape(-1, 3)
    # print(X.shape)

##    kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
##    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
##    segmented_img = segmented_img.reshape(image.shape)
##    plt.imshow(segmented_img)
##    plt.show()
##    
##    #Using Clustering for Preprocessing
##    X_digits, y_digits = load_digits(return_X_y=True)
##    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
##    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
##    log_reg.fit(X_train, y_train)
##    log_reg_score = log_reg.score(X_test, y_test)
##    print(log_reg_score)
##
##    
##    pipeline = Pipeline([
##        # 영상을 cluster의 centeriod와의 거리로 변환됨
##        # KMeans(n_clusters=50, random_state=42).fit_transform(X_train)
##        ("kmeans", KMeans(n_clusters=50, random_state=42)), 
##        ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
##    ])
##    pipeline.fit(X_train, y_train)
##
##    pipeline_score = pipeline.score(X_test, y_test)
##    print(pipeline_score) 
##    print('the error rate drop:', 1 - (1 - pipeline_score) / (1 - log_reg_score))
##    
    

    
    

    
    
    


    

    

    
    
          
    
    
    

    



    


    



    
 

    














