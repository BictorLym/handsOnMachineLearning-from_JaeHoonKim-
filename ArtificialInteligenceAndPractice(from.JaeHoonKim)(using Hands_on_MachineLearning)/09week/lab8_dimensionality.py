import sys
import sklearn
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
from mlxtend.data import loadlocal_mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



import warnings
warnings.filterwarnings("ignore")

def performance(y_test, y_test_pred, average='binary'):
    precision = precision_score(y_test, y_test_pred, average=average)
    recall = recall_score(y_test, y_test_pred, average=average)
    f1 = f1_score(y_test, y_test_pred, average=average)
    return precision, recall, f1


def build_3d_data():
    np.random.seed(4)
    m = 60
    w1, w2 = 0.1, 0.3
    noise = 0.1

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 3))
    X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
    return X

def PCA_using_SVD(X):
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered)
    W2 = Vt.T[:, :2]
    return X_centered.dot(W2)


def load_mnist(path):
    X_train, y_train = loadlocal_mnist(
            images_path= path+'train-images.idx3-ubyte', 
            labels_path= path+'train-labels.idx1-ubyte')
    X_test, y_test = loadlocal_mnist(
            images_path= path+'t10k-images.idx3-ubyte', 
            labels_path= path+'t10k-labels.idx1-ubyte')
    return X_train, X_test, y_train, y_test

    
#main program
if __name__ == '__main__':
    X = build_3d_data()
#     print(X[:5,:])
    
#     X2D_using_svd = PCA_using_SVD(X)
#     print(X2D_using_svd[:5, :])
    
#     pca = PCA(n_components=2)
#     X2D = pca.fit_transform(X)
#     print(X2D[:5, :])
# ##
#     print(np.allclose(X2D, -X2D_using_svd)) # 2 matrix가 거의 같은지를 확인한다.
# ##
# ##    
#     X3D_inv = pca.inverse_transform(X2D)
#     print(np.allclose(X3D_inv, X)) # projection 단계에서 약간의 손실이 있다.
##
    # # choosing the right number of dimensions
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print(cumsum)
    d = np.argmax(cumsum >= 0.95) + 1
    print('the right number of dimensions :', d)

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
    print("X_reduced: ", X_reduced[:5, :])
    
    # MNIST compression
    # MNIST_PATH = './datasets/mnist/'
    MNIST_PATH = 'C:\\Users\\mycom0703\\Desktop\\Young\\20.github_shared\\01.python\\AI\\handsOnMachineLearning-from_JaeHoonKim-\\ArtificialInteligenceAndPractice(from.JaeHoonKim)(using Hands_on_MachineLearning)\\09week\\mnist\\'
    X_train, X_test, y_train, y_test = load_mnist(MNIST_PATH)
    
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print('the right number of dimensions :', d)
    pca = PCA(n_components=154)
    X_train_reduced = pca.fit_transform(X_train)

    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_train_reduced = pca.transform(X_train)
    print('the right number of dimensions :', pca.n_components_)
    
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    rnd_clf.fit(X_train_reduced, y_train)

    X_test_reduced = pca.transform(X_test)
    y_pred = rnd_clf.predict(X_test_reduced)
    print('Random Forests:', accuracy_score(y_test, y_pred))

    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
    rnd_clf.fit(X_train, y_train)
    y_pred = rnd_clf.predict(X_test)
    print('Random Forests:', accuracy_score(y_test, y_pred))

    # Incremental PCA
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X_train, n_batches):
        print(".", end="") # not shown in the book
        inc_pca.partial_fit(X_batch)

    X_reduced = inc_pca.transform(X_train)

    # Using memmap()
    # save 
    filename = "my_mnist.data"
    m, n = X_train.shape

    X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
    X_mm[:] = X_train

    del X_mm

    #another program would load the data and use it for training
    X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

    batch_size = m // n_batches
    inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
    X_reduced = inc_pca.fit_transform(X_mm)

    # LLE(locally linear embedding): nonlinear dimensionality reduction
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    X_reduced = lle.fit_transform(X)

    # MDS, Isomap and t-SNE
    mds = MDS(n_components=2, random_state=42)
    X_reduced_mds = mds.fit_transform(X_train)

    isomap = Isomap(n_components=2)
    X_reduced_isomap = isomap.fit_transform(X_train)

    tsne = TSNE(n_components=2, random_state=42)
    X_reduced_tsne = tsne.fit_transform(X_train)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    X_reduced_lda = lda.transform(X_train)
    
    


    

    
    

    
    

    
    
    


    

    

    
    
          
    
    
    

    



    


    



    
 

    














