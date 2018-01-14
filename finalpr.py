# author: Weng
from sklearn.datasets import *
from sklearn.decomposition import PCA
from scipy import spatial
import numpy as np
from ppca_weng import simple_PPCA, PPCA
from ppca_wang import beyesian_PPCA
import argparse

import matplotlib.pyplot as plt

datasets_toy = ['iris', 'digits', 'wine', 'breast_cancer']

def calcGE(trX,trY):
    errcount = 0.0
    # use kdtree to do 1-NN classification expect for the target point itself
    kdmap = spatial.KDTree(trX)
    for i in range(trX.shape[0]):
        qr = trX[i]
        trueY = trY[i]

        dis, ind = kdmap.query(qr,k=2)
        # choose the neighbor
        if trY[ind[1]] != trueY:
            errcount = errcount + 1
    return errcount/(trX.shape[0])

def testwithToy(dataname):
    # load toy datasets example
    X = None
    y = None
    remain_dim = None

    if dataname == 'iris':
        data = load_iris()
        y=data.target
        X=data.data
    elif dataname == 'digits':
        data = load_digits()
        y = data.target
        X = data.data
    elif dataname == 'wine':
        data = load_wine()
        y = data.target
        X = data.data
    elif dataname == 'breast_cancer':
        data = load_breast_cancer()
        y = data.target
        X = data.data

    # visulization in 2D
    # standard PCA
    dim = X.shape[1]
    # remained dim
    if dim > 20:
        remain_dim = 10
    else:
        remain_dim = dim - 1

    print("The original dimension: {}.".format(dim))
    print("We are now increase the principle component from 1 to {}.".format(remain_dim))
    GE_standard = np.zeros(remain_dim)
    GE_ppca = np.zeros(remain_dim)
    GE_bpca = np.zeros(remain_dim)
    for i in range(remain_dim):
        # standard PCA
        print("==> remain component number:{}".format(i+1))
        skpca = PCA(n_components=i + 1)
        skreduce_X = skpca.fit_transform(X)
        GE_standard[i] = calcGE(skreduce_X, y)
        print("Generation error of standard PCA --> {}".format(GE_standard[i]))
        # ppca use EM
        # pp = PPCA() # use general version with sigma
        pp = simple_PPCA() # use simple version without sigma
        pp.fit(X, i + 1)
        ppcareduce_X = pp.transform()
        #print(ppcareduce_X.shape)
        GE_ppca[i] = calcGE(ppcareduce_X, y)
        print("Generation error of PPCA --> {}".format(GE_ppca[i]))
        # Bayesian PCA use EM
        bpca = beyesian_PPCA()
        bpca.fit(X, i + 1)
        bpcaLatent = bpca.transform()
        print(bpcaLatent.shape)
        GE_bpca[i] = calcGE(bpcaLatent, y)
        print("Generation error of BPCA --> {}".format(GE_bpca[i]))

    line_standard, = plt.plot(np.arange(1, remain_dim + 1), GE_standard, label='Line Standard PCA')
    line_emppca, = plt.plot(np.arange(1, remain_dim + 1), GE_ppca, label='Line EM PPCA')
    line_bpca, = plt.plot(np.arange(1, remain_dim + 1), GE_bpca, label='Line Bayesian PCA')    

    plt.legend(handles=[line_standard, line_emppca, line_bpca])
    plt.xlabel('Remained number of component')
    plt.xlabel('Generation error with 1-NN')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test generation error of PPCA and standard PCA in toy dataset")
    parser.add_argument("dataset_id", type=int, help="the id number of toy dataset")
    dataset_name = datasets_toy[parser.parse_args().dataset_id]
    print("Test on the toy dataset: {}".format(dataset_name))
    testwithToy(dataset_name)
