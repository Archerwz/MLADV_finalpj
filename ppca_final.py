# include PPCA, kernel PCA, beyesian PCA
import numpy as np
from math import *

from sklearn.datasets import *
from sklearn.decomposition import PCA
from scipy import spatial
import numpy as np
#from ppca_weng import simple_PPCA, PPCA
import argparse

# </author: ZEHANG WENG>
def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

# sigma=0
class simple_PPCA():
    def __init__(self):
        self.data = None
        self.stddata = None
        self.n = None
        self.dim = None
        self.mean = None
        self.W = None
        self.M = None
        self.EZ = None
        self.redim = None
        np.random.seed(10)

    def fit(self, data, reduce_dim=2):
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.data = reduce_dim
        self.redim = reduce_dim
        # calculate the mean of input data
        self.mean = np.mean(data, axis=0)
        # standarize the data
        # self.stddata = data / self.mean
        self.stddata = (data - self.mean).T
        # initiate the projection matrix W
        self.W = np.random.random((self.dim, self.redim))
        # old_W = self.W

        stop=0
        # EM
        while(True):
            # E step
            # self.EZ = np.dot( np.dot( np.linalg.pinv(np.dot(self.W.T, self.W)) ,self.W.T), self.stddata)，出来全是nan
            self.EZ = np.dot(np.dot(np.linalg.pinv(np.dot(self.W.T, self.W)), self.W.T), self.stddata)
            # if self.EZ.shape != (self.n, self.redim):
            #     print(self.EZ.shape)
            #     raise RuntimeError("The size of EZ is wrong")
            # print(self.EZ)
            if self.EZ.shape != (self.redim, self.n):
                print(self.EZ.shape)
                raise RuntimeError("The size of EZ is wrong")
            self.M = np.dot(self.W.T, self.W)

            # M step
            # self.W = np.dot(np.dot(self.stddata.T, self.EZ.T), np.linalg.pinv(np.dot(self.EZ, self.EZ.T)))
            # print(self.W)
            self.W = np.dot(np.dot(self.stddata, self.EZ.T), np.linalg.pinv(np.dot(self.EZ, self.EZ.T)))

            # check converge or not
            # actually we need to use E[lnp(X,Z|mu,W,sigma)] to check convergence
            # if (1 - cosine_similarity(self.W.reshape(-1), old_W.reshape(-1))) < 0.00001:
            if stop > 10000:
                break
            else:
                # old_W = self.W
                stop = stop + 1

    def transform(self, newinputx=None):
        if newinputx is None:
            latent_z = np.dot(np.dot(np.linalg.pinv(self.M), self.W.T), self.stddata)
        else:
            if newinputx.shape[1] != self.dim:
                raise RuntimeError("The shape of input X should be KxD")
            else:
                newinputx = newinputx.reshape(self.dim, -1)
                latent_z = np.dot(np.dot(np.linalg.pinv(self.M), self.W.T), (newinputx - self.mean))
        if latent_z.shape[0] != self.redim:
            raise RuntimeError("The size of latent z is wrong")
        return latent_z.T

# if want to ignore sigma, please use standard_PPCA with fast speed
class PPCA():
    def __init__(self):
        # data: DxN
        self.data = None
        # stddata: DxN
        self.stddata = None
        self.n = None
        # dim: D
        self.dim = None
        self.mean = None
        # W: DxM
        self.W = None
        self.sigma = None
        self.M = None
        # EZ: MxN
        self.EZ = None
        # EZEZT: MxM
        self.EZZT = None
        # EZnEZnT: MxM
        self.EZnZnT = None
        # redim: M
        self.redim = None
        np.random.seed(10)

    def fit(self, data, reduce_dim=2):
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.data = data.T
        self.redim = reduce_dim
        # calculate the mean of input data
        self.mean = np.mean(data, axis=0)
        # standarize the data
        # self.stddata = data / self.mean
        self.stddata = (data - self.mean).T
        # initiate the projection matrix W
        self.W = np.random.random((self.dim, self.redim))
        # initiate the noise sigma
        self.sigma = np.random.random()*0.001
        # initiate EZ, EZZT, EZnZnT
        self.EZ = np.zeros((self.redim, self.n))
        self.EZZT = np.zeros((self.redim, self.redim))
        self.EZnZnT = np.zeros((self.n,self.redim, self.redim))

        stop=0
        # EM
        while(True):
            # E step
            # self.EZ = np.dot( np.dot( np.linalg.pinv(np.dot(self.W.T, self.W)) ,self.W.T), self.stddata)，出来全是nan
            # calculate M: MxM
            self.M = np.dot(self.W.T, self.W) + self.sigma * np.eye(self.redim, self.redim)
            # EZ: MxN
            self.EZ = np.dot(np.dot(np.linalg.pinv(self.M), self.W.T), self.stddata)
            # EZEZT: MxM
            self.EZZT = self.sigma * np.linalg.pinv(self.M) + np.dot(self.EZ, self.EZ.T)
            # EZnZnT: MxM, need to store n map, because in the update of sigma, we need to calculate a matrix multiplication in the trace
            for i in range(self.n):
                EZn = self.EZ[:,i].reshape(-1,1)
                self.EZnZnT[i] = self.sigma * np.linalg.pinv(self.M) + np.dot(EZn, EZn.T)

            # if self.EZ.shape != (self.n, self.redim):
            #     print(self.EZ.shape)
            #     raise RuntimeError("The size of EZ is wrong")
            # print(self.EZ)
            if self.EZ.shape != (self.redim, self.n):
                print(self.EZ.shape)
                raise RuntimeError("The size of EZ is wrong")
            # self.M = np.dot(self.W.T, self.W)

            # M step
            # self.W = np.dot(np.dot(self.stddata, self.EZ), np.linalg.pinv(np.dot(self.EZ, self.EZ.T)))
            self.W = np.dot(np.dot(self.stddata, self.EZ.T), np.linalg.pinv(self.EZZT))
            if self.W.shape != (self.dim, self.redim):
                raise RuntimeError("The size of W is wrong")
            # updata sigma
            newsigma = 0
            for i in range(self.n):
                # get xn and calculate the power of norm
                stdXn = self.stddata[:,i].reshape(-1,1)
                l2normXn = np.linalg.norm(stdXn) ** 2
                # get EZn, Mx1redim
                EZn = self.EZ[:,i].reshape(-1,1)
                # get EZnZnt, MxM
                EZnZnt = self.EZZT[:,]
                newsigma = newsigma + l2normXn - 2 * np.dot(np.dot(EZn.T, self.W.T), stdXn) + np.trace(np.dot(np.dot(self.EZnZnT[i], self.W.T), self.W))
            self.sigma = newsigma/(self.n*self.dim)
            # check converge or not
            # actually we need to use E[lnp(X,Z|mu,W,sigma)] to check convergence
            # if (1 - cosine_similarity(self.W.reshape(-1), old_W.reshape(-1))) < 0.00001:
            if stop > 100:
                break
            else:
                # old_W = self.W
                stop = stop + 1

    # newinputx should be KxD
    # output KxM
    def transform(self, newinputx=None):
        if newinputx is None:
            latent_z = np.dot(np.dot(np.linalg.pinv(self.M), self.W.T), self.stddata)
        else:
            if newinputx.shape[1] != self.dim:
                raise RuntimeError("The shape of input X should be KxD")
            else:
                newinputx = newinputx.reshape(self.dim, -1)
                latent_z = np.dot(np.dot(np.linalg.pinv(self.M), self.W.T), (newinputx - self.mean))
        if latent_z.shape[0] != self.redim:
            raise RuntimeError("The size of latent z is wrong")
        return latent_z.T
# </ Weng>

# Zhou Wang
# newinputx should be KxD
# output KxM
class kernel_PPCA():
    def __init__(self):
        self.data = None
        self.stddata = None
        self.n = None
        self.dim = None
        self.mean = Noneredim
        self.redim = None
        np.random.seed(10)

    def fit(self, data, reduce_dim=2):
        # input data: NxD
        pass
    def transform(self, newinputx=None):
        # newinput: KxD
        pass

# Beyesian Zesen Wang
# newinputx should be KxD
# output KxM
class bayesian_PPCA():
    
    def __init__(self):
        np.random.seed(10)
        self.run = False
        pass
        
    def fit(self, data, reduce_dim=2):
    
        self.realM = reduce_dim
        
        if self.run:
            return
        self.run = True
        # self.data: d * n
        # reduce: d -> m
        self.n = data.shape[0]
        self.d = data.shape[1]
        self.m = self.d - 1
        #self.realM = reduce_dim
        self.mean = np.mean(data, axis=0)
        self.data = data.T
        self.nData = (data - self.mean).T
        self.W = 10 * np.random.random((self.d, self.m)) - 5
        self.alpha = 10 * np.random.random((self.m, ))
        self.sigma = np.random.random() * 0.01
        iteration = 0
        while True:
            self.M = np.dot(self.W.T, self.W) + self.sigma * np.eye(self.m, self.m)
            Z = np.dot(np.dot(np.linalg.pinv(self.M), self.W.T), self.nData)
            ZZT = [np.zeros((self.m, self.m)) for i in range(self.n)]
            for i in range(self.n):
                ZZT[i] = self.sigma * np.linalg.pinv(self.M) + np.dot(Z[:, i, None], Z[:, i, None].T)
            # update alpha
            for i in range(self.m):
                self.alpha[i] = self.d / np.dot(self.W[:, i], self.W[:, i])
            # update W
            A = np.diag(self.alpha)
            left = np.zeros((self.d, self.m))
            right = self.sigma * A
            for i in range(self.n):
                left = left + np.dot(self.nData[:, i, None], Z[:, i, None].T)
                right = right + ZZT[i]
            self.W = np.dot(left, np.linalg.pinv(right))
            # update sigma
            self.sigma = 0.
            for i in range(self.n):
                self.sigma = self.sigma + np.dot(self.nData[:, i], self.nData[:, i]) \
                        - 2 * np.dot(np.dot(Z[:, i, None].T, self.W.T), self.nData[:, i, None]) \
                        + np.trace(np.dot(np.dot(ZZT[i], self.W.T), self.W))
            self.sigma = self.sigma / self.n / self.d
            if iteration > 100:
                break
            iteration += 1
            
        pass
    def transform(self, newinputx=None):
        if newinputx is None:
            index = np.argsort(self.alpha)
            selectW = np.zeros((self.d, 0))
            for i in range(self.realM):
                selectW = np.insert(selectW, [i], self.W[:,index[i], None], axis = 1)
            self.M = np.dot(selectW.T, selectW) + self.sigma * np.eye(self.realM, self.realM)
            return np.dot(np.dot(np.linalg.pinv(self.M), selectW.T), self.nData).T
        pass
        
# For debug
if __name__ == "__main__":
    data = load_iris()
    y = data.target
    X = data.data
    bpca = beyesian_PPCA()
    bpca.fit(X, 1)
    print (bpca.alpha)
    print (np.argsort(bpca.alpha))
    print (bpca.transform())

        
        
        
        
        
        
        
        
        
