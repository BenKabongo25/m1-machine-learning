# Machine Learning
# Sorbonne Université - Master DAC
# Ben Kabongo
#
# TME 04

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

perceptron_loss = lambda w, X, y: np.maximum(0, -y * (X@w))
perceptron_grad = lambda w, X, y: ((-y * (X@w)) > 0) * (-y * X)
hinge_loss = lambda w, X, y, alpha,lmbda: np.maximum(0, alpha - y * (X@w)) + lmbda * (w@w).sum()
hinge_loss_grad = lambda w, X, y, alpha, lmbda: (((alpha -y * (X@w)) > 0) * (-y * X)) + lmbda * w.reshape(-1)


class GradientDescentMode:
    BATCH       = 0
    MINI_BATCH  = 1
    STOCHASTIC  = 2

class InitializationMode:
    ONE    = 0
    RANDOM = 1


class Lineaire(object):
    def __init__(self,
                loss=perceptron_loss,
                loss_g=perceptron_grad,
                max_iter=100,               # epochs
                eps=1e-2,                   # learning rate
                projection=None,            # fonction de projection
                projection_args={},         # projection args, if proj_gauss
                penalty=False,              # penalty : False -> None, True -> L2
                penalty_alpha=0,            # penalty alpha
                penalty_lambda=1e-2,        # penalty coef
                gradient_mode=GradientDescentMode.BATCH,
                batch_size=100,              # batch size if Mini batch
                initialization_mode=InitializationMode.RANDOM
                ):
        self.max_iter = max_iter
        self.eps = eps
        self.loss = loss
        self.loss_g = loss_g

        # parameters
        self.w = None
        self.initialization_mode = initialization_mode

        # projection
        self.projection = projection
        if projection == proj_gauss:
            self.projection = lambda X: proj_gauss(X, **projection_args)

        # gradient mode
        self.gradient_mode  = gradient_mode
        self.batch_size     = batch_size
        if self.gradient_mode == GradientDescentMode.STOCHASTIC:
            self.batch_size = 1

        # penalty
        if penalty and loss==hinge_loss:
            self.loss = lambda w, X, y: hinge_loss(w, X, y, penalty_alpha, penalty_lambda)
            self.loss_g = lambda w, X, y: hinge_loss_grad(w, X, y, penalty_alpha, penalty_lambda)

        self.all_w                  = []
        self.loss_values            = []
        self.loss_test_values       = []
        self.accuracy_values        = []
        self.accuracy_test_values   = []
        
    def fit(self, X, y, X_test=None, y_test=None):
        X_proj = X
        X_test_proj = X_test
        if self.projection is not None:
            X_proj = self.projection(X)
            X_test_proj = self.projection(X_test)

        if self.initialization_mode == InitializationMode.RANDOM:
            self.w = .001 * (2 * np.random.uniform(0, 1, X_proj.shape[1]) - 1)
            self.w = self.w.reshape(-1, 1)
        else:
            self.w = .001 * np.ones((X_proj.shape[1], 1))
        self.all_w = [self.w]

        self.loss_values = [self.loss(self.w, X_proj, y).mean()]
        self.accuracy_values = [self.score(X, y)]

        test = (X_test is not None) and (y_test is not None)
        if test:
            self.loss_test_values = [self.loss(self.w, X_test_proj, y_test).mean()]
            self.accuracy_test_values = [self.score(X_test, y_test)]

        N = len(X)
        for _ in range(self.max_iter):

            # Mini batch : m , stochastic : 1
            if self.gradient_mode in (GradientDescentMode.MINI_BATCH, GradientDescentMode.STOCHASTIC):
                idx = np.random.choice(N, self.batch_size)
                X_batch = X_proj[idx]
                y_batch = y[idx]

            # Batch : N
            else:
                X_batch = X_proj
                y_batch = y

            self.w -= self.eps * self.loss_g(self.w, X_batch, y_batch).mean(0).reshape(-1,1)
            self.all_w.append(self.w)

            self.loss_values.append(self.loss(self.w, X_proj, y).mean())
            self.accuracy_values.append(self.score(X, y))

            if test: 
                self.loss_test_values.append(self.loss(self.w, X_test_proj, y_test).mean())
                self.accuracy_test_values.append(self.score(X_test, y_test))

    def predict(self, X):
        if self.projection is not None:
            X = self.projection(X)
        return np.sign(X @ self.w)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.where(y_pred == y, 1, 0).mean()


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest")

def proj_poly(X):
    n, d = X.shape
    d_ = int(1 + 2*d + (d *(d-1)/2))
    X_ = np.zeros((n, d_))
    X_[:, 0] = 1
    k = 2*d
    for j in range(d):
        X_[:, j+1] = X[:, j]
        X_[:, j+1+d] = X[:, j]**2
        for i in range(j+1,d):
            k += 1
            X_[:, k] = X[:, j] * X[:, i]
    return X_

def proj_biais(X):
    return np.append(X, np.ones((len(X), 1)),axis=1)

def proj_gauss(X, base, sigma):
    assert X.shape[1] == base.shape[1]
    return np.array([(((x - base)**2).sum(1) / 2*sigma) for x in X])
    

if __name__ =="__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
