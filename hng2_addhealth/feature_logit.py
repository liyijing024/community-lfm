"""
Predict a feature in a Facebook ego network with Logistic Regression.
"""

import numpy as np
import scipy.optimize as opt
from scipy.stats.distributions import norm
import sys
import matplotlib.pyplot as plt
import math


def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))

# negative log likelihood since we use a minimizing function
def logLikelihood(theta, X, y, lam):
    ll = -np.sum(np.log(1+np.exp(-np.dot(X, theta)))) - np.sum((1-y)*(np.dot(X, theta)))
    # regularizer
    # L2
    #ll -= lam*np.sum(np.square(theta))
    # L1 for feature selection
    ll -= lam*sum(abs(t) for t in theta)
    return -ll

# negative derivative since we use a minimizing function
def fprime(theta, X, y, lam):
    dl = np.zeros_like(theta)
    sig_logit = 1-sigmoid(np.dot(X, theta))
    for j in range(len(theta)):
        dl[j] += np.sum(X[:,j]*sig_logit, axis=0)
        dl[j] -= np.sum(X[:,j]*(-(y-1)))
        # L2
        #dl[j] -= 2*lam*theta[j]
        # L1
        #dl[j] -= lam*(np.absolute(theta[j])/theta[j])
        dl[j] -= lam
    return -dl

def pred(a, t):
    if a >= t:
        return 1.0
    else:
        return 0.0

vpred = np.vectorize(pred)


def rmse(y1, y2):
    return np.sqrt(((y1 - y2)**2).mean())


def train_model(X, y, lam):
    # random initialization
    eps = 0.00001
    theta0 = np.random.rand(len(X[0]))*eps-(eps/2)
    #theta0 = np.zeros_like(X[0])
    # minimize negative loglikelihood
    theta,l,info = opt.fmin_l_bfgs_b(logLikelihood, theta0, fprime, args = (X, y, lam))
    return theta


def test_model(X, theta, y):
    #y_pred_raw = [sigmoid(inner(X[i], theta)) for i in range(len(X))]
    y_pred_raw = sigmoid(np.dot(X, theta))
    err = rmse(y_pred_raw, y)
    t = 0.5
    y_pred = vpred(y_pred_raw, t)
    acc = np.mean(y_pred == y)

    pred1 = 0
    acc1 = 0
    for i in range(len(y)):
        if y[i] == 1:
            acc1 += 1
        if y_pred[i] == 1:
            pred1 += 1

    percent_pred_pos = pred1*1.0/len(y)
    percent_actual_pos = acc1*1.0/len(y)

    return err, acc, percent_pred_pos, percent_actual_pos

