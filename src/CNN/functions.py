import numpy as np


def relu(X):
    return np.maximum(0, X)


def relu_backward(Z):
    dz = Z.copy()
    dz[Z > 0] = 1
    return dz


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / np.sum(expZ)


def softmax_backward(r):
    soft = softmax(r)
    return soft * (1 - soft)


def cross_entropy(p):
    return -np.log(p)
