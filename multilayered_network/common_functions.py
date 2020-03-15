import numpy as np


# step activation function
def step(weighted_sum):
    if weighted_sum <= 0:
        return 0
    else:
        return 1


# sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def cross_entropy(a, y):
    return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))


def mean_squared(a, y):
    return np.sum((a - y) ** 2) / y.size
