# -*- coding: utf-8 -*-
from __future__ import division

import math
from random import random

import numpy as np
from sklearn.cross_validation import train_test_split

"""
This module is for data generating purposes.
"""

def generateSinusCluster(size, location=0.0, scale=0.5, amplitude=0.2, freq=3):
    """
    Function that genrates a random gaussian cluster and separates between two classes with a sinus function.

    @type size: number
    @param size: Size of the generated dataset
    @type location: float
    @param size: Mean (“centre”) of the distribution.
    @type scale: float
    @param scale: Standard deviation (spread or “width”) of the distribution.
    @type amplitude: float
    @param amplitude: Amplitude of the underlying sinus function.
    @type freq: float
    @param freq: Frequency of the underlying sinus function
    @rtype: Array
    @return: Arrays for datapoints and labels
    """

    data = np.random.normal(loc=location, scale=scale, size=(size, 2))
    dataX = np.array([row[0] for row in data])
    dataY = np.array([row[1] for row in data])

    label = np.zeros(size)
    for i in range(size):
        if dataY[i] > amplitude * math.sin(dataX[i] * math.pi * freq):
            label[i] = 1
        else:
            label[i] = -1

    return data, label


def generateNonLinearClusters(size, x_noise=False, y_noise=False, noise_prob=0.2):
    x1 = np.random.normal(size=(size, 2))
    x1[:, 0] += 2
    y1 = np.ones(size) * 1

    x2 = np.random.normal(size=(size, 2))
    x2[:, 0] -= 2
    x2[:, 1] += 6
    y2 = np.ones(size) * -1

    x3 = np.random.normal(size=(size, 2))
    x3[:, 0] += 2
    x3[:, 1] += 12
    y3 = np.ones(size) * 1

    X = np.vstack((np.vstack((x1, x2)), x3))
    Y = np.concatenate((np.concatenate((y1, y2)), y3))

    if x_noise:
        X = _add_x_noise(X, noise_prob)
    if y_noise:
        Y = _add_y_noise(Y, noise_prob)

    x, x_test, y, y_test = train_test_split(X, Y, train_size=0.6)

    return x, x_test, y, y_test


def _add_x_noise(X, noise_prob):
    for i in range(X.shape[0]):
        if random() > 1 - noise_prob:
            X[i] = X[i] + random()
    return X


def _add_y_noise(Y, noise_prob):
    for i in range(len(Y)):
        if random() > 1 - noise_prob:
            Y[i] = Y[i] * -1
    return Y
