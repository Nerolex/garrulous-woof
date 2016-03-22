from __future__ import division

import numpy as np


def score(y_hat, y):
    """
    Method that calculates the score for a given test set X,y.
    :param X: Test vector of datapoints
    :param y: Test vector of labels
    :return: score value
    """
    score = 0
    y_hat = np.array(y_hat)
    y = np.array(y)
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            score += 1
    score /= y.shape[0]
    return score


def score2(y_hat, y):
    y_hat = np.array(y_hat)
    y = np.array(y)
    score = y_hat == y
    return np.average(score)


y1 = [1, 1, 1, 1, -1, -1, -1, -1]
y2 = [1, -1, -1, -1, -1, -1, -1, -1]

print score(y1, y2)
print score2(y1, y2)
