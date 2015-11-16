from __future__ import division

import numpy as np


def getPointsCloseToHyperplaneByFactor(clf, X, factor):
    """
    Function that tries to find data close to the separating hyperplane by a scaling factor.

    @type clf: Classifier
    @param clf: Classifier to be used. LinearSVC expected.
    @type X: Array
    @param X: Array of Datapoints to be searched in.
    @type factor: float
    @param factor: Factor that determines how close the points to the separating hyperplane should be.

    @rtype: Array
    @return: Points above and below the hyperplane by margin times factor. Array of data points that match the above criterias.
    """
    above = np.array((0, 0))
    below = np.array((0, 0))
    w = clf.coef_[0]
    b = clf.intercept_[0]

    for i in range(X.shape[0]):
        posMargin = (np.inner(w, X[i]) + b) + factor
        if (posMargin >= 0):
            above = np.vstack((above, X[i]))
        else:
            below = np.vstack((below, X[i]))
    return above[1:], below[1:]


def getPointsCloseToHyperplaneByCount(clf, X, count):
    # TODO implement this method
    return 0
