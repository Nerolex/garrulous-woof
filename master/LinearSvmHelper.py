from __future__ import division

import numpy as np


def getHyperplane(clf):
    """
    Function that extracts the separating hyperplane out of an linear SVC classifier and returns a plottable line.

    @type clf: Classifier
    @param clf: A linear SVC classifier.
    @rtype: Array
    @return: Array of X and Y coordinates of a segment of the separating hyperplane
    """

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    return xx, yy


def getMarginPlanes(clf, factor=1.0):
    """
    Function that calculates the margin parallels to the separating hyperplane of an linear SVC classifier and returns two plottable lines.

    @type clf: Classifier
    @param clf: A linear SVC classifier.
    @rtype: Array
    @type factor: float
    @param factor: Optional scaling factor for plotting purposes.
    @return: Arrays of X and Y coordinates of a segment of the parallels of the separating hyperplane
    """

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    yy_down = yy + factor * margin
    yy_up = yy - factor * margin

    return yy_down, yy_up


def hyperplane(clf, X, constant):
    """
    Function that classifies points with the hyperplane of the given classifier. A constant is used to move the hyperplane.

    @type clf: Classifier
    @param clf: Classifier to be used. LinearSVC expected.
    @type X: Array
    @param X: Array of Datapoints to be classified.
    @type constant: float
    @param constant: Constant that is used to move the hyperplane.

    @rtype: Array
    @return: Points above and below the hyperplane by margin times factor. Array of data points that match the above criterias.
    """
    above = np.array((0, 0))
    below = np.array((0, 0))
    w = clf.coef_[0]
    b = clf.intercept_[0]

    for i in range(X.shape[0]):
        posMargin = (np.inner(w, X[i]) + b) + constant
        if (posMargin >= 0):
            above = np.vstack((above, X[i]))
        else:
            below = np.vstack((below, X[i]))
    return above[1:], below[1:]
