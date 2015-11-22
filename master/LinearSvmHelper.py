from __future__ import division

import time

import numpy as np

"""
This is a helper module. It defines useful functions for working with linear SVMs.
"""

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
    xx = np.linspace(-2, 2)
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
    xx = np.linspace(-2, 2)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    yy_down = yy + factor * margin
    yy_up = yy - factor * margin

    return yy_down, yy_up


def hyperplane(clf, X, y, constant):
    """
    Function that classifies points with the hyperplane of the given classifier. A constant is used to move the hyperplane.

    @type clf: Classifier
    @param clf: Classifier to be used. LinearSVC expected.
    @type X: Array
    @param X: Array of Datapoints to be classified.
    @param y: Array of correct labels.
    @type constant: float
    @param constant: Constant that is used to move the hyperplane.

    @rtype: Array
    @return: Returns 4 Arrays: x_up, y_up, x_down, y_down. Points and Labels above and below the hyperplane.
    """
    timeStart = time.time()
    x_up = np.array([]).reshape(0, X.shape[1])
    y_up = np.array([]).reshape(0)
    x_down = np.array([]).reshape(0, X.shape[1])
    y_down = np.array([]).reshape(0)
    print("Ls.hyperplane Array reshaping took", (time.time() - timeStart) * 1000, "ms")

    w = clf.coef_[0]
    b = clf.intercept_[0]

    timeStart = time.time()
    for i in range(X.shape[0]):
        posMargin = (np.inner(w, X[i]) + b) + constant
        if (posMargin >= 0):
            x_up = np.vstack((x_up, X[i]))
            y_up = np.append(y_up, y[i])
        else:
            x_down = np.vstack((x_down, X[i]))
            y_down = np.append(y_down, y[i])
    print("Ls.hyperplane For looptook", (time.time() - timeStart) * 1000, "ms")
    return x_up, x_down, y_up, y_down


def getMargin(clf, X):
    """
    Returns the function value of the affine linear function provided by the linear classifier clf.

    @param clf: A Linear SVM.
    @param X: Datapoints.
    @return:
    """

    w = clf.coef_[0]
    b = clf.intercept_[0]

    margin = np.inner(w, X) + b

    return margin


def margins(clf, X, y):
    """
    Function that calculates the margins of the given Points and the given classifier. Returns two arrays: One with positive distances (above hyperplane) and one with negative distances (below hyperplane).
    @param clf: Classifier to be used. LinearSVC expected.
    @param X: Array of datapoints to be classified.
    @param y: Array of labels.
    @return: Returns 4 Arrays: x_up, y_up, x_down, y_down. Points and Labels above and below the hyperplane.
    """

    x_up = np.array([]).reshape(0, X.shape[1] + 1)
    x_down = np.array([]).reshape(0, X.shape[1] + 1)
    y_up = np.array([]).reshape(0, 1)
    y_down = np.array([]).reshape(0, 1)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    for i in range(X.shape[0]):
        margin = (np.inner(w, X[i]) + b)
        tmp = np.append(X[i], [margin])
        if (margin >= 0):
            x_up = np.vstack((x_up, tmp))
            y_up = np.append(y_up, y[i])
        else:
            x_down = np.vstack((x_down, tmp))
            y_down = np.vstack((y_down, y[i]))
    return x_up, x_down, y_up, y_down
