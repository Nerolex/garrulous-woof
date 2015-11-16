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
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    return xx, yy
