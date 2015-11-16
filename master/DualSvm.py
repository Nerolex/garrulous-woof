from __future__ import division

import numpy as np

import LinearSvmHelper as ls

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""


# TODO: Methode implementieren, die Daten anhand Punktemenge anpasst.
# TODO: Predict und Fit uebernehmen

def getPointsCloseToHyperplaneByFactor(clf, X, factor):
    """
    @param clf: Linear Classifier to be used.
    @param X: Array of unlabeled datapoints.
    @param factor: Factor that determines how close the data should be to the hyperplane.
    @return: Arrays of data out of and in the scope defined by the other parameters.
    """
    outer, tmp = ls.hyperplane(clf, X, -factor)
    inner, outer1 = ls.hyperplane(clf, tmp, factor)
    outer = np.vstack((outer, outer1))

    return outer, inner

def getPointsCloseToHyperplaneByCount(clf, X, count):
    # TODO implement this method
    return 0
