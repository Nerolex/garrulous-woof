from __future__ import division

import numpy as np

import LinearSvmHelper as ls

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""

# TODO: Predict und Fit uebernehmen

class DualSvm(object):
    def __init__(self, useFactor=True):
        """
        @param useFactor: Boolean that determines if the region for the inner svm should be calculated by a factor or by a number of points.
        """
        self.useFactor = useFactor

    def fit(self, X, y):
        # TODO: Cross-Validation?
        """
        @param X: Training vector
        @param y: Target vector relative to X
        @return: Returns self.
        """
        
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
        """

        @param clf: Linear Classifier to be used.
        @param X: Array of unlabeled datapoints.
        @param count: Count of points to be taken into consideration
        @return: Array of points defined by the other parameters
        """

        # prevent invalid user input
        if count > X.shape[0]:
            raise Exception('The count must not be higher than the size of X!')

        # Get Margins of all given point
        above, below = ls.margins(clf, X)

        # Sort both arrays
        above = above[above[:,
                      2].argsort()]  # see http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column for details
        below = below[below[:, 2].argsort()]
        result = np.array((0, 0, 0))

        # Convert both arrays to lists. This is necessary to use the list.pop() method later on.
        above = above.tolist()
        below = below.tolist()

        for i in range(count):
            if (i % 2 == 0):  # Take turns in popping elements from either array
                # Check if an element is present. It could be that one of the arrays is empty, while the other is not. In this case, remove the element from the other array.
                if (len(above) > 0):
                    tmp = above.pop(0)
                else:
                    tmp = below.pop()
                result = np.vstack((result, tmp))
            else:
                if (len(below) > 0):
                    tmp = below.pop()
                else:
                    tmp = above.pop()
                result = np.vstack((result, tmp))

        return result[1:]
