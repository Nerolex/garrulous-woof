# -*- coding: utf-8 -*-
from __future__ import division

import time

import numpy as np
import sklearn.svm as SVC
from sklearn.grid_search import GridSearchCV

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""


# ToDo: Gridsearch - Write to output file

class DualSvm(object):
    def __init__(self, cLin, cGauss, gamma, useFactor=False, factor=0, count=0, searchGauss=False, searchLin=False,
                 verbose=False):
        """

        @param cLin:      C parameter for linear svm
        @param cGauss:    C parameter for gaussian svm
        @param gamma:     Gamma parameter for the gaussian svm
        @param useFactor: Boolean that determines if the region for the inner svm should be calculated by a factor or by a number of points.
        @param factor:    Used if useFactor is set to True. Determines a factor which is used to determine close points to the hyperplane.
        @param count:     Used if useFactor is set to False. Determines a count of points which is used to determine close points to the hyperplane.
        @param searchGauss: Determines if gridSearch shall be used to determine best params for C and gamma for the gaussian svm.
        @param searchLin: Determines if gridSearch shall be used to determine best params for C for the linear svm.
        @return:          Returns self.
        """

        # Paramters
        self._cLin = cLin
        self._cGauss = cGauss
        self._gamma = gamma
        self._useFactor = useFactor
        self._factor = factor
        self._count = count
        self._searchGauss = searchGauss
        self._searchLin = searchLin

        self._nGauss = -1
        self._nLin = -1
        self._verbose = verbose

        # Intern objects
        self._linSVC = SVC.LinearSVC(C=self._cLin)
        self._gaussSVC = SVC.SVC(C=self._cGauss, kernel="rbf", gamma=self._gamma)

    # region Getters and Setters
    @property
    def useFactor(self):
        return self._useFactor

    @useFactor.setter
    def useFactor(self, value):
        self._useFactor = value

    @property
    def cLin(self):
        return self._cLin

    @cLin.setter
    def cLin(self, value):
        self._cLin = value
        self._linSVC(C=value)

    @property
    def cGauss(self):
        return self._cGauss

    @cGauss.setter
    def cGauss(self, value):
        self._cGauss = value
        self._gaussSVC(C=value)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._gaussSVC(gamma=value)

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, value):
        self._factor = value

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value

    @property
    def nGauss(self):
        return self._nGauss

    @nGauss.setter
    def nGauss(self, value):
        self._nGauss = value

    @property
    def nLin(self):
        return self._nLin

    @nLin.setter
    def nLin(self, value):
        self._nLin = value

    #endregion

    def fit(self, X, y):
        """
        Fits a linear SVC on the given data.
        Afterwards, certain datapoints are selected and given to a gaussian SVC. The selection is dependant on the attribute L{useFactor} of this object.


        @param X: Training vector
        @param y: Target vector relative to X
        @return: Returns self.
        """

        if (self._verbose):
            print("Starting fitting process.\n")

        # If set to True, this will search for the best C with gridsearch:
        if (self._searchLin):
            if (self._verbose):
                print("Starting Gridsearch for linear SVC.")
            self._cLin = self.gridsearchForLinear(X, y)

        if (self._verbose):
            print("\t Starting fitting process for linear SVC.")
        timeStartLin = time.time()
        self._linSVC.fit(X, y)
        self._timeFitLin = time.time() - timeStartLin
        if (self._verbose):
            print("\t Completed fitting process for linear SVC.")

        if (self._verbose):
            print(" \t Sorting points for classifiers.")
        timeStartOverhead = time.time()
        # Determine which method to use for finding points for the gaussian SVC
        if (self._useFactor == True):
            timeStart = time.time()
            x, y, margins = self.getPointsCloseToHyperplaneByFactor(X, y, self._factor)
            self._nGauss = x.shape[0]  # Measure the number of points for gauss classifier:
            print "\t Sorting points took ", round(((time.time() - timeStart) * 1000), 2), "s"
            self._margins = margins
        else:
            x, y, margins = self.getPointsCloseToHyperplaneByCount(X, y, self._count)
            self._nGauss = x.shape[0]  # Measure the number of points for gauss classifier:
            self._margins = margins
        self._timeOverhead = time.time() - timeStartOverhead
        if (self._verbose):
            print("\t Sorting finished.")

        # Measure the number of points for linear classifier:
        self._nLin = X.shape[0] - self._nGauss

        # If set to True, this will search for the best C with gridsearch:
        if (self._searchGauss):
            if (self._verbose):
                print("\t Starting gridsearch for gaussian classifier.")
            self._cGauss, self._gamma = self.gridsearchForGauss(x, y)

        if (self._verbose):
            print("\t Starting fitting process for gaussian SVC.")
        timeStartGauss = time.time()
        self._gaussSVC.fit(x, y)
        self._timeFitGauss = time.time() - timeStartGauss
        if (self._verbose):
            print("\t Completed fitting process for gaussian SVC.")
        if (self._verbose):
            print("\nFinished fitting process.\n")

    def predict(self, X):
        """

        @param X:
        @return:
        """

        predictions = []

        if (self._verbose):
            print "Starting prediction process."

        i = 0  # Keep track of current position in Vector X
        lastVal = -1
        for x in X:
            if (self._verbose):
                curPos = round((i / X.shape[0]) * 100, 2)
                if (curPos * 1000 % 1000 == 0) and lastVal != curPos:
                    lastVal = curPos
                    print int(curPos), "%"

            # Determine where to put the current point
            margin = self._linSVC.decision_function(x)

            if self._margins[0] <= margin <= self._margins[1]:
                predictions.append(self._gaussSVC.predict(x))
            else:
                predictions.append(self._linSVC.predict(x))
            i += 1

        if (self._verbose):
            print "Predicting finished."

        return predictions

    def score(self, X, y):
        y_hat = self.predict(X)
        score = 0
        for i in range(y.shape[0]):
            if y[i] == y_hat[i]:
                score += 1
        score /= y.shape[0]
        return score

    def getPointsCloseToHyperplaneByFactor(self, X, y, factor):
        """
        @param clf: Linear Classifier to be used.
        @param X: Array of unlabeled datapoints.
        @param factor: Factor that determines how close the data should be to the hyperplane.
        @return: Returns data and labels within and without the calculated regions.
        """

        timeStart = time.time()
        margins = abs(self._linSVC.decision_function(X))

        indices = np.where(margins <= factor)
        try:
            x_inner = X[indices]
            y_inner = y[indices]
        except Exception:
            print("\tFactor chosen too low. No data matched this margin. Using getPointsByCount instead..")
            return self.getPointsCloseToHyperplaneByCount(X, y, self._count)
        # Keep track of minimal and maximal margins
        margins = [-factor, factor]

        return x_inner, y_inner, margins

    def getPointsCloseToHyperplaneByCount(self, X, y, count):
        """
        @param clf: Linear Classifier to be used.
        @param X: Array of unlabeled datapoints.
        @param factor: Factor that determines how close the data should be to the hyperplane.
        @return: Returns data and labels within and without the calculated regions.
        """
        timeStart = time.time()
        margins = abs(self._linSVC.decision_function(X))

        # Calculate the actual number of points to be taken into account
        n = count * X.shape[0]
        indices = np.argpartition(margins, n)[:n]  # get the indices of the n smallest elements
        x_inner = X[indices]
        y_inner = y[indices]
        # Keep track of minimal and maximal margins
        maxMargin = max(margins[indices])
        margins = [-maxMargin, maxMargin]

        return x_inner, y_inner, margins

    def gridsearchForLinear(self, X, y):
        # LinSvm gridSearch
        param_grid = [
            {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
        # cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC.LinearSVC(), param_grid=param_grid)
        grid.fit(X, y)

        _C = grid.best_params_['C']
        self._linSVC.set_params(C=_C)

        return _C

    def gridsearchForGauss(self, X, y):
        param_grid = [
            {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
             'kernel': ['rbf']}, ]
        # param_grid = [
        #    {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001],
        #     'kernel': ['rbf']}, ]
        # cv = StratifiedShuffleSplit(y, n_iter=3, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC.SVC(), param_grid=param_grid)
        grid.fit(X, y)

        _C = grid.best_params_['C']
        _gamma = grid.best_params_['gamma']

        self._gaussSVC.set_params(C=_C, gamma=_gamma)
        return _C, _gamma
