# -*- coding: utf-8 -*-
from __future__ import division

import datetime
import time

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""


class DualSvm(object):
    def __init__(self, cLin, cGauss, gamma, count=0, searchGauss=False, searchLin=False,
                 verbose=False):
        """

        @param cLin:      C parameter for linear svm
        @param cGauss:    C parameter for gaussian svm
        @param gamma:     Gamma parameter for the gaussian svm
        @param count:     Used if useFactor is set to False. Determines a count of points which is used to determine close points to the hyperplane.
        @param searchGauss: Determines if gridSearch shall be used to determine best params for C and gamma for the gaussian svm.
        @param searchLin: Determines if gridSearch shall be used to determine best params for C for the linear svm.
        @return:          Returns self.
        """

        # Paramters
        self._cLin = cLin
        self._cGauss = cGauss
        self._gamma = gamma
        self._count = count
        self._searchGauss = searchGauss
        self._searchLin = searchLin

        self._nGauss = -1
        self._nLin = -1
        self._verbose = verbose

        # Intern objects
        self._linSVC = SVC(C=self._cLin, kernel="linear")
        self._gaussSVC = SVC(C=self._cGauss, kernel="rbf", gamma=self._gamma)

    # region Getters and Setters
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

    # endregion

    def console(self, str):
        time_str = "[" + datetime.datetime.now().strftime('%H:%M:%S') + "]: "
        print(time_str + str)

    def fit(self, X, y):
        """
        Fits a linear SVC on the given data.
        Afterwards, certain datapoints are selected and given to a gaussian SVC. The selection is dependant on the attribute L{useFactor} of this object.


        @param X: Training vector
        @param y: Target vector relative to X
        @return: Returns self.
        """

        if (self._verbose):
            self.console("Starting fitting process.\n")

        # If set to True, this will search for the best C with gridsearch:
        if (self._searchLin):
            if (self._verbose):
                self.console("Starting Gridsearch for linear SVC.")
            self._cLin = self.gridsearchForLinear(X, y)

        if (self._verbose):
            self.console("Starting fitting process for linear SVC.")
        timeStartLin = time.time()
        self._linSVC.fit(X, y)
        self._timeFitLin = time.time() - timeStartLin
        if (self._verbose):
            self.console("Completed fitting process for linear SVC.")

        if (self._verbose):
            self.console("Sorting points for classifiers.")
        timeStartOverhead = time.time()

        x, y, margins = self.getPointsCloseToHyperplaneByCount(X, y, self._count)

        try:
            self._nGauss = x.shape[0]  # Measure the number of points for gauss classifier:
        except AttributeError:
            self._nGauss = len(x)

        self._margins = margins

        self._timeOverhead = time.time() - timeStartOverhead
        if (self._verbose):
            self.console("Sorting finished.")

        # Measure the number of points for linear classifier:
        self._nLin = X.shape[0] - self._nGauss

        # If set to True, this will search for the best C with gridsearch:
        if (self._searchGauss and self._nGauss != 0):
            if (self._verbose):
                self.console("Starting gridsearch for gaussian classifier.")
            self._cGauss, self._gamma = self.gridsearchForGauss(x, y)

        if (self._verbose):
            self.console("Starting fitting process for gaussian SVC.")

        timeStartGauss = time.time()

        if self._nGauss != 0:
            self._gaussSVC.fit(x, y)
        self._timeFitGauss = time.time() - timeStartGauss

        if (self._verbose):
            self.console("Completed fitting process for gaussian SVC.")
        if (self._verbose):
            self.console("Finished fitting process.\n")

    def predict(self, X):
        """

        @param X:
        @return:
        """

        timeStart = time.time()

        n = np.ceil(self._count * X.shape[0])

        if n == 0 or self._count == 0.0:
            predictions = self._linSVC.predict(X)
            self._timePredict = time.time() - timeStart
            return predictions

        if 0.0 < self._count < 1.0:
            fx = self._linSVC.decision_function(X)
            gaussIndices = np.where(np.logical_and(self._margins[0] <= fx, fx < self._margins[1]))
            linIndices = np.where(np.logical_or(self._margins[0] > fx, fx >= self._margins[1]))
            linPreds = self._linSVC.predict(X[linIndices])
            gaussPreds = self._gaussSVC.predict(X[gaussIndices])
            predictions = np.zeros(len(linPreds) + len(gaussPreds))
            predictions[linIndices] = linPreds
            predictions[gaussIndices] = gaussPreds
            self._timePredict = time.time() - timeStart
            return predictions

        if self._count == 1.0:
            predictions = self._gaussSVC.predict(X)
            self._timePredict = time.time() - timeStart
            return predictions

        # If no condition matched
        raise Exception("Fatal error: Count param")


    def score(self, X, y):
        y_hat = self.predict(X)
        score = 0
        for i in range(y.shape[0]):
            if y[i] == y_hat[i]:
                score += 1
        score /= y.shape[0]
        return score

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
        n = np.ceil(count * X.shape[0])

        # 3 Cases to consider:
        # 1) n or count = 0: All points should be classified by the linear classifier. No points given to the gaussian
        # 2) n or count > 0 and count < 1 standard case
        # 3) count = 1 All points should be classified by the gaussian classifier.

        if n == 0 or count == 0.0:
            x_inner = []
            y_inner = []
            max_margin = 0
        if 0.0 < count < 1.0:
            indices = np.argpartition(margins, n)[:n]  # get the indices of the n smallest elements
            x_inner = X[indices]
            y_inner = y[indices]
            # Keep track of minimal and maximal margins
            max_margin = max(margins[indices])
        if count == 1.0:
            x_inner = X
            y_inner = y
            max_margin = 99999

        margins = [-max_margin, max_margin]

        return x_inner, y_inner, margins

    def gridsearchForLinear(self, X, y):
        # LinSvm gridSearch
        C_range = np.logspace(-2, 10, 13, base=10.0)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(kernel="linear"), param_grid=param_grid, cv=cv)
        grid.fit(X, y)

        _C = grid.best_params_['C']
        self._linSVC.set_params(C=_C)

        self.console("Linear SVC: Finished coarse gridsearch with params: C: " + str(_C))

        return _C

    def gridsearchForGauss(self, X, y):
        C_range = np.logspace(-2, 10, 13, base=10.0)
        gamma_range = np.logspace(-9, 3, 13, base=10.0)
        param_grid = dict(gamma=gamma_range, C=C_range)

        cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv)
        grid.fit(X, y)
        _C = grid.best_params_['C']
        _gamma = grid.best_params_['gamma']

        self.console("Gauss SVC: Finished coarse gridsearch with params: C: " + str(_C) + " gamma: " + str(_gamma))

        C_range_2 = np.linspace(_C - 0.2 * _C, _C + 0.2 * _C, num=5)
        gamma_range_2 = np.linspace(_C - 0.2 * _gamma, _gamma + 0.2 * _gamma, num=5)
        param_grid = dict(gamma=gamma_range_2, C=C_range_2)
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv)
        grid.fit(X, y)

        _C = grid.best_params_['C']
        _gamma = grid.best_params_['gamma']

        self.console("Gauss SVC: Finished fine gridsearch with params: C: " + str(_C) + " gamma: " + str(_gamma))

        self._gaussSVC.set_params(C=_C, gamma=_gamma)
        return _C, _gamma
