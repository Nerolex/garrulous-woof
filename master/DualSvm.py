# -*- coding: utf-8 -*-
from __future__ import division

import datetime
import time

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""


class DualSvm(object):
    def __init__(self, cLin, cGauss, gamma, k=0, searchGauss=False, searchLin=False,
                 verbose=False):
        """
        The constructor of the class. Here the important members are initialized.

        :param cLin:      C parameter for linear svm
        :param cGauss:    C parameter for gaussian svm
        :param gamma:     Gamma parameter for the gaussian svm
        :param k:         k has to be in the range [0,1]. It determines which percentage of closest points should be given to the gaussian svm, sorted by their margins.
        :param searchGauss: Determines if gridSearch shall be used to determine best params for C and gamma for the gaussian svm.
        :param searchLin: Determines if gridSearch shall be used to determine best params for C for the linear svm.
        :param verbose: Debug parameter for logging events into a file debug.txt.
        :return:          Returns self.

        """

        # Paramters
        self._cLin = cLin
        self._cGauss = cGauss
        self._gamma = gamma
        self._k = k
        self._searchGauss = searchGauss
        self._searchLin = searchLin

        self._nGauss = -1
        self._nLin = -1
        self._verbose = verbose

        # Intern objects
        self._linSVC = LinearSVC(C=self._cLin)
        self._gaussSVC = SVC(C=self._cGauss, kernel="rbf", gamma=self._gamma)

        self._debugFile = open('master/output/debug.txt', 'a')

    # region Getters and Setters
    @property
    def cLin(self):
        """
        The C parameter for the linear SVM.
        @return: C for linear SVM
        """
        return self._cLin

    @cLin.setter
    def cLin(self, value):
        self._cLin = value
        self._linSVC(C=value)

    @property
    def cGauss(self):
        """
        The C parameter for the gauss SVM.
        @return: C for gauss SVM
        """
        return self._cGauss

    @cGauss.setter
    def cGauss(self, value):
        self._cGauss = value
        self._gaussSVC(C=value)

    @property
    def gamma(self):
        """
        The gamma parameter for the gauss SVM.
        @return: gamma for gauss SVM
        """
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._gaussSVC(gamma=value)

    @property
    def count(self):
        """
        The percentage of points that should be given to the second classifier.

        @return: k
        """
        return self._k

    @count.setter
    def count(self, value):
        self._k = value

    @property
    def nGauss(self):
        """
        The number of points that were used training the gauss SVM.
        @return: nGauss
        """
        return self._nGauss

    @nGauss.setter
    def nGauss(self, value):
        self._nGauss = value

    @property
    def nLin(self):
        """
        The number of points that were used training the linear SVM.
        @return: nLin
        """
        return self._nLin

    @nLin.setter
    def nLin(self, value):
        self._nLin = value

    @property
    def margins(self):
        """
        List with two elements. Defines the range of margins which is used in the predict() method to determine if a point should be given to the gauss svm.
        Values are 0 if all points should be classified by the linear classifier.
        Values are -1 if all points should be classified by the gaussian classifier.
        @return: minMargin, maxMargin
        """
        return self._margins

    @margins.setter
    def margins(self, value):
        self._margins = value

    @property
    def verbose(self):
        """
        Debug parameter. Used to limit the logging level.
        @return: self._verbose
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    # endregion

    def console(self, str):
        """
        Debug function. Logs the timestamp for the given string in an external debug file.

        :param str: String to be logged.
        :return: none
        """
        time_str = "[" + datetime.datetime.now().strftime('%H:%M:%S') + "]: "
        self._debugFile.write(time_str + str + "\n")

    def fit(self, X, y):
        """
        Fits a linear SVC on the given data.
        Afterwards, certain datapoints are selected and given to a gaussian SVC. The selection is dependant on the attribute useFactor of this object.

        :param X: Training vector
        :param y: Target vector relative to X
        :return: Returns self.
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

        x, y, margins = self.getPointsCloseToHyperplaneByCount(X, y, self._k)

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
        Predicts the labels for the given data vector X. Uses the range _margins defined in the fit()-method to determine which classifier should predict which element in the data vector x.

        :param X: Data vector.
        :return: Vector of predictions.

        """


        timeStart = time.time()
        if self._verbose:
            self.console("Starting predicting.")

        """
        If-Construct to account for the border cases (all points for one classifier):

        (1) margins = [0, 0]: All points used for the linear SVM.
        (2) e.g. margins = [-0.3, 0.3] Points distributed between both. Standard case.
        (3) margins = [1, -1]: All points used for the gauss SVM. (fit()-Method set margins to -1)
        """

        if self._margins[1] == 0.0: #(1)
            predictions = self._linSVC.predict(X)
            self._timePredict = time.time() - timeStart
            if self._verbose:
                self.console("Finished predicting.")
            return predictions

        if 0.0 < self.margins[1] < 1.0: #(2)
            fx = abs(self._linSVC.decision_function(X)) / np.linalg.norm(self._linSVC.coef_[0])
            gaussIndices = np.where(np.logical_and(self._margins[0] <= fx, fx < self._margins[1]))
            linIndices = np.where(np.logical_or(self._margins[0] > fx, fx >= self._margins[1]))
            linPreds = self._linSVC.predict(X[linIndices])
            gaussPreds = self._gaussSVC.predict(X[gaussIndices])
            predictions = np.zeros(len(linPreds) + len(gaussPreds))
            predictions[linIndices] = linPreds
            predictions[gaussIndices] = gaussPreds
            self._timePredict = time.time() - timeStart
            if self._verbose:
                self.console("Finished predicting.")
            return predictions

        if self._marings[1] == -1: #(3)
            predictions = self._gaussSVC.predict(X)
            self._timePredict = time.time() - timeStart
            if self._verbose:
                self.console("Finished predicting.")
            return predictions

        # If no condition matched
        raise Exception("Fatal error: Count param")


    def score(self, X, y):
        """
        Method that calculates the score for a given test set X,y.
        @param X: Test vector of datapoints
        @param y: Test vector of labels
        @return: score value
        """
        y_hat = self.predict(X)
        score = 0
        for i in range(y.shape[0]):
            if y[i] == y_hat[i]:
                score += 1
        score /= y.shape[0]
        return score

    def getPointsCloseToHyperplaneByCount(self, X, y, count):
        """
        Helper method for determining the subset of points to be given to the gaussian classifier.

        :param X: Array of unlabeled datapoints.
        :param k: k has to be in the range [0,1]. It determines which percentage of closest points should be given to the gaussian svm, sorted by their margins.
        :return: Returns data vectors x_inner, y_inner and margins. x_inner and y_inner represent the labeled subset which will be given to the gaussian svm. margins is a list with to elements, which represents the interval, in which the gaussian classifier should be used. This is used in the predict()-method.
        """
        timeStart = time.time()
        margins = abs(self._linSVC.decision_function(X)) / np.linalg.norm(self._linSVC.coef_[0])

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
            max_margin = -1

        margins = [-max_margin, max_margin]

        return x_inner, y_inner, margins

    def gridsearchForLinear(self, X, y):
        """
        Parameter tuning for the linear classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal values of the first grid.

        @param X: Data x
        @param y: Labels y
        @return: Best parameters
        """
        self.console("Linear SVC: Starting coarse gridsearch for gaussian classifier.")
        # LinSvm gridSearch
        C_range = np.logspace(-2, 10, 13, base=10.0)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(LinearSVC(), param_grid=param_grid, cv=cv, n_jobs=4)
        grid.fit(X, y)

        _C = grid.best_params_['C']

        self.console("Linear SVC: Finished coarse gridsearch with params: C: " + str(_C))
        self.console("Linear SVC: Starting fine gridsearch:")

        C_range_2 = np.linspace(_C - 0.2 * _C, _C + 0.2 * _C, num=5)
        param_grid = dict(C=C_range_2)
        grid = GridSearchCV(LinearSVC(), param_grid=param_grid, cv=cv, n_jobs=4)
        grid.fit(X, y)

        _C = grid.best_params_['C']
        self._linSVC.set_params(C=_C)
        self.console("Linear SVC: Finished fine gridsearch with params: C: " + str(_C))

        return _C

    def gridsearchForGauss(self, X, y):
        """
        Parameter tuning for the gauss classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal val

        @param X: Data x
        @param y: Labels y
        @return: Best parameters
        """
        self.console("Gauss SVC: Starting gridsearch for gaussian classifier.")
        C_range = np.logspace(-2, 10, 13, base=10.0)
        gamma_range = np.logspace(-9, 3, 13, base=10.0)
        param_grid = dict(gamma=gamma_range, C=C_range)

        cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, n_jobs=4)
        grid.fit(X, y)
        _C = grid.best_params_['C']
        _gamma = grid.best_params_['gamma']

        self.console("Gauss SVC: Finished coarse gridsearch with params: C: " + str(_C) + " gamma: " + str(_gamma))
        self.console("Gauss SVC: Starting fine for gaussian classifier.")

        C_range_2 = np.linspace(_C - 0.2 * _C, _C + 0.2 * _C, num=5)
        gamma_range_2 = np.linspace(_C - 0.2 * _gamma, _gamma + 0.2 * _gamma, num=5)
        param_grid = dict(gamma=gamma_range_2, C=C_range_2)
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, n_jobs=4)
        grid.fit(X, y)

        _C = grid.best_params_['C']
        _gamma = grid.best_params_['gamma']

        self.console("Gauss SVC: Finished fine gridsearch with params: C: " + str(_C) + " gamma: " + str(_gamma))

        self._gaussSVC.set_params(C=_C, gamma=_gamma)
        return _C, _gamma
