# -*- coding: utf-8 -*-
from __future__ import division

import datetime
import time

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
import multiprocessing

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""


class DualSvm(object):
    def __init__(self, c_lin, c_gauss, gamma, k=0, search_gauss=False, search_lin=False,
                 verbose=False):
        """
        The constructor of the class. Here the important members are initialized.

        :param c_lin:      Penalty parameter C of the error term of the linear support vector machine.
        :param c_gauss:    Penalty parameter C of the error term of gaussian support vector machine
        :param gamma:     Kernel coefficient for the gaussian svm
        :param k:         k has to be in the range [0,1]. It determines which percentage of closest points should be given to the gaussian svm, sorted by their margins.
        :param search_gauss: Determines if gridSearch shall be used to determine best params for C and gamma for the gaussian svm.
        :param search_lin: Determines if gridSearch shall be used to determine best params for C for the linear svm.
        :param verbose: Debug parameter for logging events into a file debug.txt.
        :return:          Returns self.

        """

        # Parameters
        self._c_lin = c_lin
        self._c_gauss = c_gauss
        self._gamma = gamma
        self._k = k
        self._search_gauss = search_gauss
        self._search_lin = search_lin

        self._n_gauss = -1
        self._n_lin = -1
        self._verbose = verbose

        # Intern objects
        self._lin_svc = LinearSVC(C=self._c_lin)
        self._gauss_svc = SVC(C=self._c_gauss, kernel="rbf", gamma=self._gamma)

        try:
            self._debug_file = open('master/output/debug.txt', 'a')
        except(Exception):
            self._debug_file = open('output/debug.txt', 'a')

    # region Getters and Setters
    @property
    def c_lin(self):
        """
        The C parameter for the linear SVM.
        :return: C for linear SVM
        """
        return self._c_lin

    @c_lin.setter
    def c_lin(self, value):
        self._c_lin = value
        self._lin_svc(C=value)

    @property
    def c_gauss(self):
        """
        The C parameter for the gauss SVM.
        :return: C for gauss SVM
        """
        return self._c_gauss

    @c_gauss.setter
    def c_gauss(self, value):
        self._c_gauss = value
        self._gauss_svc(C=value)

    @property
    def gamma(self):
        """
        The gamma parameter for the gauss SVM.
        :return: gamma for gauss SVM
        """
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._gauss_svc(gamma=value)

    @property
    def k(self):
        """
        The percentage of points that should be given to the second classifier.

        :return: k
        """
        return self._k

    @property
    def time_fit_lin(self):
        return self._time_fit_lin

    @property
    def time_fit_gauss(self):
        return self._time_fit_gauss

    @property
    def time_overhead(self):
        return self._time_overhead

    @property
    def time_predict(self):
        return self._time_predict

    @k.setter
    def k(self, value):
        self._k = value

    @property
    def n_gauss(self):
        """
        The number of points that were used training the gauss SVM.
        :return: n_gauss
        """
        return self._n_gauss

    @n_gauss.setter
    def n_gauss(self, value):
        self._n_gauss = value

    @property
    def n_lin(self):
        """
        The number of points that were used training the linear SVM.
        :return: n_lin
        """
        return self._n_lin

    @n_lin.setter
    def n_lin(self, value):
        self._n_lin = value

    @property
    def margins(self):
        """
        List with two elements. Defines the range of margins which is used in the predict() method to determine if a point should be given to the gauss svm.
        Values are 0 if all points should be classified by the linear classifier.
        Values are -1 if all points should be classified by the gaussian classifier.

        :return: minMargin, maxMargin
        """
        return self._margins

    @margins.setter
    def margins(self, value):
        self._margins = value

    @property
    def verbose(self):
        """
        Debug parameter. Used to limit the logging level.

        :return: self._verbose
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def lin_svc(self):
        return self._lin_svc

    @property
    def gauss_svc(self):
        return self._gauss_svc

    @property
    def search_gauss(self):
        return self._search_gauss

    @search_gauss.setter
    def search_gauss(self, value):
        self._search_gauss = value

    @property
    def search_lin(self):
        return self._search_lin

    @search_lin.setter
    def search_lin(self, value):
        self._search_lin = value


    # endregion

    def console(self, str):
        """
        Debug function. Logs the timestamp for the given string in an external debug file.

        :param str: String to be logged.
        :return: none
        """
        time_str = "[" + datetime.datetime.now().strftime('%H:%M:%S') + "]: "
        self._debug_file.write(time_str + str + "\n")

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Fits a linear SVC on the given data.
        Afterwards, certain datapoints are selected and given to a gaussian SVC. The selection is dependant on the attribute useFactor of this object.

        :param X: Training vector
        :param y: Target vector relative to X
        :return: Returns self.
        """

        if self._verbose:
            self.console("Starting fitting process.\n")

        # If set to True, this will search for the best C with gridsearch:
        if self._search_lin:
            if self._verbose:
                self.console("Starting Gridsearch for linear SVC.")
            self._c_lin = self.gridsearch_for_linear(X, y)

        if self._verbose:
            self.console("Starting fitting process for linear SVC.")
        time_start_lin = time.time()
        self._lin_svc.fit(X, y)
        self._time_fit_lin = time.time() - time_start_lin
        if self._verbose:
            self.console("Completed fitting process for linear SVC.")

        if self._verbose:
            self.console("Sorting points for classifiers.")
        time_start_overhead = time.time()

        x, y, margins = self.get_points_close_to_hyperplane_by_count(X, y, self._k)

        try:
            self._n_gauss = x.shape[0]  # Measure the number of points for gauss classifier:
        except AttributeError:
            self._n_gauss = len(x)

        self._margins = margins

        self._time_overhead = time.time() - time_start_overhead
        if (self._verbose):
            self.console("Sorting finished.")

        # Measure the number of points for linear classifier:
        self._n_lin = X.shape[0] - self._n_gauss

        # If set to True, this will search for the best C with gridsearch:
        if self._search_gauss and self._n_gauss != 0:
            if self._verbose:
                self.console("Starting gridsearch for gaussian classifier.")
            self._c_gauss, self._gamma = self.gridsearch_for_gauss(x, y)

        if self._verbose:
            self.console("Starting fitting process for gaussian SVC.")

        time_start_gauss = time.time()

        if self._n_gauss != 0:
            self._gauss_svc.fit(x, y)
        self._time_fit_gauss = time.time() - time_start_gauss

        if self._verbose:
            self.console("Completed fitting process for gaussian SVC.")
        if self._verbose:
            self.console("Finished fitting process.\n")

    def predict(self, X):
        """
        Predicts the labels for the given data vector X. Uses the range _margins defined in the fit()-method to determine which classifier should predict which element in the data vector x.

        :param X: Data vector.
        :return: Vector of predictions.

        """

        time_start = time.time()
        if self._verbose:
            self.console("Starting predicting.")

        """
        If-Construct to account for the border cases (all points for one classifier):

        (1) margins = [0, 0]: All points used for the linear SVM.
        (2) e.g. margins = [-0.3, 0.3] Points distributed between both. Standard case.
        (3) margins = [1, -1]: All points used for the gauss SVM. (fit()-Method set margins to -1)
        """

        if self._margins[1] == 0.0:  # (1)
            predictions = self._lin_svc.predict(X)
            self._time_predict = time.time() - time_start
            if self._verbose:
                self.console("Finished predicting.")
            return predictions

        if 0.0 < self.margins[1] < 1.0:  # (2)
            fx = abs(self._lin_svc.decision_function(X)) / np.linalg.norm(self._lin_svc.coef_[0])
            gauss_indices = np.where(np.logical_and(self._margins[0] <= fx, fx < self._margins[1]))
            lin_indices = np.where(np.logical_or(self._margins[0] > fx, fx >= self._margins[1]))
            lin_predictions = self._lin_svc.predict(X[lin_indices])
            gauss_predictions = self._gauss_svc.predict(X[gauss_indices])
            predictions = np.zeros(len(lin_predictions) + len(gauss_predictions))
            predictions[lin_indices] = lin_predictions
            predictions[gauss_indices] = gauss_predictions
            self._time_predict = time.time() - time_start
            if self._verbose:
                self.console("Finished predicting.")
            return predictions

        if self._marings[1] == -1:  # (3)
            predictions = self._gauss_svc.predict(X)
            self._time_predict = time.time() - time_start
            if self._verbose:
                self.console("Finished predicting.")
            return predictions

        # If no condition matched
        raise Exception("Fatal error: Count param")

    def score(self, X, y):
        """
        Method that calculates the score for a given test set X,y.
        :param X: Test vector of datapoints
        :param y: Test vector of labels
        :return: score value
        """
        y_hat = self.predict(X)
        score = 0
        for i in range(y.shape[0]):
            if y[i] == y_hat[i]:
                score += 1
        score /= y.shape[0]
        return score

    def get_points_close_to_hyperplane_by_count(self, X, y, k):
        """
        Helper method for determining the subset of points to be given to the gaussian classifier.

        :param X: Array of unlabeled datapoints.
        :param k: k has to be in the range [0,1]. It determines which percentage of closest points should be given to the gaussian svm, sorted by their margins.
        :return: Returns data vectors x_inner, y_inner and margins. x_inner and y_inner represent the labeled subset which will be given to the gaussian svm. margins is a list with to elements, which represents the interval, in which the gaussian classifier should be used. This is used in the predict()-method.
        """
        time_start = time.time()
        margins = abs(self._lin_svc.decision_function(X)) / np.linalg.norm(self._lin_svc.coef_[0])

        # Calculate the actual number of points to be taken into account
        n = np.ceil(k * X.shape[0])

        # 3 Cases to consider:
        # 1) n or k = 0: All points should be classified by the linear classifier. No points given to the gaussian
        # 2) n or k > 0 and count < 1 standard case
        # 3) count = 1 All points should be classified by the gaussian classifier.

        if n == 0 or k == 0.0:
            x_inner = []
            y_inner = []
            max_margin = 0
        if 0.0 < k < 1.0:
            indices = np.argpartition(margins, n)[:n]  # get the indices of the n smallest elements
            x_inner = X[indices]
            y_inner = y[indices]
            # Keep track of minimal and maximal margins
            max_margin = max(margins[indices])
        if k == 1.0:
            x_inner = X
            y_inner = y
            max_margin = -1

        margins = [-max_margin, max_margin]

        return x_inner, y_inner, margins

    def gridsearch_for_linear(self, X, y):
        """
        Parameter tuning for the linear classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal values of the first grid.

        :param X: Data x
        :param y: Labels y
        :return: Best parameters
        """
        self.console("Linear SVC: Starting coarse gridsearch for gaussian classifier.")
        # LinSvm gridSearch
        c_range = np.logspace(-2, 10, 13, base=10.0)
        param_grid = dict(C=c_range)
        grid = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=4)
        grid.fit(X, y)

        _c = grid.best_params_['C']

        self.console("Linear SVC: Finished coarse gridsearch with params: C: " + str(_c))
        self.console("Linear SVC: Starting fine gridsearch:")

        c_range_2 = np.linspace(_c - 0.2 * _c, _c + 0.2 * _c, num=5)
        param_grid = dict(C=c_range_2)
        grid = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=4)
        grid.fit(X, y)

        _c = grid.best_params_['C']
        self._lin_svc.set_params(C=_c)
        self.console("Linear SVC: Finished fine gridsearch with params: C: " + str(_c))

        return _c

    def gridsearch_for_gauss(self, X, y):
        """
        Parameter tuning for the gauss classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal val

        :param X: Data x
        :param y: Labels y
        :return: Best parameters
        """
        n_cpu = multiprocessing.cpu_count()
        print("Using multiprocessing. Avaible cores: " + str(n_cpu))
        self.console("Gauss SVC: Starting gridsearch for gaussian classifier.")
        c_range = np.logspace(-2, 2, 5, base=10.0)
        gamma_range = np.logspace(-2, 2, 5, base=10.0)
        param_grid = dict(gamma=gamma_range, C=c_range)

        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, n_jobs=n_cpu)
        grid.fit(X, y)
        _c = grid.best_params_['C']
        _gamma = grid.best_params_['gamma']

        print("First search complete. Starting second search...")

        self.console("Gauss SVC: Finished coarse gridsearch with params: C: " + str(_c) + " gamma: " + str(_gamma))
        self.console("Gauss SVC: Starting fine for gaussian classifier.")

        c_range_2 = np.linspace(_c - 0.2 * _c, _c + 0.2 * _c, num=5)
        gamma_range_2 = np.linspace(_c - 0.2 * _gamma, _gamma + 0.2 * _gamma, num=5)
        param_grid = dict(gamma=gamma_range_2, C=c_range_2)
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, n_jobs=n_cpu)
        grid.fit(X, y)

        _c = grid.best_params_['C']
        _gamma = grid.best_params_['gamma']

        self.console("Gauss SVC: Finished fine gridsearch with params: C: " + str(_c) + " gamma: " + str(_gamma))

        self._gauss_svc.set_params(C=_c, gamma=_gamma)
        return _c, _gamma
