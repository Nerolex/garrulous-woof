# -*- coding: utf-8 -*-
from __future__ import division

import time

import numpy as np
from sklearn.svm import SVC, LinearSVC

import Console

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""


class DualSvm(object):
    def __init__(self, c_lin=0.001, c_gauss=10, gamma=0.01, k=0, search_gauss=False, verbose=True):
        """
        The constructor of the class. Here the important members are initialized.

        :param c_lin:      Penalty parameter C of the error term of the linear support vector machine.
        :param c_gauss:    Penalty parameter C of the error term of gaussian support vector machine
        :param gamma:     Kernel coefficient for the gaussian svm
        :param k:         k has to be in the range [0,1]. It determines which percentage of closest points should be given to the gaussian svm, sorted by their margins.
        :param verbose: Debug parameter for logging events into a file debug.txt.
        :return:          Returns self.

        """

        # Parameters
        self._c_lin = c_lin
        self._c_gauss = c_gauss
        self._gamma = gamma
        self._k = k

        self._n_gauss = -1
        self._n_lin = -1
        self._verbose = verbose

        # Intern objects
        self._lin_svc = LinearSVC(C=self._c_lin)
        self._gauss_svc = SVC(C=self._c_gauss, kernel="rbf", gamma=self._gamma)
        self._magins = [0, 0]

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
        self._lin_svc.set_params(C=value)

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
        self._gauss_svc.set_params(C=value)

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
        self._gauss_svc.set_params(gamma=value)

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

    # endregion
    def fit_lin_svc(self, X, y):
        self._lin_svc.fit(X, y)

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Fits a linear SVC on the given data.
        Afterwards, certain datapoints are selected and given to a gaussian SVC. The selection is dependant on the attribute useFactor of this object.

        :param X: Training vector
        :param y: Target vector relative to X
        :return: Returns self.
        """
        self._fit_called = True

        if self._verbose:
            Console.write("Starting fitting process.\n")
            Console.write("Starting fitting process for linear SVC.")

        time_start_lin = time.time()
        self._lin_svc.fit(X, y)
        self._time_fit_lin = time.time() - time_start_lin

        if self._verbose:
            Console.write("Completed fitting process for linear SVC.")
            Console.write("Sorting points for classifiers.")

        time_start_overhead = time.time()
        x, y, margins = self.get_points_close_to_hyperplane_by_count(X, y, self._k)
        try:
            self._n_gauss = x.shape[0]  # Measure the number of points for gauss classifier:
        except AttributeError:
            self._n_gauss = len(x)
        self._margins = margins
        self._time_overhead = time.time() - time_start_overhead

        if (self._verbose):
            Console.write("Sorting finished.")
            Console.write("Starting fitting process for gaussian SVC.")

        # Measure the number of points for linear classifier:
        self._n_lin = X.shape[0] - self._n_gauss

        time_start_gauss = time.time()
        if self._n_gauss != 0:
            self._gauss_svc.fit(x, y)
        self._time_fit_gauss = time.time() - time_start_gauss

        if self._verbose:
            Console.write("Completed fitting process for gaussian SVC.")
            Console.write("Finished fitting process.\n")

        return self

    def predict(self, X):
        """
        Predicts the labels for the given data vector X. Uses the range _margins defined in the fit()-method to determine which classifier should predict which element in the data vector x.

        :param X: Data vector.
        :return: Vector of predictions.

        """

        time_start = time.time()
        if self._verbose:
            Console.write("Starting predicting.")

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
                Console.write("Finished predicting.")
            return predictions

        if 0.0 < self._margins[1]:  # (2)
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
                Console.write("Finished predicting.")
            return predictions

        if self._margins[1] == -1:  # (3)
            predictions = self._gauss_svc.predict(X)
            self._time_predict = time.time() - time_start
            if self._verbose:
                Console.write("Finished predicting.")
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
