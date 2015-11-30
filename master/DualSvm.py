from __future__ import division

import collections
import time

import numpy as np
import sklearn.svm as SVC
from scipy.sparse import vstack

import LinearSvmHelper as ls

"""
This module implements a dual SVM approach to accelerate the fitting and prediction process.
"""

class DualSvm(object):
    def __init__(self, cLin, cGauss, gamma, useFactor=True, factor=0, count=0):
        """

        @param cLin:      C parameter for linear svm
        @param cGauss:    C parameter for gaussian svm
        @param gamma:     Gamma parameter for the gaussian svm
        @param useFactor: Boolean that determines if the region for the inner svm should be calculated by a factor or by a number of points.
        @param factor:    Used if useFactor is set to True. Determines a factor which is used to determine close points to the hyperplane.
        @param count:     Used if useFactor is set to False. Determines a count of points which is used to determine close points to the hyperplane.
        @return:          Returns self.
        """

        # Paramters
        self._cLin = cLin
        self._cGauss = cGauss
        self._gamma = gamma
        self._useFactor = useFactor
        self._factor = factor
        self._count = count

        # Intern objects
        self._linSVC = SVC.LinearSVC(C=self._cLin)
        self._gaussSVC = SVC.SVC(C=self._cGauss, kernel="rbf", gamma=self._gamma)


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

    def printTableFormatted(self, title, args):
        print("\t\t\t" + title + "\t\t\t")
        for arg in args:
            line = ""
            for arg2 in arg:
                line = line + (arg2 + "\t")
            print(line)
    def fit(self, X, y):
        """
        Fits a linear SVC on the given data.
        Afterwards, certain datapoints are selected and given to a gaussian SVC. The selection is dependant on the attribute L{useFactor} of this object.


        @param X: Training vector
        @param y: Target vector relative to X
        @return: Returns self.
        """

        timeStartLin = time.time()
        self._linSVC.fit(X, y)
        self._timeFitLin = time.time() - timeStartLin

        timeStartOverhead = time.time()
        # Determine which method to use for finding points for the gaussian SVC
        if (self._useFactor == True):
            timeStart = time.time()
            x, y, margins = self.getPointsCloseToHyperplaneByFactor(X, y, self._factor)
            print("Time Calc points:", (time.time() - timeStart) * 1000)
            self._margins = margins
        else:
            # TODO: this method needs changing and should not be used.
            x, y, margins = self.getPointsCloseToHyperplaneByCount(X, y, self._count)
            self._margins = margins
        self._timeOverhead = time.time() - timeStartOverhead

        timeStartGauss = time.time()
        self._gaussSVC.fit(x, y)
        self._timeFitGauss = time.time() - timeStartGauss

        # printArgs = [["Fit Linear SVM", '{:f}'.format(timeFitLin*1000), "ms"], ["Fit Gaussian SVM", '{:f}'.format(timeFitGauss*1000), "ms"],
        #             ["Overhead", '{:f}'.format(timeOverhead*1000), "ms"]]
        # self.printTableFormatted("Time to fit:", printArgs)

    def predict_obsolete(self, X):
        """

        @param X:
        @return:
        """

        # Prepare arrays for the data
        size = X.shape[1] + 1
        x_lin = np.array([]).reshape(0, size)
        x_gauss = np.array([]).reshape(0, size)

        i = 0  # Keep track of current position in Vector X
        for x in X:
            # Determine where to put the current point
            margin = ls.getMargin(self._linSVC, x)

            if self._margins[0] <= margin <= self._margins[1]:
                tmp = np.append(x, i)
                x_gauss = np.vstack((x_gauss, tmp))
            else:
                tmp = np.append(x, i)
                x_lin = np.vstack((x_lin, tmp))
            i += 1

        # Keep track of which dimensions to slice. Try only to slice the first columns, in which the data lies:
        toSlice = np.arange(0, x_gauss.shape[1] - 1, 1)

        tmp = x_gauss[:, toSlice]  # TODO: Does this work as intended?
        if tmp.size > 0:
            y_gauss = self._gaussSVC.predict(tmp)
        else:
            y_gauss = []
        tmp = x_lin[:, toSlice]
        if tmp.size > 0:
            y_lin = self._linSVC.predict(tmp)
        else:
            y_lin = []

        # Assign predictions to data points
        xy_gauss = np.c_[x_gauss[:, [X.shape[1]]], y_gauss]
        xy_lin = np.c_[x_lin[:, [X.shape[1]]], y_lin]

        predictions = np.vstack((xy_lin, xy_gauss))
        predictions = predictions[predictions[:, 0].argsort()]

        return predictions[:, 1]

    def predict(self, X):
        """

        @param X:
        @return:
        """

        # Prepare dictionaries for the data
        x_lin = {}
        x_gauss = {}

        i = 0  # Keep track of current position in Vector X
        for x in X:
            # Determine where to put the current point
            margin = ls.getMargin(self._linSVC, x)

            if self._margins[0] <= margin <= self._margins[1]:
                tmp = {i: x}
                x_gauss.update(tmp)
                # tmp = np.append(x, i)
                # x_gauss = np.vstack((x_gauss, tmp))
            else:
                tmp = {i: x}
                x_lin.update(tmp)
            i += 1

        if len(
                x_gauss) > 0:  # Check if one of the dictionaries is empty. This could be the case if either all or none points fall into the desired range.
            try:
                tmp = vstack(list(x_gauss.values()))  # build a csr_matrix out of the values of the dictionaries
            except ValueError:
                tmp = np.vstack(list(x_gauss.values()))
            y_gauss = self._gaussSVC.predict(tmp)
        else:
            y_gauss = []
        if len(x_lin) > 0:
            try:
                tmp = vstack(list(x_lin.values()))  # build a csr_matrix out of the values of the dictionaries
            except ValueError:
                tmp = np.vstack(list(x_lin.values()))
            y_lin = self._linSVC.predict(tmp)
        else:
            y_lin = []

        # Debug. Measure the number of points for both classifiers:
        print("Number of points used for gaussian: ", len(x_gauss), len(x_gauss) / (len(x_gauss) + len(x_lin)) * 100,
              "% of total.")
        print("Number of points used for linear: ", len(x_lin), len(x_lin) / (len(x_gauss) + len(x_lin)) * 100,
              "% of total.")

        # Use the key values of the starting dictionaries to remember the origin of the prediction. In this section of the code, the old dictionarie's values (the data points) are overwritten with the values of the predictions.
        i = 0
        for key, item in x_gauss.items():
            x_gauss[key] = y_gauss[i]
            i += 1
        i = 0
        for key, item in x_lin.items():
            x_lin[key] = y_lin[i]
            i += 1

        # Build dictionary of predictions for ordering purposes...
        predictions = {}
        x_gauss.update(x_lin)
        predictions.update(x_gauss)  # ..and start with concatenating the generated dictionaries
        predictions = collections.OrderedDict(
            sorted(predictions.items()))  # ..then start sorting the dictionary by its values, to get the original order
        predictions = np.array(
            list(predictions.values()))  # extract the values out of the OrderedDict with some casting-magic

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
        margins = ls.getMargin(self._linSVC, X)
        indices = np.where(abs(margins) <= factor)
        x_inner = X[indices]
        y_inner = y[indices]
        # Keep track of minimal and maximal margins
        margins = [-factor, factor]

        return x_inner, y_inner, margins

    def getPointsCloseToHyperplaneByCount(self, X, y, count):
        """
        @param clf: Linear Classifier to be used.
        @param X: Array of unlabeled datapoints.
        @param count: Count of points to be taken into consideration
        @return: Array of points defined by the other parameters
        """
        #TODO: This method needs changing! Take a subsample of the points by factor!

        # prevent invalid user input
        if count > X.shape[0]:
            raise Exception('The count must not be higher than the size of X!')

        # Get Margins of all given point
        x_up, x_down, y_up, y_down = ls.marginsSorted(self._linSVC, X, y)

        # Concatenate arrays, so that each point is associated with the correct label
        x_up_labels = np.c_[x_up, y_up]
        x_down_labels = np.c_[x_down, y_down]
        # Sort both arrays
        x_up_labels = x_up_labels[x_up_labels[:,
                      2].argsort()]  # see http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column for details
        x_down_labels = x_down_labels[x_down_labels[:, 2].argsort()]
        result = np.array([]).reshape(0, X.shape[1] + 2)  #TODO: Does this work as intended?

        # Convert both arrays to lists. This is necessary to use the list.pop() method later on.
        x_up_labels = x_up_labels.tolist()
        x_down_labels = x_down_labels.tolist()

        for i in range(count):
            if (i % 2 == 0):  # Take turns in popping elements from either array
                # Check if an element is present. It could be that one of the arrays is empty, while the other is not. In this case, remove the element from the other array.
                if (len(x_up_labels) > 0):
                    tmp = x_up_labels.pop(0)
                else:
                    tmp = x_down_labels.pop()
                result = np.vstack((result, tmp))
            else:
                if (len(x_down_labels) > 0):
                    tmp = x_down_labels.pop()
                else:
                    tmp = x_up_labels.pop()
                result = np.vstack((result, tmp))

        # Keep track of which dimensions to slice. Try only to slice the first columns, in which the data lies:
        toSlice = np.arange(0, X.shape[1], 1)

        x = result[:, toSlice]  # TODO: Does this work as intended?
        y = result[:, result.shape[1] - 1]
        margin = result[:, result.shape[1] -2]
        margins = [min(margin), max(margin)]

        return x, y, margins

    def get_params(self):
        # TODO implement me
        return -1

    def getMargin(self, x):
        return ls.getMargin(self._linSVC, x)
