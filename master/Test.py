from __future__ import division

import time

import matplotlib.pyplot as plt
import sklearn.svm as SVC

import DataLoader as dl
import DualSvm as ds
import LinearSvmHelper as ls
import PlotHelper as pl

"""
This is a testing module.
"""

# TODO GridSearch ausprobieren -> Interface!


def plot_sinus(x, y, clf, factor):
    pl.contour(clf, [-2, 2], [-2, 2], 0.02)

    xx, yy = ls.getHyperplane(clf._linSVC)
    yy_up, yy_down = ls.getMarginPlanes(clf._linSVC, factor)

    plt.plot(xx, yy_up, "k--")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.show()


def getClf(clfType):
    if clfType == "dualSvm":
        factor = 0.1
        count = 100
        cLin = 0.01
        cGauss = 100
        gamma = 10
        return ds.DualSvm(cLin, cGauss, gamma, True, factor, count)
    elif clfType == "linear":
        return SVC.LinearSVC(C=0.01)
    elif clfType == "gauss":
        return SVC.SVC(kernel="rbf", C=100, gamma=10)


def run_test():
    _CLASSIFIER = "dualSvm"
    _DATA = "covtype"

    x, x_test, y, y_test = dl.load_data(_DATA)
    clf = getClf(_CLASSIFIER)

    timeStart = time.time()
    clf.fit(x, y)
    timeFit = time.time() - timeStart

    print("Time to Fit", '{:f}'.format(timeFit), "s")
    print("Error:", 1 - clf.score(x_test, y_test))
    if _CLASSIFIER == "dualSvm":
        print("\t gauss:", '{:f}'.format(clf._timeFitGauss), "s")
        print("\t linear:", '{:f}'.format(clf._timeFitLin), "s")
        print("\t overhead:", '{:f}'.format(clf._timeOverhead), "s")
