# -*- coding: utf-8 -*-
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


def printLine(size):
    line = ""
    for i in range(size):
        line += "-"
    print(line)

def getClf(clfType):
    if clfType == "dualSvm":
        useFactor = False
        factor = 0.7
        count = 0.1
        cLin = 10
        cGauss = 1
        gamma = 0.3
        searchLin = True
        searchGauss = True
        return ds.DualSvm(cLin, cGauss, gamma, useFactor, factor, count, searchLin, searchGauss)
    elif clfType == "linear":
        return SVC.LinearSVC(C=10)
    elif clfType == "gauss":
        return SVC.SVC(kernel="rbf", C=100, gamma=10)


def run_test():
    _CLASSIFIER = "dualSvm"
    _DATA = "sinus"

    x, x_test, y, y_test = dl.load_data(_DATA)
    printDataStatistics(_DATA, x, x_test)

    clf = getClf(_CLASSIFIER)

    timeStart = time.time()
    clf.fit(x, y)
    timeFit = time.time() - timeStart

    if (_CLASSIFIER == "dualSvm"):
        printLine(20)
        # Warum wird hier ein Fehler geworfen?
        print("Points used for gaussian classifier:", clf._nGauss)
        print("Points used for linear classifier:", clf._nLin)
        print("\n")


    printTimeStatistics(_CLASSIFIER, clf, timeFit, x_test, y_test)

    plot_sinus(x, y, clf, clf._factor)

def printTimeStatistics(_CLASSIFIER, clf, timeFit, x_test, y_test):
    print("Time taken:")
    printLine(20)
    print("Time to Fit", '{:f}'.format(timeFit), "s")
    print("Error:", 1 - clf.score(x_test, y_test))
    if _CLASSIFIER == "dualSvm":
        print("\t gauss:", '{:f}'.format(clf._timeFitGauss), "s")
        print("\t linear:", '{:f}'.format(clf._timeFitLin), "s")
        print("\t overhead:", '{:f}'.format(clf._timeOverhead), "s")


def printDataStatistics(_DATA, x, x_test):
    print("Data used: ", _DATA)
    printLine(20)
    print("Size:")
    print("Total:\t", x.shape[0] + x_test.shape[0])
    print("Train:\t", x.shape[0])
    print("Test:\t", x_test.shape[0])
    print("\n")


run_test()
