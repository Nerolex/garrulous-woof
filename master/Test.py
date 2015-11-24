from __future__ import division

import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.cross_validation as cv
import sklearn.datasets as da
import sklearn.svm as SVC

import DataDistributions as dd
import DualSvm as ds
import LinearSvmHelper as ls
import PlotHelper as pl

"""
This is a test module.
"""


# TODO GridSearch ausprobieren -> Interface!

def load_data(dataType):
    if dataType == "sinus":
        x, x_test, y, y_test = load_sinus()
    elif dataType == "iris":
        x, x_test, y, y_test = load_iris()
    return x, x_test, y, y_test


def load_iris():
    data = da.load_iris()
    y = data.target
    x = data.data
    x, x_test, y, y_test = cv.train_test_split(data.data, data.target, test_size=1 / 3)
    y = np.where(y == 1, -1, 1)
    y_test = np.where(y_test == 1, -1, 1)
    return x, x_test, y, y_test


def load_sinus():
    size = 5000
    location = 0.0
    scale = 0.5
    amplitude = 0.3
    freq = 3.5
    x, y = dd.generateSinusCluster(size, location, scale, amplitude, freq)
    x_test, y_test = dd.generateSinusCluster(size * 3, location, scale, amplitude, freq)
    return x, x_test, y, y_test


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
        factor = 0.7
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
    meanError = 0
    meanTime = 0
    numberRuns = 10
    usedClassifier = "linear"
    meanLin = 0
    meanGauss = 0
    meanOverhead = 0

    for i in range(numberRuns):
        x, x_test, y, y_test = load_data("sinus")

        clf = getClf(usedClassifier)

        timeStart = time.time()
        clf.fit(x, y)
        meanTime += (time.time() - timeStart) * 1000

        if (usedClassifier == "dualSvm"):
            meanGauss += clf._timeFitGauss * 1000
            meanLin += clf._timeFitLin * 1000
            meanOverhead += clf._timeOverhead * 1000

        meanError += (1 - clf.score(x, y))

    meanError /= numberRuns
    meanTime /= numberRuns

    if usedClassifier == "dualSvm":
        meanGauss /= numberRuns
        meanLin /= numberRuns
        meanOverhead /= numberRuns

    test = clf.get_params()
    print(usedClassifier)
    print("Mean Time to Fit", '{:f}'.format(meanTime), "ms")
    if usedClassifier == "dualSvm":
        print("\t gauss:", '{:f}'.format(meanGauss), "ms")
        print("\t linear:", '{:f}'.format(meanLin), "ms")
        print("\t overhead:", '{:f}'.format(meanOverhead), "ms")
    print("Mean Error: ", '{:f}'.format(meanError))
    # plot_sinus(x, y, dualSvm, factor)


def run_test1():
    x, x_test, y, y_test = load_data("sinus")
    clf = getClf("dualSvm")
    clf.fit(x, y)

    plot_sinus(x, y, clf, 0.7)


def run_test2():
    a = np.zeros((500, 1))
    a = np.where(a == 0, 5, 0)
    b = np.zeros((500, 1))
    b = np.where(b == 5, 1, 0)
    c = np.zeros((500, 1))
    c = np.where(c == 0, 5, 0)
    d = np.vstack((a, b, c))

    timeStart1 = time.time()
    e = (np.where(d == 0))[0]
    print("Time Where d==0:", (time.time() - timeStart1) * 1000)
    # print(e)
    timeStart2 = time.time()
    f = d[e]
    print("Time d[e]", (time.time() - timeStart2) * 1000)
    # print(d[e])
