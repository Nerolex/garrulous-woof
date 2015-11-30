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
        x, x_test, y, y_test = dl.load_data("sinus")

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
    # ToDO: Anscheinend ist cod-rna linear sehr gut separierbar. Suche nach Datensätzen, bei dem dies nicht der Fall ist.
    x, x_test, y, y_test = dl.load_data("codrna")
    timeStart = time.time()
    clf = getClf("linear")
    clf.fit(x, y)
    print("Time to fit: ", time.time() - timeStart)
    print("Error: ", 1 - clf.score(x_test, y_test))
    # print("Fit gauss:", clf._timeFitGauss)
    # print("Fit lin:", clf._timeFitLin)
    #print("Overhead:", clf._timeOverhead)
