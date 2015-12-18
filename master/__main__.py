# -*- coding: utf-8 -*-

import sys
import time

import sklearn.svm as SVC

import DataLoader as dl
import DualSvm as ds


def printLine(size):
    line = ""
    for i in range(size):
        line += "-"
    print(line)


def getClf(clfType):
    if clfType == "dualSvm":
        useFactor = False
        factor = 0.8
        count = 0.1
        cLin = 10
        cGauss = 1
        gamma = 0.3
        searchLin = True
        searchGauss = True
        return ds.DualSvm(cLin, cGauss, gamma, useFactor, factor, count, searchLin, searchGauss, True)
    elif clfType == "linear":
        return SVC.LinearSVC(C=10)
    elif clfType == "gauss":
        return SVC.SVC(kernel="rbf", C=100, gamma=10)


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


def main(args):
    '''
    Usage: e.g. master linear iris
    @param args:
    @return:
    '''
    classifiers = ["gauss", "linear", "dualSvm"]
    data = ["sinus", "iris", "cod-rna", "covtype", "a1a", "w8a", "banana", "ijcnn"]
    _CLASSIFIER = 0
    _DATA = 0

    if len(args) == 3:
        if (args[1] in classifiers):
            _CLASSIFIER = args[1]
        else:
            str = "Classifier not recognized. Did you try one of the avaiable classifiers? (" + classifiers + ")"
            raise (
                ValueError(str
                           ))
        if (args[2] in data):
            _DATA = args[2]
        else:
            str = "Data not recognized. Did you try one of the avaiable datasets? (" + data + ")"
            raise (ValueError(str))

        x, x_test, y, y_test = dl.load_data(_DATA)
        printDataStatistics(_DATA, x, x_test)

        clf = getClf(_CLASSIFIER)

        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        if (_CLASSIFIER == "dualSvm"):
            printLine(20)
            print("Points used for gaussian classifier:", clf._nGauss)
            print("Points used for linear classifier:", clf._nLin)
            print("\n")

        printTimeStatistics(_CLASSIFIER, clf, timeFit, x_test, y_test)
    else:
        print(args)
        raise (ValueError("Invalid number of arguments."))


# test.run_test()
print(len(sys.argv))
print(sys.argv)
main(sys.argv)
