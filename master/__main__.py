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
        # Load config file
        config = open('dualsvm.conf', 'r')
        for line in config:
            split_line = line.split(":")
            if (split_line[0] == "useFactor"):
                useFactor = split_line[1].strip("\n") == "True"
            if (split_line[0] == "factor"):
                factor = float(split_line[1].strip("\n"))
            if (split_line[0] == "count"):
                count = float(split_line[1].strip("\n"))
            if (split_line[0] == "cLin"):
                cLin = float(split_line[1].strip("\n"))
            if (split_line[0] == "cGauss"):
                cGauss = float(split_line[1].strip("\n"))
            if (split_line[0] == "gamma"):
                gamma = float(split_line[1].strip("\n"))
            if (split_line[0] == "searchLin"):
                searchLin = split_line[1].strip("\n") == "True"
            if (split_line[0] == "searchGauss"):
                searchGauss = split_line[1].strip("\n") == "True"
            if (split_line[0] == "verbose"):
                verbose = split_line[1].strip("\n") == "True"
        config.close()

        return ds.DualSvm(cLin, cGauss, gamma, useFactor, factor, count, searchLin, searchGauss, verbose)
    elif clfType == "linear":
        return SVC.LinearSVC(C=10)
    elif clfType == "gauss":
        return SVC.SVC(kernel="rbf", C=100, gamma=10)


def printTimeStatistics(_CLASSIFIER, clf, timeFit, x_test, y_test):
    print "Time taken:"
    printLine(20)
    print "Time to Fit", '{:f}'.format(timeFit), "s"
    print "Calculating score:"
    print  1 - clf.score(x_test, y_test)
    if _CLASSIFIER == "dualSvm":
        print "gauss:", '{:f}'.format(clf._timeFitGauss), "s ", round(
            (clf._timeFitGauss / (clf._timeFitLin + clf._timeFitGauss + clf._timeOverhead) * 100), 2), "%"
        print "linear:", '{:f}'.format(clf._timeFitLin), "s", round(
            (clf._timeFitLin / (clf._timeFitLin + clf._timeFitGauss + clf._timeOverhead) * 100), 2), "%"
        print "overhead:", '{:f}'.format(clf._timeOverhead), "s", round(
            (clf._timeOverhead / (clf._timeFitLin + clf._timeFitGauss + clf._timeOverhead) * 100), 2), "%"


def printDataStatistics(_DATA, x, x_test):
    print "\nData used: ", _DATA
    printLine(20)
    print "Size:"
    print "Total:", x.shape[0] + x_test.shape[0]
    print "Train:", x.shape[0]
    print "Test:", x_test.shape[0]
    print "\n"


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
            cl = ""
            for classifier in classifiers:
                cl += classifier + ", "
            str = "Classifier not recognized. Did you try one of the avaiable classifiers? (" + cl + ")"
            raise (
                ValueError(str
                           ))
        if (args[2] in data):
            _DATA = args[2]
        else:
            da = ""
            for dat in data:
                da = dat + ", "
            str = "Data not recognized. Did you try one of the avaiable datasets? (" + da + ")"
            raise (ValueError(str))

        x, x_test, y, y_test = dl.load_data(_DATA)
        printDataStatistics(_DATA, x, x_test)

        clf = getClf(_CLASSIFIER)

        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        if (_CLASSIFIER == "dualSvm"):
            printLine(20)
            print "Points used for gaussian classifier:", clf._nGauss, " ", round(
                (float(clf._nGauss) / float(clf._nGauss + clf._nLin) * 100), 2), "%"
            print "Points used for linear classifier:", clf._nLin, " ", round(
                (float(clf._nLin) / float(clf._nGauss + clf._nLin) * 100), 2), "%"
            print "\n"

        printTimeStatistics(_CLASSIFIER, clf, timeFit, x_test, y_test)
    else:
        print(args)
        raise (ValueError("Invalid number of arguments."))

main(sys.argv)
