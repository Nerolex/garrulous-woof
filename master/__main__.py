# -*- coding: utf-8 -*-

import sys
import time

import sklearn.svm as SVC
from sklearn.grid_search import GridSearchCV

import DataLoader as dl
import DualSvm as ds


def printLine(file, size):
    line = ""
    for i in range(size):
        line += "-"
    file.write(line)


def getClf(clfType):
    if clfType == "dualSvm":
        # Load config file
        config = open('master/dualsvm.conf', 'r')
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

        return ds.DualSvm(cLin, cGauss, gamma, useFactor, factor, count, searchGauss, searchLin, verbose)
    elif clfType == "linear":
        return SVC.LinearSVC(C=0.0001)
    elif clfType == "gauss":
        return SVC.SVC(kernel="rbf", C=100, gamma=10)


def printTimeStatistics(file, _CLASSIFIER, clf, timeFit, x_test, y_test):
    file.write("\n")
    printLine(file, 20)
    file.write("Time and Error Statistics")
    printLine(file, 20)
    tmp = "\nTime to Fit: " + '{:f}'.format(timeFit) + "s\n"
    file.write(tmp)
    error = 1 - clf.score(x_test, y_test)
    tmp = "Error: " + str(error) + "\n"
    file.write(tmp)
    if _CLASSIFIER == "dualSvm":
        tmp = "gauss: " + '{:f}'.format(clf._timeFitGauss) + "s \t(" + str(round(
            (clf._timeFitGauss / (clf._timeFitLin + clf._timeFitGauss + clf._timeOverhead) * 100), 2)) + "%)\n"
        file.write(tmp)
        tmp = "linear: " + '{:f}'.format(clf._timeFitLin) + "s \t(" + str(round(
            (clf._timeFitLin / (clf._timeFitLin + clf._timeFitGauss + clf._timeOverhead) * 100), 2)) + "%)\n"
        file.write(tmp)
        tmp = "overhead: " + '{:f}'.format(clf._timeOverhead) + "s \t(" + str(round(
            (clf._timeOverhead / (clf._timeFitLin + clf._timeFitGauss + clf._timeOverhead) * 100), 2)) + "%)\n"
        file.write(tmp)


def printDataStatistics(file, _DATA, x, x_test):
    tmp = "Data used: " + _DATA
    file.write(tmp)
    file.write("\n")
    printLine(file, 20)
    file.write("Size of data")
    printLine(file, 20)
    file.write("\n")
    tmp = "Total: " + str(x.shape[0] + x_test.shape[0]) + "\n"
    file.write(tmp)
    tmp = "Train: " + str(x.shape[0]) + "\n"
    file.write(tmp)
    tmp = "Test: " + str(x_test.shape[0]) + "\n"
    file.write(tmp)
    file.write("\n")


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
    _GRIDSEARCH = False

    # Load config file
    output = open('master/dualsvm_result.txt', 'a')
    output.write("\n")
    printLine(output, 20)
    tmp = "Dual Svm run started on " + str(time.asctime(time.localtime(time.time())))
    output.write(tmp)
    printLine(output, 20)

    if len(args) == 3:
        if (args[1] in classifiers):
            _CLASSIFIER = args[1]
        else:
            cl = ""
            for classifier in classifiers:
                cl += classifier + ", "
            tmp = "Classifier not recognized. Did you try one of the avaiable classifiers? (" + cl + ")"
            raise (
                ValueError(tmp
                           ))
        if (args[2] in data):
            _DATA = args[2]
        else:
            da = ""
            for dat in data:
                da = dat + ", "
            tmp = "Data not recognized. Did you try one of the avaiable datasets? (" + da + ")"
            raise (ValueError(tmp))

        x, x_test, y, y_test = dl.load_data(_DATA)
        tmp = "\nClassifier used: " + _CLASSIFIER + "\n"
        output.write(tmp)
        printDataStatistics(output, _DATA, x, x_test)

        clf = getClf(_CLASSIFIER)

        if (_CLASSIFIER == "linear" and _GRIDSEARCH == True):
            # LinSvm gridSearch
            param_grid = [
                {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
            # cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC.LinearSVC(), param_grid=param_grid)
            grid.fit(x, y)

            C = grid.best_params_['C']
            tmp = "Linear C: " + C
            output.write(tmp)
            # clf.set_params({'C': C})

        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        if (_CLASSIFIER == "dualSvm"):
            printLine(output, 20)
            output.write("Point distribution")
            printLine(output, 20)
            output.write("\n")
            tmp = "Points used for gaussian classifier: " + str(clf._nGauss) + " \t(" + str(round(
                (float(clf._nGauss) / float(clf._nGauss + clf._nLin) * 100), 2)) + "%)" + "\n"
            output.write(tmp)
            tmp = "Points used for linear classifier: " + str(clf._nLin) + " \t(" + str(round(
                (float(clf._nLin) / float(clf._nGauss + clf._nLin) * 100), 2)) + "%)" + "\n"
            output.write(tmp)
            output.write("\n")

            printLine(output, 20)
            output.write("Params used")
            printLine(output, 20)
            output.write("\n")
            tmp = "Linear: C: " + str(clf._linSVC.C) + "\n"
            output.write(tmp)
            tmp = "Gaussian: C: " + str(clf._gaussSVC.C) + " gamma: " + str(clf._gaussSVC.gamma) + "\n"
            output.write(tmp)

        printTimeStatistics(output, _CLASSIFIER, clf, timeFit, x_test, y_test)
        output.close()
    else:
        output.write(args)
        raise (ValueError("Invalid number of arguments."))

main(sys.argv)
