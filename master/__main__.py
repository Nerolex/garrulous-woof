# -*- coding: utf-8 -*-

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

        return ds.DualSvm(cLin, cGauss, gamma, False, 0.0, 0.0, searchGauss, searchLin, verbose)
    elif clfType == "linear":
        return SVC.LinearSVC(C=0.0001)
    elif clfType == "gauss":
        return SVC.SVC(kernel="rbf", C=100, gamma=10)


def printTimeStatistics(file, _CLASSIFIER, clf, timeFit, x_test, y_test):
    _timeFit = timeFit  # HH:MM
    _timeFitLin = clf._timeFitLin  # SS_MSMS
    _timeFitGauss = clf._timeFitGauss  # MM:SS
    _timeFitOver = clf._timeOverhead  # MSMS
    _timeTotal = _timeFitLin + _timeFitGauss + _timeFitOver

    _score = clf.score(x_test, y_test)
    _error = round((1 - _score) * 100, 2)

    _percentGaussTotal = round((_timeFitGauss / _timeTotal) * 100, 2)
    _percentLinTotal = round((_timeFitLin / _timeTotal * 100), 2)
    _percentOverTotal = round((_timeFitOver / _timeTotal * 100), 2)

    # Bring Time data in readable format
    _timeFit = secondsToHourMin(_timeFit)
    _timeFitLin = secondsToSecMilsec(_timeFitLin)
    _timeFitGauss = secondsToMinSec(_timeFitGauss)
    _timeFitOver = secondsToMilsec(_timeFitOver)
    
    file.write("\n")
    printLine(file, 20)
    file.write("Time and Error Statistics")
    printLine(file, 20)
    tmp = "\nTime to Fit: " + _timeFit + "\n"
    file.write(tmp)
    tmp = "Error: " + str(_error) + "%\n"
    file.write(tmp)
    if _CLASSIFIER == "dualSvm":
        tmp = "gauss: " + _timeFitGauss + "\t(" + str(_percentGaussTotal) + "%)\n"
        file.write(tmp)
        tmp = "linear: " + _timeFitLin + "\t(" + str(_percentLinTotal) + "%)\n"
        file.write(tmp)
        tmp = "overhead: " + _timeFitOver + "\t(" + str(_percentOverTotal) + "%)\n"
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


def printMiscStatsDualSvm(clf, output):
    printLine(output, 20)
    output.write("Point distribution")
    printLine(output, 20)
    output.write("\n")
    tmp = ""
    if clf._useFactor:
        tmp = "Decision by factor: " + str(clf._factor) + "\n"
    else:
        tmp = "Decision by count: " + str(clf._count) + "\n"
    tmp += "Boundary at: [" + str(round(clf._margins[0], 3)) + "," + str(round(clf._margins[1], 3)) + "]\n"
    output.write(tmp)
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

    tmp = "Linear: C: " + toPowerOfTen(clf._linSVC.C) + "\n"
    output.write(tmp)
    tmp = "Gaussian: C: " + toPowerOfTen(clf._gaussSVC.C) + "; gamma: " + toPowerOfTen(clf._gaussSVC.gamma) + "\n"
    output.write(tmp)


def printGridsearchStatisticsDualSvm(clf, output):
    printLine(output, 20)
    output.write("Gridsearch")
    printLine(output, 20)
    output.write("\n")
    tmp = "Gridsearch for linear?: " + clf._searchLin + "\n"
    output.write(tmp)
    tmp = "Gridsearch for gauss?: " + clf._searchGauss + "\n\n"
    output.write(tmp)


def printHeader(output):
    output.write("\n")
    printLine(output, 20)
    tmp = "Dual Svm run started on " + str(time.asctime(time.localtime(time.time())))
    output.write(tmp)
    printLine(output, 20)


def toPowerOfTen(k):
    return ("%.E" % k)


def secondsToHourMin(s):
    '''
    @param s:
    @return:
    s -> HH:MM
    '''
    _m, _s = divmod(s, 60)
    _h, _m = divmod(_m, 60)
    result = "%dh %02dm" % (_h, _m)
    return result


def secondsToMinSec(s):
    '''
    @param s:
    @return:
    s->MM:SS
    '''
    _m, _s = divmod(s, 60)
    _s = round(_s, 0)
    result = "%dm %02ds" % (_m, _s)
    return result


def secondsToSecMilsec(s):
    '''
    @param s:
    @return:
    s-> SS:MSMS
    '''
    _s = s
    _ms = round(s % 1, 3) * 1000
    result = "%ds %03dms" % (_s, _ms)
    return result


def secondsToMilsec(s):
    '''
    @param s:
    @return:
    s-> MSMS
    '''
    _s = s
    _ms = round(s, 3) * 1000
    result = "%02dms" % (_ms)
    return result


def main(args):
    '''
    Usage: e.g. master linear iris
    @param args:
    @return:
    '''

    #Define valid arguments for console input
    classifiers = ["gauss", "linear", "dualSvm"]
    data = ["sinus", "iris", "cod-rna", "covtype", "a1a", "w8a", "banana", "ijcnn", "skin"]

    #Initial assignments
    _CLASSIFIER = 0
    _DATA = 0
    _GRIDSEARCH = False

    #Output
    output = open('master/dualsvm_result.txt', 'a')
    printHeader(output)

    #Do the following only if enough arguments are given
    if len(args) >= 3:
        if (args[1] in classifiers):  #Set the classifier
            _CLASSIFIER = args[1]
        else:  #Clf was not found
            cl = ""
            for classifier in classifiers:
                cl += classifier + ", "
            tmp = "Classifier not recognized. Did you try one of the avaiable classifiers? (" + cl + ")"
            raise (
                ValueError(tmp
                           ))
        if (args[2] in data):  #Set the data
            _DATA = args[2]
        else:  #data mpt found
            da = ""
            for dat in data:
                da = dat + ", "
            tmp = "Data not recognized. Did you try one of the avaiable datasets? (" + da + ")"
            raise (ValueError(tmp))

        #Load the data
        x, x_test, y, y_test = dl.load_data(_DATA)
        tmp = "\nClassifier used: " + _CLASSIFIER + "\n"
        output.write(tmp)
        printDataStatistics(output, _DATA, x, x_test)

        #Load the classifier
        clf = getClf(_CLASSIFIER)

        #Only for dual svm
        if len(args) == 4 and type(args[3]) == float and _CLASSIFIER == "dualSvm":
            clf._count = args[3]

        #Only for linear svm
        if (_CLASSIFIER == "linear" and _GRIDSEARCH == True):
            if _GRIDSEARCH == True:
                # LinSvm gridSearch
                param_grid = [
                    {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
                grid = GridSearchCV(SVC.LinearSVC(), param_grid=param_grid)
                grid.fit(x, y)
                _C = grid.best_params_['C']
                clf.set_params(C=_C)
            tmp = "Linear C: " + _C
            output.write(tmp)

        # Fit the classifier on the given data and measure the time taken.
        #Notice that the result can be distored by the gridsearch for the dual svm.
        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        if (_CLASSIFIER == "dualSvm"):
            printMiscStatsDualSvm(clf, output)

        printTimeStatistics(output, _CLASSIFIER, clf, timeFit, x_test, y_test)
        output.close()
    else:
        output.write(args)
        raise (ValueError("Invalid number of arguments."))





# main(sys.argv)
main(['', 'dualSvm', 'ijcnn', 0.2])
main(['', 'dualSvm', 'ijcnn', 0.4])
main(['', 'dualSvm', 'ijcnn', 0.6])

main(['', 'dualSvm', 'skin', 0.2])
main(['', 'dualSvm', 'skin', 0.4])
main(['', 'dualSvm', 'skin', 0.6])

main(['', 'dualSvm', 'cod-rna', 0.2])
main(['', 'dualSvm', 'cod-rna', 0.4])
main(['', 'dualSvm', 'cod-rna', 0.6])

print("Done!")
