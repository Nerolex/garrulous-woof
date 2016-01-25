# -*- coding: utf-8 -*-

import time
import warnings

import sklearn.svm as SVC

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

        return ds.DualSvm(cLin, cGauss, gamma, 0.0, searchGauss, searchLin, verbose)
    elif clfType == "linear":
        return SVC.LinearSVC(C=0.0001)
    elif clfType == "gauss":
        return SVC.SVC(kernel="rbf", C=100, gamma=10)


# region Output methods
def printTimeStatistics(raw_output, _CLASSIFIER, clf, timeFit, x_test, y_test):
    _timeFitLin = clf._timeFitLin  # SS_MSMS
    _timeFitGauss = clf._timeFitGauss  # MM:SS
    _timeFitOver = clf._timeOverhead  # MSMS
    _timeTotal = _timeFitLin + _timeFitGauss + _timeFitOver

    _score = clf.score(x_test, y_test)
    _error = round((1 - _score) * 100, 2)

    _timePredict = clf._timePredict  ## SS_MSMS

    _percentGaussTotal = round((_timeFitGauss / _timeTotal) * 100, 2)
    _percentLinTotal = round((_timeFitLin / _timeTotal * 100), 2)
    _percentOverTotal = round((_timeFitOver / _timeTotal * 100), 2)

    # Bring Time data in readable format
    _timeFit = secondsToHourMin(_timeTotal)
    _timeFitLin = secondsToSecMilsec(_timeFitLin)
    _timeFitGauss = secondsToMinSec(_timeFitGauss)
    _timeFitOver = secondsToMilsec(_timeFitOver)
    _timePredict = secondsToSecMilsec(_timePredict)

    raw_output[7].append(_timeFit + ";")
    raw_output[8].append(_timeFitGauss + "\t(" + str(_percentGaussTotal) + "%)" + ";")
    raw_output[9].append(_timeFitLin + "\t(" + str(_percentLinTotal) + "%)" + ";")
    raw_output[10].append(_timeFitOver + "\t(" + str(_percentOverTotal) + "%)" + ";")
    raw_output[11].append(_timePredict + ";")
    raw_output[12].append(str(_error) + "%;")

    return raw_output


def printDataStatistics(output, _DATA, x, x_test):
    tmp = _DATA + " (Tot: " + str(x.shape[0] + x_test.shape[0]) + " Train: " + str(x.shape[0]) + " Test: " + str(
            x_test.shape[0]) + ")\n"
    output.write(tmp)


def printMiscStatsDualSvm(clf, raw_output):
    gauss_stat = str(clf._nGauss) + " (" + str(
        round((float(clf._nGauss) / float(clf._nGauss + clf._nLin) * 100), 2)) + "%);"
    lin_stat = str(clf._nLin) + " \t(" + str(
        round((float(clf._nLin) / float(clf._nGauss + clf._nLin) * 100), 2)) + "%);"
    dec_margin = str(round(clf._margins[1], 3)) + ";"
    lin_c = toPowerOfTen(clf._linSVC.C) + ";"
    gauss_c = toPowerOfTen(clf._gaussSVC.C) + ";"
    gauss_gamma = toPowerOfTen(clf._gaussSVC.gamma) + ";"
    try:
        n_gaussSVs = str(clf._gaussSVC.n_support_[0] + clf._gaussSVC.n_support_[1]) + ";"
    except AttributeError:
        n_gaussSVs = "0;"

    raw_output[0].append(gauss_stat)
    raw_output[1].append(lin_stat)
    raw_output[2].append(str(dec_margin).replace(".", ","))
    raw_output[3].append(lin_c)
    raw_output[4].append(gauss_c)
    raw_output[5].append(gauss_gamma)
    raw_output[6].append(n_gaussSVs)


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
    printLine(output, 20)
    tmp = "Dual Svm run started on " + str(time.asctime(time.localtime(time.time())))
    output.write(tmp)
    printLine(output, 20)
    output.write("\n")


#endregion

#region Conversion Methods
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


#endregion

def run(x, x_test, y, y_test, count, gridGauss, gridLin, raw_output):
        #Load the classifier
        clf = getClf("dualSvm")
        clf._searchGauss = gridGauss
        clf._searchLin = gridLin
        clf._count = count

        #Notice that the result can be distored by the gridsearch for the dual svm.
        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        printMiscStatsDualSvm(clf, raw_output)
        printTimeStatistics(raw_output, "dualSvm", clf, timeFit, x_test, y_test)


def run_batch(data):
    # Output
    date = str(time.asctime(time.localtime(time.time())))

    raw_output = [["Points gaussian;"],  # 0
                  ["Points linear;"],  # 1
                  ["Dec. margin;"],  # 2
                  ["C linear;"],  # 3
                  ["C gauss;"],  # 4
                  ["gamma gauss;"],  # 5
                  ["number SVs gauss;"],  # 6
                  ["time to fit;"],  # 7
                  ["      gauss;"],  # 8
                  ["      linear;"],  # 9
                  ["      overhead;"],  # 10
                  ["time to predict;"],  # 11
                  ["Error;"]  # 12
                  ]

    # Load the data
    x, x_test, y, y_test = dl.load_data(data)
    gridGauss = False
    gridLin = False
    for j in range(4):  # Smaller steps from 0 to 20: 0, 5, 10, 15
        if j == 0:
            gridLin = True
        run(x, x_test, y, y_test, 0.05 * j, gridGauss, gridLin, raw_output)
        gridLin = False
    for i in range(5):  # Bigger steps from 20 to 100: 20, 40, 60, 80, 100
        if i == 0:
            gridGauss = True
        run(x, x_test, y, y_test, 0.2 * (i + 1), gridGauss, gridLin, raw_output)
        gridGauss = False

    header = data + " " + date
    header = header.replace(" ", "_")
    header = header.replace(":", "_")
    file = 'master/output/' + header + ".csv"
    output = open(file, 'a')
    printHeader(output)
    printDataStatistics(output, data, x, x_test)

    for row in raw_output:
        str_row = ""
        for cell in row:
            str_row += cell
        str_row += "\n"
        output.write(str_row)
    output.close()


# main program
warnings.filterwarnings("ignore", category=DeprecationWarning)
# run_batch("ijcnn")
run_batch("cod-rna")
# run_batch("skin")
#run_batch("covtype")
print("Done!")
