# -*- coding: utf-8 -*-

import time
import warnings

import Console
import Conversions
import DataLoader
import DualSvm as ds


def appendTimeStatistics(raw_output, _CLASSIFIER, clf, timeFit, x_test, y_test):
    _timeFitLin = clf.time_fit_lin  # SS_MSMS
    _timeFitGauss = clf.time_fit_gauss  # MM:SS
    _timeFitOver = clf.time_overhead  # MSMS
    _timeTotal = _timeFitLin + _timeFitGauss + _timeFitOver

    _score = clf.score(x_test, y_test)
    _error = round((1 - _score) * 100, 2)

    _timePredict = clf.time_predict  ## SS_MSMS

    _percentGaussTotal = round((_timeFitGauss / _timeTotal) * 100, 2)
    _percentLinTotal = round((_timeFitLin / _timeTotal * 100), 2)
    _percentOverTotal = round((_timeFitOver / _timeTotal * 100), 2)

    # Bring Time data in readable format
    _timeFit = Conversions.secondsToHourMin(_timeTotal)
    _timeFitLin = Conversions.secondsToSecMilsec(_timeFitLin)
    _timeFitGauss = Conversions.secondsToMinSec(_timeFitGauss)
    _timeFitOver = Conversions.secondsToMilsec(_timeFitOver)
    _timePredict = Conversions.secondsToSecMilsec(_timePredict)

    raw_output[7].append(_timeFit + ";")
    raw_output[8].append(_timeFitGauss + "\t(" + str(_percentGaussTotal) + "%)" + ";")
    raw_output[9].append(_timeFitLin + "\t(" + str(_percentLinTotal) + "%)" + ";")
    raw_output[10].append(_timeFitOver + "\t(" + str(_percentOverTotal) + "%)" + ";")
    raw_output[11].append(_timePredict + ";")
    raw_output[12].append(str(_error) + "%;")

    return raw_output


def appendMiscStatsDualSvm(clf, raw_output):
    gauss_stat = str(clf.n_gauss) + " (" + str(
        round((float(clf.n_gauss) / float(clf.n_gauss + clf.n_lin) * 100), 2)) + "%);"
    lin_stat = str(clf.n_lin) + " \t(" + str(
        round((float(clf.n_lin) / float(clf.n_gauss + clf.n_lin) * 100), 2)) + "%);"
    dec_margin = str(round(clf.margins[1], 3)) + ";"
    lin_c = Conversions.toPowerOfTen(clf.lin_svc.C) + ";"
    gauss_c = Conversions.toPowerOfTen(clf.gauss_svc.C) + ";"
    gauss_gamma = Conversions.toPowerOfTen(clf.gauss_svc.gamma) + ";"
    try:
        n_gaussSVs = str(clf.gauss_svc.n_support_[0] + clf.gauss_svc.n_support_[1]) + ";"
    except AttributeError:
        n_gaussSVs = "0;"

    raw_output[0].append(gauss_stat)
    raw_output[1].append(lin_stat)
    raw_output[2].append(str(dec_margin).replace(".", ","))
    raw_output[3].append(lin_c)
    raw_output[4].append(gauss_c)
    raw_output[5].append(gauss_gamma)
    raw_output[6].append(n_gaussSVs)


def writeHeader(output):
    tmp = "Dual Svm run started on " + str(time.asctime(time.localtime(time.time())))
    output.write(tmp)
    output.write("\n")


def writeDataStatistics(output, _DATA, x, x_test):
    tmp = _DATA + " (Tot: " + str(x.shape[0] + x_test.shape[0]) + " Train: " + str(x.shape[0]) + " Test: " + str(
            x_test.shape[0]) + ")\n"
    output.write(tmp)

def run(x, x_test, y, y_test, k, gridGauss, gridLin, raw_output):
        #Load the classifier
        clf = ds.DualSvm()
        clf.search_gauss = gridGauss
        clf.search_lin = gridLin
        clf.k = k

        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        appendMiscStatsDualSvm(clf, raw_output)
        appendTimeStatistics(raw_output, "dualSvm", clf, timeFit, x_test, y_test)


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
    x, x_test, y, y_test = DataLoader.load_data(data)
    gridGauss = False
    gridLin = False
    Console.write("Starting batch run, " + data)
    for j in range(4):  # Smaller steps from 0 to 20: 0, 5, 10, 15
        if j == 0:
            gridLin = True
        if j == 1:
            gridGauss = True
        Console.write("Batch run " + str(j) + ", k = " + str(0.05 * j))
        run(x, x_test, y, y_test, 0.05 * j, gridGauss, gridLin, raw_output)
        gridLin = False
        gridGauss = False
    for i in range(5):  # Bigger steps from 20 to 100: 20, 40, 60, 80, 100
        if i == 0:
            gridGauss = True
        Console.write("Batch run " + str(i + 4) + ", k = " + str(0.2 * (i + 1)))
        run(x, x_test, y, y_test, 0.2 * (i + 1), gridGauss, gridLin, raw_output)
        gridGauss = False
    Console.write("Batch run complete.")

    header = data + " " + date
    header = header.replace(" ", "_")
    header = header.replace(":", "_")
    try:
        file = 'master/output/' + header + ".csv"
        output = open(file, 'a')
    except(Exception):
        file = 'output/' + header + ".csv"
        output = open(file, 'a')

    writeHeader(output)
    writeDataStatistics(output, data, x, x_test)
    for row in raw_output:
        str_row = ""
        for cell in row:
            str_row += cell
        str_row += "\n"
        output.write(str_row)
    output.close()


# main program
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    run_batch("ijcnn")
    run_batch("cod-rna")
    run_batch("skin")
    run_batch("covtype")
    Console.write("Done!")
