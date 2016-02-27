# -*- coding: utf-8 -*-

import datetime
import time

import Converter


def write(str):
    debug = open("debug.txt", mode='a')
    time_str = "[" + datetime.datetime.now().strftime('%H:%M:%S') + "]: "
    print(time_str + str + "\n")
    debug.write(time_str + str + "\n")


def appendStats(raw_output, clf, score, k, timeFit, timePredict):
    _timeFitLin = clf.time_fit_lin  # SS_MSMS
    _timeFitGauss = clf.time_fit_gauss  # MM:SS
    _timeFitOver = clf.time_overhead  # MSMS
    # _timeTotal = _timeFitLin + _timeFitGauss + _timeFitOver
    _timeTotal = timeFit
    _score = score
    _error = round((1 - _score) * 100, 2)
    _timePredict = timePredict  ## SS_MSMS

    gauss_stat = str(clf.n_gauss) + " (" + str(
        round((float(clf.n_gauss) / float(clf.n_gauss + clf.n_lin) * 100), 2)) + "%);"
    lin_stat = str(clf.n_lin) + " \t(" + str(
        round((float(clf.n_lin) / float(clf.n_gauss + clf.n_lin) * 100), 2)) + "%);"
    lin_c = Converter.toPowerOfTen(clf.lin_svc.C) + ";"
    gauss_c = Converter.toPowerOfTen(clf.gauss_svc.C) + ";"
    gauss_gamma = Converter.toPowerOfTen(clf.gauss_svc.gamma) + ";"

    try:
        n_gaussSVs = str(clf.gauss_svc.n_support_[0] + clf.gauss_svc.n_support_[1]) + ";"
    except AttributeError:
        n_gaussSVs = "0;"

    # Bring Time data in readable format
    # _timeFit = Conversions.secondsToHourMin(_timeTotal)
    _timeFitLin = Converter.secondsToSecMilsec(_timeFitLin)
    _timeFitGauss = Converter.secondsToMinSec(_timeFitGauss)
    _timeFitOver = Converter.secondsToMilsec(_timeFitOver)
    _timePredict = Converter.secondsToSecMilsec(_timePredict)

    raw_output[0].append(gauss_stat)
    raw_output[1].append(lin_stat)
    raw_output[2].append(lin_c)
    raw_output[3].append(gauss_c)
    raw_output[4].append(gauss_gamma)
    raw_output[5].append(n_gaussSVs)
    raw_output[6].append(str(_timeTotal).replace(".", ",") + "s;")
    raw_output[7].append(_timeFitGauss + ";")
    raw_output[8].append(_timeFitLin + ";")
    raw_output[9].append(_timeFitOver + ";")
    raw_output[10].append(_timePredict + ";")
    raw_output[11].append(str(_error) + "%;")
    raw_output[12].append(str(k) + ";")

    return raw_output


def writeHeader(output, _DATA, x, x_test):
    tmp = "Dual Svm run started on " + str(time.asctime(time.localtime(time.time())))
    output.write(tmp)
    output.write("\n")

    tmp = _DATA + " (Tot: " + str(x.shape[0] + x_test.shape[0]) + " Train: " + str(x.shape[0]) + " Test: " + str(
        x_test.shape[0]) + ")\n"
    output.write(tmp)


def writeContent(output, raw_output):
    for row in raw_output:
        str_row = ""
        for cell in row:
            str_row += cell
        str_row += "\n"
        output.write(str_row)


def createFile(data, date):
    filestring = data + " " + date
    filestring = filestring.replace(" ", "_")
    filestring = filestring.replace(":", "_")
    try:
        file = 'master/output/' + filestring + ".csv"
        output = open(file, 'a')
    except(Exception):
        file = 'output/' + filestring + ".csv"
        output = open(file, 'a')
    return output
