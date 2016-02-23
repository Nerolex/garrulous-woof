# -*- coding: utf-8 -*-
from __future__ import division

import multiprocessing
import time
import warnings

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC

import Console
import Conversions
import DataLoader
import DualSvm as ds


def gridsearch_for_linear(X, y):
    """
    Parameter tuning for the linear classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal values of the first grid.

    :param X: Data x
    :param y: Labels y
    :return: Best parameters
    """
    Console.write("Linear SVC: Starting coarse gridsearch.")
    # LinSvm gridSearch
    c_range = np.logspace(-2, 10, 13, base=10.0)
    param_grid = dict(C=c_range)
    grid = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=4)  # TODO Variable njobs
    grid.fit(X, y)

    _c = grid.best_params_['C']

    Console.write("Linear SVC: Finished coarse gridsearch with params: C: " + str(_c))
    Console.write("Linear SVC: Starting fine gridsearch:")

    # c_range_2 = np.linspace(_c - 0.5 * _c, _c + 0.5 * _c, num=5)
    c_range_2 = [_c - 0.5 * _c, _c, 2 * _c]
    param_grid = dict(C=c_range_2)
    grid = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=4)
    grid.fit(X, y)

    _c = grid.best_params_['C']
    Console.write("Linear SVC: Finished fine gridsearch with params: C: " + str(_c))

    return _c


def gridsearch_for_gauss(X, y):
    """
    Parameter tuning for the gauss classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal val

    :param X: Data x
    :param y: Labels y
    :return: Best parameters
    """
    n_cpu = multiprocessing.cpu_count()
    print("Using multiprocessing. Avaiable cores: " + str(n_cpu))
    Console.write("Gauss SVC: Starting gridsearch for gaussian classifier.")
    c_range = np.logspace(-4, 4, 9, base=10.0)
    gamma_range = np.logspace(-6, 2, 9, base=10.0)
    param_grid = dict(gamma=gamma_range, C=c_range)

    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, n_jobs=n_cpu)
    grid.fit(X, y)
    _c = grid.best_params_['C']
    _gamma = grid.best_params_['gamma']

    print("First search complete. Starting second search...")

    Console.write("Gauss SVC: Finished coarse gridsearch with params: C: " + str(_c) + " gamma: " + str(_gamma))
    Console.write("Gauss SVC: Starting fine for gaussian classifier.")

    # c_range_2 = np.linspace(_c - 0.5 * _c, _c + 0.5 * _c, num=5)
    c_range_2 = [_c - 0.5 * _c, _c, 2 * _c]
    # gamma_range_2 = np.linspace(_gamma - 0.5 * _gamma, _gamma + 0.5 * _gamma, num=5)
    gamma_range_2 = [_gamma - 0.5 * _gamma, _gamma, 2 * _gamma]

    param_grid = dict(gamma=gamma_range_2, C=c_range_2)
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, n_jobs=n_cpu)
    grid.fit(X, y)

    _c = grid.best_params_['C']
    _gamma = grid.best_params_['gamma']

    Console.write("Gauss SVC: Finished fine gridsearch with params: C: " + str(_c) + " gamma: " + str(_gamma))

    return _c, _gamma

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

    raw_output[7].append(str(_timeTotal).replace(".", ",") + "s;")
    # raw_output[8].append(_timeFitGauss + "\t(" + str(_percentGaussTotal) + "%)" + ";")
    raw_output[8].append(_timeFitGauss + ";")
    # raw_output[9].append(_timeFitLin + "\t(" + str(_percentLinTotal) + "%)" + ";")
    raw_output[9].append(_timeFitLin + ";")
    # raw_output[10].append(_timeFitOver + "\t(" + str(_percentOverTotal) + "%)" + ";")
    raw_output[10].append(_timeFitOver + ";")
    raw_output[11].append(_timePredict + ";")
    raw_output[12].append(str(_error) + "%;")

    return raw_output


def appendMiscStatsDualSvm(clf, raw_output):
    gauss_stat = str(clf.n_gauss) + " (" + str(
        round((float(clf.n_gauss) / float(clf.n_gauss + clf.n_lin) * 100), 2)) + "%);"
    lin_stat = str(clf.n_lin) + " \t(" + str(
        round((float(clf.n_lin) / float(clf.n_gauss + clf.n_lin) * 100), 2)) + "%);"
    dec_margin = str(round(clf._gauss_distance, 3)) + ";"
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
    k = 0
    c_lin = [10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000,
             10000000000]
    c_gauss = [0, 800, 1, 1, 10, 10, 10, 10, 10]
    gamma = [0, 0.01, 200, 200, 0.001, 0.001, 0.001, 0.001, 0.001]
    Console.write("Starting batch run, " + data)
    gridLinear = False
    gridGauss = False
    use_distance = True
    n = 0

    for j in range(4):  # Smaller steps from 0 to 20: 0, 5, 10, 15
        n = j
        Console.write("Batch run " + str(j) + ", k = " + str(0.05 * j))
        # Load the classifier
        k = 0.05 * j
        clf = ds.DualSvm(use_distance=use_distance)
        clf.k = k

        # Parameter Tuning
        if j == 0 and gridLinear:  # In the first run, calculate best parameters for linear svm
            c_lin[n] = gridsearch_for_linear(x, y)
        else:
            clf.c_lin = c_lin[n]
            clf.fit_lin_svc(x,
                            y)  # Fit linear classifier beforehand. This is necessary for the get_points method to work correctly.
            x_gauss, y_gauss, margins = clf.get_points_close_to_hyperplane_by_count(x, y, k)
            if gridGauss:
                c_gauss[n], gamma[n] = gridsearch_for_gauss(x_gauss,
                                                            y_gauss)  # In the following runs, do the same for the gaussian svm, as the subset of points for the classifier is changing

        # Apply Parameters
        clf.c_gauss = c_gauss[n]
        clf.gamma = gamma[n]
        clf.c_lin = c_lin[n]

        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        appendMiscStatsDualSvm(clf, raw_output)
        appendTimeStatistics(raw_output, "dualSvm", clf, timeFit, x_test, y_test)

    for i in range(5):  # Bigger steps from 20 to 100: 20, 40, 60, 80, 100
        n = 4 + i
        Console.write("Batch run " + str(i + 4) + ", k = " + str(0.2 * (i + 1)))

        # Load the classifier
        k = 0.2 * (i + 1)
        clf = ds.DualSvm(use_distance=use_distance)
        clf.k = k

        if i == 0:
            clf.c_lin = c_lin[n]
            clf.fit_lin_svc(x, y)
            x_gauss, y_gauss, margins = clf.get_points_close_to_hyperplane_by_count(x, y, k)
            if gridGauss and 1 <= i < 3:
                c_gauss[n], gamma[n] = gridsearch_for_gauss(x_gauss, y_gauss)

        # Apply Parameters
        clf.c_gauss = c_gauss[n]
        clf.gamma = gamma[n]
        clf.c_lin = c_lin[n]

        timeStart = time.time()
        clf.fit(x, y)
        timeFit = time.time() - timeStart

        appendMiscStatsDualSvm(clf, raw_output)
        appendTimeStatistics(raw_output, "dualSvm", clf, timeFit, x_test, y_test)

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
    #run_batch("ijcnn")
    run_batch("cod-rna")
    # run_batch("skin")
    # run_batch("covtype")
    # run_batch("cluster")
    # run_batch("clusterx")
    #run_batch("clustery")
    Console.write("Done!")
