# -*- coding: utf-8 -*-

import sys
import time
import warnings

from Classifier import DualSvm
from Data import DataLoader
from ParameterTuning import Gridsearcher
from Tools import IOHelper

'''
Main module for the project. Takes a list of data strings as argument and automatically searches in /data/ for the corresponding dataset and loads the data.
A DualSVM Classifier is created, trained and tested on the data. The results are saved to /ouput/NAME_OF_DATA.csv
'''

class Run_Results(object):
    def __init__(self):
        self.n_gauss = []
        self.n_lin = []
        self.c_lin = []
        self.c_gauss = []
        self.gamma = []
        self.sv_gauss = []
        self.time_fit = []
        self.time_fit_gauss = []
        self.time_fit_linear = []
        self.time_fit_overhead = []
        self.time_predict = []
        self.error = []
        self.k = []

        self.length = 0

    def fitLength(self, length):
        if length > self.length:
            for i in range(length - self.length):
                self.n_gauss.append(0)
                self.n_lin.append(0)
                self.c_lin.append(0)
                self.c_gauss.append(0)
                self.gamma.append(0)
                self.sv_gauss.append(0)
                self.time_fit.append(0)
                self.time_fit_gauss.append(0)
                self.time_fit_linear.append(0)
                self.time_fit_overhead.append(0)
                self.time_predict.append(0)
                self.error.append(0)
                self.k.append(0)
            self.length = length


def run_batch(data, n_iterations=1, random_decision=False, singleParam=False):
    '''
    Represents multiple runs of the DualSVM classifier.

    :param data: The data to load.
    :param n_iterations: The number of runs on the data. Results will be averaged.
    :param random_decision: If true, points are given to the second classifier by random means. If false, distance to hyperplane is used.
    :param singleParam: If true, uses tuned values for C and Gamma for the second classifier for k=0.1. If false, uses all stored values in the corresponding params-file.
    :return:
    '''
    # Keep Track of the date of the start of the program
    date = str(time.asctime(time.localtime(time.time())))

    all_results = []

    for i in range(n_iterations):
        results = Run_Results()

        IOHelper.write("Iteration " + str(i))
        # Load the data
        x, x_test, y, y_test = DataLoader.load_data(data)
        c_lin, c_gauss, gamma = Gridsearcher.loadParametersFromFile(data, singleParam)
        IOHelper.write("Starting batch run, " + data)

        for j in range(4):  # Smaller steps from 0 to 20: 0, 5, 10, 15
            n = j
            k = 0.05 * j
            IOHelper.write("Batch run " + str(j) + ", k = " + str(k))
            run(not random_decision, k, n, c_gauss, gamma, c_lin, x, x_test, y, y_test, results)
        for i in range(5):  # Bigger steps from 20 to 100: 20, 40, 60, 80, 100
            n = 4 + i
            k = 0.2 * (i + 1)
            IOHelper.write("Batch run " + str(i + 4) + ", k = " + str(k))
            run(not random_decision, k, n, c_gauss, gamma, c_lin, x, x_test, y, y_test, results)

        all_results.append(results)
        IOHelper.write("Batch run complete.")

    length = len(all_results[0].c_lin)
    end_result = Run_Results()
    end_result.fitLength(length)
    end_result.c_lin = all_results[0].c_lin
    end_result.c_gauss = all_results[0].c_gauss
    end_result.gamma = all_results[0].gamma
    end_result.n_gauss = all_results[0].n_gauss
    end_result.n_lin = all_results[0].n_lin
    end_result.k = all_results[0].k
    for result in all_results:
        for i in range(length):
            end_result.sv_gauss[i] += result.sv_gauss[i]
            end_result.time_fit[i] += result.time_fit[i]
            end_result.time_fit_gauss[i] += result.time_fit_gauss[i]
            end_result.time_fit_linear[i] += result.time_fit_linear[i]
            end_result.time_fit_overhead[i] += result.time_fit_overhead[i]
            end_result.time_predict[i] += result.time_predict[i]
            end_result.error[i] += result.error[i]
    for i in range(length):
        end_result.sv_gauss[i] /= n_iterations
        end_result.time_fit[i] /= n_iterations
        end_result.time_fit_gauss[i] /= n_iterations
        end_result.time_fit_linear[i] /= n_iterations
        end_result.time_fit_overhead[i] /= n_iterations
        end_result.time_predict[i] /= n_iterations
        end_result.error[i] /= n_iterations

    IOHelper.createShortFile(data, date, end_result, random_decision, singleParam)
    IOHelper.createLongFile(data, date, end_result, random_decision, singleParam)


def run(use_distance, k, n, c_gauss, gamma, c_lin, x, x_test, y, y_test, results):
    '''
    Represents a single run of the DualSVM classifier.
    '''
    clf = DualSvm(use_distance=use_distance)
    clf.k = k

    # Apply Parameters
    clf.c_gauss = c_gauss[n]
    clf.gamma = gamma[n]
    clf.c_lin = c_lin

    timeStart = time.time()
    clf.fit(x, y)
    timeFit = time.time() - timeStart

    timeStart = time.time()
    score = clf.score(x_test, y_test)
    timePredict = time.time() - timeStart

    results.n_gauss.append(clf.n_gauss)
    results.n_lin.append(clf.n_lin)
    results.c_lin.append(round(clf.c_lin, 4))
    results.c_gauss.append(round(clf.c_gauss, 4))
    results.gamma.append(round(clf.gamma, 4))
    try:
        results.sv_gauss.append(clf.gauss_svc.n_support_[0] + clf.gauss_svc.n_support_[1])
    except AttributeError:
        results.sv_gauss.append(0)
    results.time_fit.append(round(timeFit, 4))
    results.time_fit_gauss.append(round(clf.time_fit_gauss, 4))
    results.time_fit_linear.append(round(clf.time_fit_lin, 4))
    results.time_fit_overhead.append(round(clf.time_overhead, 4))
    results.time_predict.append(round(timePredict, 4))
    results.error.append(round(1 - score, 4))
    results.k.append(k)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    data = sys.argv
    data.pop(0)
    n = 5

    for datastring in data:
        run_batch(datastring, 1, False)
        run_batch(datastring, n, True)
        run_batch(datastring, 1, False, True)

    '''
    data = "ijcnn"
    x, x_test, y, y_test = DataLoader.load_data(data)
    print(data + "Train: x " + str(x.shape[0]) + " y " + str(y.shape[0]) + " Test: x " + str(x_test.shape[0]) + " y: " + str(y_test.shape[0]))
    data = "shuttle"
    x, x_test, y, y_test = DataLoader.load_data(data)
    print(data + "Train: x " + str(x.shape[0]) + " y " + str(y.shape[0]) + " Test: x " + str(x_test.shape[0]) + " y: " + str(y_test.shape[0]))
    data = "covtype"
    x, x_test, y, y_test = DataLoader.load_data(data)
    print(data + "Train: x " + str(x.shape[0]) + " y " + str(y.shape[0]) + " Test: x " + str(x_test.shape[0]) + " y: " + str(y_test.shape[0]))
    data = "skin"
    x, x_test, y, y_test = DataLoader.load_data(data)
    print(data + "Train: x " + str(x.shape[0]) + " y " + str(y.shape[0]) + " Test: x " + str(x_test.shape[0]) + " y: " + str(y_test.shape[0]))
    data = "mnist"
    x, x_test, y, y_test = DataLoader.load_data(data)
    print(data + "Train: x " + str(x.shape[0]) + " y " + str(y.shape[0]) + " Test: x " + str(x_test.shape[0]) + " y: " + str(y_test.shape[0]))
    '''
    '''
    Converter.convertParamsToCsv("skin", "output/skin-formatParams.csv")
    Converter.convertParamsToCsv("covtype", "output/covtype-formatParams.csv")
    Converter.convertParamsToCsv("shuttle", "output/shuttle-formatParams.csv")
    Converter.convertParamsToCsv("ijcnn", "output/ijcnn-formatParams.csv")
    Converter.convertParamsToCsv("mnist", "output/mnist-formatParams.csv")
    '''
