# -*- coding: utf-8 -*-

import time
import warnings

from Classifier import DualSvm
from Data import DataLoader
from ParameterTuning import Gridsearcher
from Tools import IOHelper


def run_batch(data):
    # Keep Track of the date of the start of the program
    date = str(time.asctime(time.localtime(time.time())))

    # Build array for the output csv file
    raw_output = [["Points gaussian;"],  # 0
                  ["Points linear;"],  # 1
                  ["C linear;"],  # 2
                  ["C gauss;"],  # 3
                  ["gamma gauss;"],  # 4
                  ["number SVs gauss;"],  # 5
                  ["time to fit;"],  # 6
                  ["      gauss;"],  # 7
                  ["      linear;"],  # 8
                  ["      overhead;"],  # 9
                  ["time to predict;"],  # 10
                  ["Error;"],  # 11
                  ["Percentage gaussian;"]  # 12
                  ]

    # Load the data
    x, x_test, y, y_test = DataLoader.load_data(data)
    k = 0
    c_lin, c_gauss, gamma = Gridsearcher.loadParametersFromFile(data)

    IOHelper.write("Starting batch run, " + data)
    use_distance = True
    n = 0

    for j in range(4):  # Smaller steps from 0 to 20: 0, 5, 10, 15
        n = j
        k = 0.05 * j
        IOHelper.write("Batch run " + str(j) + ", k = " + str(k))
        run(c_gauss, c_lin, gamma, k, n, raw_output, use_distance, x, x_test, y, y_test)

    for i in range(5):  # Bigger steps from 20 to 100: 20, 40, 60, 80, 100
        n = 4 + i
        k = 0.2 * (i + 1)
        IOHelper.write("Batch run " + str(i + 4) + ", k = " + str(k))
        run(c_gauss, c_lin, gamma, k, n, raw_output, use_distance, x, x_test, y, y_test)

    IOHelper.write("Batch run complete.")

    output = IOHelper.createFile(data, date)
    IOHelper.writeHeader(output, data, x, x_test)
    IOHelper.writeContent(output, raw_output)
    output.close()


def run(c_gauss, c_lin, gamma, k, n, raw_output, use_distance, x, x_test, y, y_test):
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

    IOHelper.appendStats(raw_output, clf, score, k, timeFit, timePredict)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    run_batch("ijcnn")
