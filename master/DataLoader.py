import numpy as np
import sklearn.cross_validation as cv
import sklearn.datasets as da

import DataDistributions as dd


def load_data(dataType):
    if dataType == "sinus":
        x, x_test, y, y_test = load_sinus()
    elif dataType == "iris":
        x, x_test, y, y_test = load_iris()
    elif dataType == "codrna":
        x, x_test, y, y_test = load_codrna()
    return x, x_test, y, y_test


def load_codrna():
    data, target = da.load_svmlight_file("data/cod_rna/cod-rna.txt", 8)
    x, x_test, y, y_test = data, data, target, target
    # = cv.train_test_split(data, data, test_size=1 / 3)

    return x, x_test, y, y_test


def load_iris():
    data = da.load_iris()
    y = data.target
    x = data.data
    x, x_test, y, y_test = cv.train_test_split(data.data, data.target, test_size=1 / 3)
    y = np.where(y == 1, -1, 1)
    y_test = np.where(y_test == 1, -1, 1)
    return x, x_test, y, y_test


def load_sinus():
    size = 5000
    location = 0.0
    scale = 0.5
    amplitude = 0.3
    freq = 3.5
    x, y = dd.generateSinusCluster(size, location, scale, amplitude, freq)
    x_test, y_test = dd.generateSinusCluster(size * 3, location, scale, amplitude, freq)
    return x, x_test, y, y_test
