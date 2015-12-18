# -*- coding: utf-8 -*-
import numpy as np
import sklearn.cross_validation as cv
import sklearn.datasets as da

import DataDistributions as dd


def load_data(dataType):
    if dataType == "sinus":
        x, x_test, y, y_test = load_sinus()
    elif dataType == "cod-rna":
        x, x_test, y, y_test = load_codrna()
    elif dataType == "covtype":
        x, x_test, y, y_test = load_covtype()
    elif dataType == "banana":
        x, x_test, y, y_test = load_banana()
    else:
        x, x_test, y, y_test = load_libsvm_file(dataType)
    return x, x_test, y, y_test


def load_codrna():
    x, y = da.load_svmlight_file("data/cod-rna/cod-rna.txt", 8)
    x_test, y_test = da.load_svmlight_file("data/cod-rna/cod-rna.t", 8)
    return x, x_test, y, y_test


def load_banana():
    data = da.fetch_mldata("Banana IDA")
    x = data.data
    y = data.target
    x_test = x
    y_test = y
    return x, x_test, y, y_test

def load_sinus():
    size = 1000
    location = 0.0
    scale = 0.5
    amplitude = 0.3
    freq = 3.5
    x, y = dd.generateSinusCluster(size, location, scale, amplitude, freq)
    x_test, y_test = dd.generateSinusCluster(size * 3, location, scale, amplitude, freq)
    return x, x_test, y, y_test


def load_covtype():
    # ToDo: Hier schöneren Weg finden, covtype anteilig zu laden
    _SAMPLESIZE = 0.1

    data, target = da.load_svmlight_file("data/covtype/covtype.libsvm.binary", 54)
    target = np.where(target == 1, -1, 1)
    x, x_test, y, y_test = cv.train_test_split(data, target, train_size=_SAMPLESIZE)
    x_test, o, y_test, o = cv.train_test_split(x_test, y_test, train_size=_SAMPLESIZE)  # DELETE THIS
    return x, x_test, y, y_test

def load_libsvm_file(filename):
    '''
    Method that loads a libsvm file and returns training and test np-arrays.

    @param filename:
    @return:
    '''
    filestring = "data/" + filename + "/" + filename
    x, y = da.load_svmlight_file(filestring + ".txt")
    x_test, y_test = da.load_svmlight_file(filestring + ".t")
    return x, x_test, y, y_test
