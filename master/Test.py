from __future__ import division

import matplotlib.pyplot as plt
import sklearn.svm as sk

import DataDistributions as dd
import DualSvm as ds
import LinearSvmHelper as ls

"""
This is a test module.
"""

def runTest():
    size = 500
    location = 0.0
    scale = 0.5
    amplitude = 0.6
    freq = 3

    x, y = dd.generateSinusCluster(size, location, scale, amplitude, freq)
    linSvm = sk.LinearSVC().fit(x, y)

    factor = 0.7

    xx, yy = ls.getHyperplane(linSvm)
    yy_down, yy_up = ls.getMarginPlanes(linSvm, factor)
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    result = ds.getPointsCloseToHyperplaneByCount(linSvm, x, 250)

    plt.scatter(result[:, 0], result[:, 1], c=-1 * result[:, 2], cmap=plt.cm.RdYlGn)
    # plt.scatter(inner[:, 0], inner[:, 1], c=-1 * inner[:, 2], cmap=plt.cm.RdYlGn)
    plt.show()
