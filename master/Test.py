from __future__ import division

import matplotlib.pyplot as plt

import DataDistributions as dd
import DualSvm as ds
import LinearSvmHelper as ls
import PlotHelper as pl

"""
This is a test module.
"""

def runTest():
    # TODO plotting auslagern
    size = 500
    location = 0.0
    scale = 0.5
    amplitude = 0.3
    freq = 3.5

    factor = 0.7
    count = round(factor * size)

    cLin = 0.01
    cGauss = 100
    gamma = 10

    x_train, y_train = dd.generateSinusCluster(size, location, scale, amplitude, freq)
    x_test, y_test = dd.generateSinusCluster(size * 3, location, scale, amplitude, freq)
    dualSvm = ds.DualSvm(cLin, cGauss, gamma, True, factor, count)

    dualSvm.fit(x_train, y_train)

    pl.contour(dualSvm, [-2, 2], [-2, 2], 0.02)

    xx, yy = ls.getHyperplane(dualSvm._linSVC)
    yy_up, yy_down = ls.getMarginPlanes(dualSvm._linSVC, factor)
    plt.plot(xx, yy_up, "k--")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

    score = dualSvm.score(x_test, y_test)
    print("Score:" + (1 - score))
    plt.show()
