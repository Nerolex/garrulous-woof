from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import DataDistributions as dd
import DualSvm as ds
import LinearSvmHelper as ls

"""
This is a test module.
"""

def runTest():
    # TODO plotting auslagern
    # TODO andere Model importieren
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

    ###
    h = 0.02
    # x_min1, x_max1 = x[:, 0].min(), x[:, 0].max()
    # y_min1, y_max1 = x[:, 1].min(), x[:, 1].max()
    x_min1, x_max1 = -2, 2
    y_min1, y_max1 = -2, 2
    xx1, yy1 = np.meshgrid(np.arange(x_min1, x_max1, h), np.arange(y_min1, y_max1, h))
    Z = dualSvm.predict(np.c_[xx1.ravel(), yy1.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, yy1, Z, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
    ###

    xx, yy = ls.getHyperplane(dualSvm._linSVC)
    yy_up, yy_down = ls.getMarginPlanes(dualSvm._linSVC, factor)
    plt.plot(xx, yy_up, "k--")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

    score = dualSvm.score(x_test, y_test)
    print(1 - score)
    plt.show()
