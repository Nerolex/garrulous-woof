from __future__ import division

import matplotlib.pyplot as plt

import DataDistributions as dd
import DualSvm as ds
import LinearSvmHelper as ls
import PlotHelper as pl

"""
This is a test module.
"""


# TODO GridSearch ausprobieren -> Interface!

def runTest():
    size = 500
    location = 0.0
    scale = 0.5
    amplitude = 0.3
    freq = 3.5

    factor = 0.7
    count = round(factor * size)

    # data = da.load_iris()
    # print(data.data.shape, data.target.shape)
    # print(data.target)
    # y = data.target
    # x = data.data
    # TODO import Data!
    # x, x_test, y, y_test = cv.train_test_split(data.data, data.target, test_size=1/7)
    # y = np.where(y == 0, -1, 1)
    #y_test = np.where(y_test == 0, -1, 1)

    cLin = 0.01
    cGauss = 100
    gamma = 10

    x, y = dd.generateSinusCluster(size, location, scale, amplitude, freq)
    x_test, y_test = dd.generateSinusCluster(size * 3, location, scale, amplitude, freq)
    dualSvm = ds.DualSvm(cLin, cGauss, gamma, True, factor, count)

    dualSvm.fit(x, y)
    error = 1 - dualSvm.score(x_test, y_test)
    print("Error: ",error)

    pl.contour(dualSvm, [-2, 2], [-2, 2], 0.02)

    xx, yy = ls.getHyperplane(dualSvm._linSVC)
    yy_up, yy_down = ls.getMarginPlanes(dualSvm._linSVC, factor)

    plt.plot(xx, yy_up, "k--")
    plt.plot(xx, yy_down, "k--")
    plt.plot(xx, yy)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)


    plt.show()
