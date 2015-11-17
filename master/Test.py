from __future__ import division

import matplotlib.pyplot as plt

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

    dualSvm = ds.DualSvm(1, 100, 10, False, 0, 125)

    dualSvm.fit(x, y)
    dualSvm.predict(x)
    xx, yy = ls.getHyperplane(dualSvm._linSVC)

    plt.plot(xx, yy)
    plt.scatter(x[:, 0], x[:, 1])
    plt.ylim(-1, 1)
    plt.xlim(-2, 2)
    plt.show()
