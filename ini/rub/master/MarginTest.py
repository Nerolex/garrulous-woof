from __future__ import division

import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn.svm as sk


def generateSinusCluster(pSize, location=0.0, scale=0.5, amplitude=0.2, freq=3):
    data = np.random.normal(loc=location, scale=scale, size=(pSize, 2))
    dataX = np.array([row[0] for row in data])
    dataY = np.array([row[1] for row in data])

    label = np.zeros(pSize)
    for i in range(pSize):
        if dataY[i] > amplitude * math.sin(dataX[i] * math.pi * freq):
            label[i] = 1
        else:
            label[i] = -1

    return data, label


def getHyperplane(clf, xstart, xend, factor=1):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(xstart, xend)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    yy_down = yy + factor * margin
    yy_up = yy - factor * margin

    return xx, yy, yy_down, yy_up


def plotHyperplane(xx, yy):
    plt.plot(xx, yy, 'k-')


def getMarginPlane(clf, X, up=-1):
    above = np.array((0, 0))
    below = np.array((0, 0))
    w = clf.coef_[0]
    margin = 1 / np.sqrt(np.sum(w ** 2))
    b = clf.intercept_[0]

    # Calculates support vectors with margin<= 1-val
    for i in range(X.shape[0]):
        posMargin = (np.inner(w, X[i]) + b) + up
        if (posMargin >= 0):
            above = np.vstack((above, X[i]))
        else:
            below = np.vstack((below, X[i]))
    return above[1:], below[1:]


size = 500
location = 0.0
scale = 0.5
amplitude = 0.6
freq = 3

x, y = generateSinusCluster(size, location, scale, amplitude, freq)
linSvm = sk.LinearSVC().fit(x, y)

factor = 0.7

xx, yy, yy_down, yy_up = getHyperplane(linSvm, -2, 2, factor)
plotHyperplane(xx, yy)
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
red, green = getMarginPlane(linSvm, x, -factor)

green1, red1 = getMarginPlane(linSvm, green, factor)

red = np.vstack((red, red1))

plt.scatter(red[:, 0], red[:, 1], c="RED")
plt.scatter(green1[:, 0], green1[:, 1], c="GREEN")
plt.show()
plt.show()
