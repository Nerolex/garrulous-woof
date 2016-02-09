# -*- coding: utf-8 -*-
from __future__ import division

import math

import numpy as np

"""
This module is for data generating purposes.
"""

def generateSinusCluster(size, location=0.0, scale=0.5, amplitude=0.2, freq=3):
    """
    Function that genrates a random gaussian cluster and separates between two classes with a sinus function.

    @type size: number
    @param size: Size of the generated dataset
    @type location: float
    @param size: Mean (“centre”) of the distribution.
    @type scale: float
    @param scale: Standard deviation (spread or “width”) of the distribution.
    @type amplitude: float
    @param amplitude: Amplitude of the underlying sinus function.
    @type freq: float
    @param freq: Frequency of the underlying sinus function
    @rtype: Array
    @return: Arrays for datapoints and labels
    """

    data = np.random.normal(loc=location, scale=scale, size=(size, 2))
    dataX = np.array([row[0] for row in data])
    dataY = np.array([row[1] for row in data])

    label = np.zeros(size)
    for i in range(size):
        if dataY[i] > amplitude * math.sin(dataX[i] * math.pi * freq):
            label[i] = 1
        else:
            label[i] = -1

    return data, label
