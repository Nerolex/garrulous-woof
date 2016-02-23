import random

import numpy as np
import sklearn.cross_validation as cv

import DataLoader as dl

x, x_test, y, y_test = dl.load_covtype()
x, a, y, b = cv.train_test_split(x, y, train_size=0.2)

k = 0.05
n = np.ceil(k * x.shape[0])  # Random ziehen.
a = np.arange(x.shape[0])
gauss_indices = random.sample(a, int(n))
lin_indices = np.setdiff1d(np.arange(x.shape[0]), gauss_indices)
c = np.set
