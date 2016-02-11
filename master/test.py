import multiprocessing

import numpy as np
import sklearn.cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

import DataLoader as dl

x, x_test, y, y_test = dl.load_covtype()
x, a, y, b = cv.train_test_split(x, y, train_size=0.2)

c_range = np.logspace(-6, 4, 11, base=10.0)
gamma_range = np.logspace(-6, 4, 11, base=10.0)
param_grid = dict(gamma=gamma_range, C=c_range)
n_cpu = multiprocessing.cpu_count()

grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, n_jobs=n_cpu)
grid = grid.fit(x, y)

print(1 - grid.best_score_)
print(grid.best_params_)
