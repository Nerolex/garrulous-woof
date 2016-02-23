import multiprocessing

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC

import Console
import DataLoader
import DualSvm


def gridsearch_for_linear(X, y):
    """
    Parameter tuning for the linear classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal values of the first grid.

    :param X: Data x
    :param y: Labels y
    :return: Best parameters
    """
    n_cpu = multiprocessing.cpu_count()
    Console.write("Linear SVC: Starting coarse gridsearch.")
    # LinSvm gridSearch
    c_range = np.logspace(-2, 10, 13, base=10.0)
    param_grid = dict(C=c_range)
    grid = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=n_cpu)
    grid.fit(X, y)

    _c = grid.best_params_['C']

    Console.write("Linear SVC: Finished coarse gridsearch with params: C: " + str(_c))
    Console.write("Linear SVC: Starting fine gridsearch:")

    # c_range_2 = np.linspace(_c - 0.5 * _c, _c + 0.5 * _c, num=5)
    c_range_2 = [_c - 0.5 * _c, _c, 2 * _c]
    param_grid = dict(C=c_range_2)
    grid = GridSearchCV(LinearSVC(), param_grid=param_grid, n_jobs=n_cpu)
    grid.fit(X, y)

    _c = grid.best_params_['C']
    Console.write("Linear SVC: Finished fine gridsearch with params: C: " + str(_c))

    return _c


def gridsearch_for_gauss(X, y):
    """
    Parameter tuning for the gauss classifier in two stages. First tuning is done on a coarse grid, second on a finer grid at the position of the optimal val

    :param X: Data x
    :param y: Labels y
    :return: Best parameters
    """
    n_cpu = multiprocessing.cpu_count()
    print("Using multiprocessing. Avaiable cores: " + str(n_cpu))
    Console.write("Gauss SVC: Starting gridsearch for gaussian classifier.")
    c_range = np.logspace(-4, 4, 9, base=10.0)
    gamma_range = np.logspace(-6, 2, 9, base=10.0)
    param_grid = dict(gamma=gamma_range, C=c_range)

    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, n_jobs=n_cpu)
    grid.fit(X, y)
    _c = grid.best_params_['C']
    _gamma = grid.best_params_['gamma']

    print("First search complete. Starting second search...")

    Console.write("Gauss SVC: Finished coarse gridsearch with params: C: " + str(_c) + " gamma: " + str(_gamma))
    Console.write("Gauss SVC: Starting fine for gaussian classifier.")

    # c_range_2 = np.linspace(_c - 0.5 * _c, _c + 0.5 * _c, num=5)
    c_range_2 = [_c - 0.5 * _c, _c, 2 * _c]
    # gamma_range_2 = np.linspace(_gamma - 0.5 * _gamma, _gamma + 0.5 * _gamma, num=5)
    gamma_range_2 = [_gamma - 0.5 * _gamma, _gamma, 2 * _gamma]

    param_grid = dict(gamma=gamma_range_2, C=c_range_2)
    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, n_jobs=n_cpu)
    grid.fit(X, y)

    _c = grid.best_params_['C']
    _gamma = grid.best_params_['gamma']

    Console.write("Gauss SVC: Finished fine gridsearch with params: C: " + str(_c) + " gamma: " + str(_gamma))

    return _c, _gamma


def run(data):
    Console.write("Starting parameter tuning for " + data)
    x, x_test, y, y_test = DataLoader.load_data(data)
    file_string = "output/data" + "-params.txt"

    k = 0
    n = 0

    c_lin = 0
    c_gauss = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    gamma = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for j in range(4):  # Smaller steps from 0 to 20: 0, 5, 10, 15
        n = j
        Console.write("Batch run " + str(j) + ", k = " + str(0.05 * j))
        # Load the classifier
        k = 0.05 * j
        clf = DualSvm.DualSvm(use_distance=True)
        clf.k = k

        # Parameter Tuning
        if j == 0:  # In the first run, calculate best parameters for linear svm
            c_lin = gridsearch_for_linear(x, y)
        else:
            clf.c_lin = c_lin
            clf.fit_lin_svc(x,
                            y)  # Fit linear classifier beforehand. This is necessary for the get_points method to work correctly.
            x_gauss, y_gauss, margins = clf.get_points_close_to_hyperplane_by_count(x, y, k)
            c_gauss[n], gamma[n] = gridsearch_for_gauss(x_gauss,
                                                        y_gauss)  # In the following runs, do the same for the gaussian svm, as the subset of points for the classifier is changing

    for i in range(5):  # Bigger steps from 20 to 100: 20, 40, 60, 80, 100
        n = 4 + i
        Console.write("Batch run " + str(i + 4) + ", k = " + str(0.2 * (i + 1)))

        # Load the classifier
        k = 0.2 * (i + 1)
        clf = DualSvm.DualSvm(use_distance=True)
        clf.k = k

        if k <= 6:
            clf.c_lin = c_lin
            clf.fit_lin_svc(x, y)
            x_gauss, y_gauss, margins = clf.get_points_close_to_hyperplane_by_count(x, y, k)
            c_gauss[n], gamma[n] = gridsearch_for_gauss(x_gauss, y_gauss)

    output = open(file_string, 'w')
    output.write(str(c_lin) + "\n")
    for value in c_gauss:
        output.write(str(value) + ",")
    output.write("\n")
    for value in gamma:
        output.write(str(gamma) + ",")
    output.write("\n")

