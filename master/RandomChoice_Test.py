from random import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import perceptron


# 10000 P_gammaunkte, drei Gauss Verteilungen
# Rauschen auf Daten
# Rauschen auf Labeln

def add_x_noise(X):
    for i in range(X.shape[0]):
        if random() > 0.8:
            X[i] = X[i] + random()
    return X


def add_y_noise(Y):
    for i in range(len(Y)):
        if random() > 0.8:
            Y[i] = Y[i] * -1
    return Y


x1 = np.random.normal(size=(10000, 2))
x1[:, 0] += 2
y1 = np.ones(10000) * 1

x2 = np.random.normal(size=(10000, 2))
x2[:, 0] -= 2
x2[:, 1] += 6
y2 = np.ones(10000) * -1

x3 = np.random.normal(size=(10000, 2))
x3[:, 0] += 2
x3[:, 1] += 12
y3 = np.ones(10000) * 1

X = np.vstack((np.vstack((x1, x2)), x3))
Y = np.concatenate((np.concatenate((y1, y2)), y3))
x, x_test, y, y_test = train_test_split(X, Y, train_size=0.6)

score = 0
for i in range(10):
    clf = perceptron.Perceptron().fit(x, y)
    score += 1 - clf.score(x_test, y_test)
score /= 10

print "Error with/o x noise: " + str(score)

x = add_x_noise(x)
score = 0
for i in range(10):
    clf = perceptron.Perceptron().fit(x, y)
    score += 1 - clf.score(x_test, y_test)
score /= 10

print "Error with x noise: " + str(score)

a = [1, -1, -1, 1]
add_y_noise(a)
print a

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

plt.plot(xx, yy, 'k-')
plt.scatter(x1[:, 0], x1[:, 1], c=y1, cmap="seismic", vmin=-2, vmax=2)
plt.scatter(x2[:, 0], x2[:, 1], c=y2, cmap="seismic", vmin=-2, vmax=2)
plt.scatter(x3[:, 0], x3[:, 1], c=y3, cmap="seismic", vmin=-2, vmax=2)
plt.xlim((-8, 8))
plt.ylim((-4, 18))
plt.show()
