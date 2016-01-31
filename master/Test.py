import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron

# we create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

clf = Perceptron(n_iter=1).fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] + 3.4 / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

b = -w[0] / w[1]
yyn = b * xx - (clf.intercept_[0]) / w[1]

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(xx, yy, 'k-')
ax1.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, s=100)
ax2.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, s=100)
ax2.plot(xx, yyn, 'k-')
plt.show()
