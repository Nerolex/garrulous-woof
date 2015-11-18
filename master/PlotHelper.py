import matplotlib.pyplot as plt
import numpy as np


def contour(clf, rangeX, rangeY, spacing=0.02):
    """
    Helper function that helps plotting a contourplot.

    @param clf:
    @param rangeX:
    @param rangeY:
    @param spacing:
    @return:
    """
    x_min, x_max = rangeX[0], rangeX[1]
    y_min, y_max = rangeY[0], rangeY[1]
    xx1, yy1 = np.meshgrid(np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing))
    Z = clf.predict(np.c_[xx1.ravel(), yy1.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, yy1, Z, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
