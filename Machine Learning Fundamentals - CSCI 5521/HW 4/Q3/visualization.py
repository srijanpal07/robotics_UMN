from matplotlib import pyplot as plt
import numpy as np


def plot_boundary(clf, x, y):
    """
    Plot the decision boundary of the kernel perceptron, and the samples (using different
    colors for samples with different labels)
    """
    plt.figure()
    plt.scatter(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], c='r')
    plt.scatter(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], c='b')

    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    test_data = np.vstack([xx.ravel(), yy.ravel()]).T
    predicted = clf.predict(test_data)
    predicted = predicted.reshape(xx.shape)
    cp = plt.contour(xx, yy, predicted, [0.0], colors='k', linewidths=1, origin='lower')
    plt.axis('tight')
    cp.collections[0].set_label('contour plot')

    plt.legend()
    plt.show()