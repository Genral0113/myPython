import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from mlpy_linear import load_data_txt


def plot_data(X, y):
    """
    Plot 2-dimension dataset with positive(1) or negative(0) output y

    Argument:
        X -- 2-dimension dataset
        y -- output value mapped to the 2-dimension dataset

    Return:
        nil
    """
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='x')


def sigmoid(x):
    """
    Compute sigmoid of x

    Argument:
        x -- A scalar or numpy array of any size

    Return:
        s -- sigmoid(x)
    """

    s = 1.0 / (1 + np.exp(-x))

    return s


def cost_function_logistic(theta, X, y):
    """
    Compute cost and gradient with logistic regression

    Argument:
        theta -- A scalar or numpy array of n*1 shape
        X -- A scalar or numpy array of m*n shape
        y -- A scalar or numpy array of m*1 shape

    Return:
        J -- the cost
        grad -- the gradient
    """
    m = len(y)

    h = sigmoid(X.dot(theta))

    J = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h)))/m

    grad = np.dot((h - y).T, X)/m

    return J, grad


def gradient_logistic(theta, X, y):
    m, n = X.shape
    assert (theta.shape() == (n, 1))
    grad = np.dot(X.T, sigmoid(X.dot(theta)) - y)/m
    return grad.flatten()


def map_feature(X1, X2):

    degree = 6
    m = 1

    if isinstance(X1, np.ndarray):
        m = X1.shape[0]

    out = np.ones((m, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            tmp1 = np.power(X1, (i - j))
            tmp2 = np.power(X2, j)
            tmp = np.multiply(tmp1, tmp2)
            out = np.c_[out, tmp]

    return out


def plot_decision_boundary(theta, X, y):

    plot_data(X[:, [1, 2]], y)

    if X.shape[1] <= 3:
        plot_x = [np.min(X[:, 1])-2, np.max(X[:, 1])+2]
        plot_y = np.multiply(np.divide(-1, theta[2]), (np.multiply(theta[1], plot_x) + theta[0]))

        plt.plot(plot_x, plot_y)

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(map_feature(u[i], v[j]), theta)

        plt.contour(u, v, z)


if __name__ == '__main__':

    fname = r'.\rawdata\ex2data1.txt'

    X, y, m, dim = load_data_txt(fname)
    X = np.c_[np.ones((m, 1)), X]

    initial_theta = np.zeros((X.shape[1], 1))

    cost, grad = cost_function_logistic(initial_theta, X, y)
    print('Cost at initial theta (zeros): %f\n', cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): \n')
    print(' %f \n', grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    test_theta = np.array([[-25.161], [0.2062], [0.2015]])

    cost, grad = cost_function_logistic(test_theta, X, y)
    print('Cost at initial theta (zeros): %f\n', cost)
    print('Gradient at initial theta (zeros): \n')
    print(' %f \n', grad)

    # result = op.minimize(fun=cost_function_logistic, x0=initial_theta, args=(X, y), method='TNC', jac=gradient_logistic)
    # print(result)

    plot_decision_boundary(test_theta, X, y)
    plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'], loc='upper right')
    plt.show()

