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


def cost_function_logistic_regularization(theta, X, y, lam):
    """
    Compute cost and gradient for logistic regression with regularization

    Argument:
        theta -- A scalar or numpy array of n*1 shape
        X -- A scalar or numpy array of m*n shape
        y -- A scalar or numpy array of m*1 shape
        lam -- lambda

    Return:
        J -- the cost
        grad -- the gradient
    """
    m = len(y)
    J = 0
    grad = np.zeros((1, theta.shape[0]))

    h = sigmoid(X.dot(theta))
    theta[0,0] = 0
    J = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h)))/m + lam/2/m*np.sum(np.power(theta, 2))

    grad = np.dot((h - y).T, X)/m + lam/m*theta.T

    return J, grad


def gradient_logistic(theta, X, y):
    m, n = X.shape
    assert (theta.shape() == (n, 1))
    grad = np.dot(X.T, sigmoid(X.dot(theta)) - y)/m
    return grad.flatten()


def map_feature(X1, X2, degree=6):
    """
    Feature mapping function to polynomial features
    Maps the two input feature to quadratic features used in regularization logistic regression

    Argument:
        X1 -- input features with shape (m*1)
        X2 -- input features with shape (m*1)
        degree -- polynomial degree, 6 is the default

    Return:
        out -- polynomial features with shape (m*28)(28 is the default), comprising of
        1, X1, X2, X1^2, X2^2, X1^2*X2, X1*X2^2, etc ...
    """
    # compute the dimension of output array
    dim = np.sum(range(1, degree + 2))

    # retrieve the number of input samples
    m = 1
    if isinstance(X1, np.ndarray):
        m = X1.shape[0]

    # define the output array
    out = np.ones((m, dim))

    # compute the polynomial features
    k = 0
    for i in range(1, degree + 1):
        for j in range(i + 1):
            k += 1
            out[:, k] = np.multiply(np.power(X1, (i - j)), np.power(X2, j))

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

        z = z.T

        plt.contour(u, v, z, 0)


def predict(theta, X):
    m = X.shape[0]

    p = np.zeros((m, 1))

    v = sigmoid(X.dot(theta))

    for i in range(m):
        if v[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p


if __name__ == '__main__':

    # fname = r'.\rawdata\ex2data1.txt'

    # X, y, m, dim = load_data_txt(fname)
    # X = np.c_[np.ones((m, 1)), X]

    # initial_theta = np.zeros((X.shape[1], 1))

    # cost, grad = cost_function_logistic(initial_theta, X, y)
    # print('Cost at initial theta (zeros): %f\n', cost)
    # print('Expected cost (approx): 0.693\n')
    # print('Gradient at initial theta (zeros): \n')
    # print(' %f \n', grad)
    # print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    # test_theta = np.array([[-25.161], [0.2062], [0.2015]])

    # cost, grad = cost_function_logistic(test_theta, X, y)
    # print('Cost at initial theta (zeros): %f\n', cost)
    # print('Gradient at initial theta (zeros): \n')
    # print(' %f \n', grad)

    # result = op.minimize(fun=cost_function_logistic, x0=initial_theta, args=(X, y), method='TNC', jac=gradient_logistic)
    # print(result)

    # plot_decision_boundary(test_theta, X, y)
    # plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'], loc='upper right')
    # plt.show()

    # X1 = np.array([[1, 45, 85]])
    # prob = sigmoid(X1.dot(test_theta))
    # print('For a student with scores 45 and 85, we predict an admission probability of %f', prob)

    # p = predict(test_theta, X)
    # p1 = np.zeros((m, 1))
    # for i in range(m):
    #     if p[i, 0] == y[i, 00]:
    #         p1[i, 0] = 1
    #     else:
    #         p1[i, 0] = 0
    # print(np.mean(p1) * 100)

    fname = r'.\rawdata\ex2data2.txt'
    X, y, m, dim = load_data_txt(fname)

    # plot_data(X, y)
    # plt.legend(['y = 1 ', 'y = 0'])
    # plt.show()

    X = map_feature(X[:, 0], X[:, 1])

    initial_theta = np.zeros((X.shape[1], 1))
    lam = 1

    cost, grad = cost_function_logistic_regularization(initial_theta, X, y, lam)

    theta = [[1.273], [0.62488], [1.1774], [-2.0201],
             [-0.91262], [-1.4299], [0.12567], [-0.36855],
             [-0.36003], [-0.17107], [-1.4609], [-0.052499],
             [-0.61889], [-0.27375], [-1.1923], [-0.24099],
             [-0.20793], [-0.047224], [-0.27833], [-0.2966],
             [-0.45396], [-1.0455], [0.026463], [-0.29433],
             [0.014381], [-0.3287], [-0.1438], [-0.92488]]
    plot_decision_boundary(theta, X, y)
    plt.title('lambda = %f' % lam)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.show()
