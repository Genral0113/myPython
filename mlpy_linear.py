import numpy as np
import matplotlib.pyplot as plt


def load_data_txt(fname):
    """
    Load data from a txt file or csv file, data was separated by comma

    Arguments:
        fname -- the file name with full path stores the data

    Returns:
        X -- the training set
        y -- the output of training set
        m -- number of training examples
        dim -- dimension of the training examples
    """

    raw_data = np.loadtxt(fname, delimiter=',')

    m = raw_data.shape[0]  # number of examples
    dim = raw_data.shape[1]

    X = raw_data[:, 0:dim-1].reshape(m, dim-1)
    y = raw_data[:, -1].reshape(m, 1)

    assert (X.shape == (m, dim-1))
    assert (y.shape == (m, 1))

    return X, y, m, dim


def load_data_2d_txt(fname):
    """
    Load data from a txt file or csv file, data was separated by comma

    Arguments:
        fname -- the file name with full path stores the data

    Returns:
        X -- the training set
        y -- the output of training set
        m -- number of training examples
        dim -- dimension of the training examples
    """

    raw_data = np.loadtxt(fname, delimiter=',')

    m = raw_data.shape[0]  # number of examples
    dim = raw_data.shape[1]

    X = raw_data[:, 0].reshape(m, 1)
    y = raw_data[:, 1].reshape(m, 1)

    return X, y, m, dim


def compute_cost_linear(X, y, theta):
    """
    Compute the cost with linear method

    Arguments:
        X -- numpy array or matrix of training examples
        y -- numpy vector of training examples output
        theta -- the weight factor

    Returns:
        J -- the computed cost based on input theta
    """

    # number of training examples
    m = len(y)

    # compute the hypothesis value based on theta
    yhat = np.dot(X, theta)

    # compute the delta value between hypothesis and training examples output
    dy = yhat - y

    # compute the cost with gradient descent
    J = np.sum(np.power(dy, 2)/(2 * m))

    return J


def gradient_descent_linear(X, y, theta, alpha, num_iterations):
    """
    Arguments:
        X -- numpy array or matrix of training examples
        y -- numpy vector of training examples output
        theta -- the weight factor
        alpha -- learning rate
        num_iterations -- number of iterations

    Returns:
        theta -- the lowest cost point
        J_history -- cost history of iterations
    """
    # number of training examples
    m = len(y)

    # initialize the cost history
    J_history = np.zeros((num_iterations, 1))

    # iterates
    for i in range(1, num_iterations):

        # compute the hypothesis value based on theta
        yhat = np.dot(X, theta)

        # compute the delta value between hypothesis and training examples output
        dy = yhat - y

        # update theta
        theta = theta - (alpha/m) * np.dot(X.T, dy)

        # compute the cost
        J_history[i] = compute_cost_linear(X, y, theta)

    return theta, J_history


def gradient_descent_linear_multiple(X, y, theta, alpha, num_iterations):
    """
    Arguments:
        X -- numpy array or matrix of training examples
        y -- numpy vector of training examples output
        theta -- the weight factor
        alpha -- learning rate
        num_iterations -- number of iterations

    Returns:
        theta -- the lowest cost point
        J_history -- cost history of iterations
    """
    # number of training examples
    m = len(y)

    # initialize the cost history
    J_history = np.zeros((num_iterations, 1))
    temp = theta

    # iterates
    for iter in range(num_iterations):

        for i in range(len(theta)):

            # compute the hypothesis value based on theta
            yhat = np.dot(X, theta)

            # compute the delta value between hypothesis and training examples output
            dy = yhat - y

            # update theta
            temp[i, 0] = temp[i, 0] - (alpha/m) * np.sum(np.multiply(dy, X[:, i].reshape((m, 1))))
        theta = temp

        # compute the cost
        J_history[iter] = compute_cost_linear(X, y, theta)

    return theta, J_history


def feature_normalize(X):
    """
    To normalize the training examples with mean and standard deviation

    Arguments:
        X -- numpy array or matrix of training examples

    Returns:
        X_norm -- normalized X
    """

    X_norm = X

    mu = np.zeros((1, np.size(X, 1)))
    sigma = np.zeros((1, np.size(X, 1)))

    for i in range(0, np.size(X, 1)):
        mu[0:, i] = np.mean(X[:, i].reshape(m, 1))
        sigma[0:, i] = np.std(X[:, i].reshape(m, 1))

    X_norm = np.divide((X_norm - mu), sigma)

    return X_norm, mu, sigma


def normal_equation(X, y):

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

    return theta


def plot_2d_data(fname):

    X, y, m, dim = load_data_txt(fname)

    X = np.c_[np.ones((m, 1)), X]

    iterations = 1500
    alpha = 0.01

    theta = np.zeros((dim, 1))
    J = compute_cost_linear(X, y, theta)
    print('With theta = [0; 0]\nCost computed = %f\n', J)

    theta = np.array([-1, 2]).reshape(dim, 1)
    J = compute_cost_linear(X, y, theta)
    print('With theta = [-1 ; 2]\nCost computed = %f\n', J)

    theta, J_history = gradient_descent_linear(X, y, theta, alpha, iterations)
    print('Theta found by gradient descent:\n')
    print(theta)

    plt.plot(X[:,1], y, 'rx')
    plt.plot(X[:,1], np.dot(X, theta), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.show()

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(dim, 1)
            J_vals[i, j] = compute_cost_linear(X, y, t)

    J_vals = J_vals.T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx')
    plt.show()


if __name__ == '__main__':

    fname = '.\\rawdata\\ex1data1.txt'
    plot_2d_data(fname)

    fname = '.\\rawdata\\ex1data2.txt'
    X, y, m, dim = load_data_txt(fname)

    X1 = X

    X, mu, sigma = feature_normalize(X)

    X = np.c_[np.ones((m, 1)), X]

    alpha = 0.03
    num_iterations = 160
    theta = np.zeros((dim, 1))

    theta, J_history = gradient_descent_linear_multiple(X, y, theta, alpha, num_iterations)
    # plt.plot(J_history, '-b')
    # plt.xlabel('Number of iterations')
    # plt.ylabel('Cost J')
    # plt.show()

    X_pred = np.array([[1650], [3]]).reshape((1, 2))
    X_pred = np.divide((X_pred - mu), sigma)
    assert (X_pred.shape == (1, 2))
    X_pred = np.array([[1], [X_pred[0][0]], [X_pred[0][1]]]).reshape(1, 3)
    price = np.dot(X_pred, theta)
    print(' The price of a house with 1650 sq and 3 br is ' + str(price))

    X1 = np.c_[np.ones((m, 1)), X1]
    theta1 = normal_equation(X1, y)
    X_pred = np.array([[1], [1650], [3]]).reshape((1, 3))
    price1 = np.dot(X_pred, theta1)
    print(' The price of a house with 1650 sq and 3 br is ' + str(price1))
