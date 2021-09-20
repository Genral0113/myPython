import numpy as np
import math
import matplotlib.pyplot as plt


# from lr_utils import load_dataset


def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Argument:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + math.exp(-x))

    return s


def basic_tanh(x):
    """
    Compute tanh of x.

    Argument:
    x -- A scalar

    Return:
    s -- tanh(x)
    """

    s = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    return s


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


def tanh(x):
    """
    Compute tanh of x

    Argument:
    x -- A scalar or numpy array of any size

    Return:
    s -- tanh(x)
    """

    s = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return s


def sigmoid_deep(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape of Z
    cache --  returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))

    cache = Z

    return A, cache


def relu_deep(Z):
    """
    Implements the RELU activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape of Z
    cache --  returns Z as well, useful during backpropagation
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z

    return A, cache


def sigmoid_derivative(x):
    """
    Compute the gradient(also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Argument:
    x -- A scalar or numpy array of any size

    Return:
    ds -- Your computed gradient
    """

    s = sigmoid(x)
    ds = s * (1 - s)

    return ds


def image2vector(image):
    """
    Argument:
    image -- A numpy array of shape (length, height, depth)

    Return:
    v -- a vector of shape (length*height*depth, 1)
    """

    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

    return v


def normalize_rows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have until length)

    Argument:
    x -- A numpy matrix of shape (n, m)

    Return:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x
    """

    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)

    x = x / x_norm

    return x


def softmax(x):
    """
    Calculates the softmax of each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Return:
    s -- A numpy matrix equal to the softmax of x, shape (n, m)
    """

    x_exp = np.exp(x)

    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    s = x_exp / x_sum

    return s


def L1(yhat, y):
    """
    Argument:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Return:
    loss -- the value of the L1 loss function defined above
    """

    loss = np.sum(abs(yhat - y))

    return loss


def L2(yhat, y):
    """
    Argument:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Return:
    loss -- the value of the L2 loss function defined above
    """

    loss = np.sum(np.square(yhat - y))

    return loss


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) fow w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want ( or number of parameters in this case)

    Return:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))

    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation.

    Argument:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if not-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]

    # forward propagation (from X to cost)
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # backward propagation (to find grad)
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm.

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if not-cat, 1 if cat) of size (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve
    """

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using the learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" prediction the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0, i] to actual predictions p[0, i]
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_train)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimized()
    print_cost -- set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # predict test/train set example
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # print train/test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iteration": num_iterations}

    return d


def load_planar_dataset(m):
    """
    Arguments:
    m -- training samples

    Returns:
    X -- a matrix with shape (m, 2)
    Y --  a matrix with shape(m, 1)
    """

    np.random.seed(1)
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1))
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


# https://blog.csdn.net/liyinhua123456/article/details/78484362
def plot_decision_boundary(model, X, y):
    # set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    # generate a grid of points with distance h between them
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the contour and training examples
    plt.contour(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.reshape(X[0, :].shape), cmap=plt.cm.Spectral)
    # plt.show()


def layer_size(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """

    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parms -- python dictionary containing your parameters:
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# https://blog.csdn.net/ljp1919/article/details/78175918
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimension of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL"
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l - 1])
        b1 -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activation from previous layer (or input data): (size of previous layer, number of examples)
    W -- weight matrix: numpy array of shape (size of current layer, size of previous layer)
    b - bias vector, numpy array of shape (size of current layer, 1)

    Returns:
    Z -- the input the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W", "b"; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weight matrix: numpy array of shape (size of current layer, size of previous layer)
    b - bias vector, numpy array of shape (size of current layer, 1)
    activation -- the activation to be used in his layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A --  the output of the activation functions, also called pre-activation value
    cache --  a python dictionary containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        # Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid_deep(Z)
    elif activation == "relu":
        # Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_deep(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters - output of initialize_parameters_deep()

    Returns:
    AL - last post-activation value
    caches - list of caches containing:
         every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L02)
         the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # implement [LINEAR -> RELU]*(L-1). Add "cache" to the "cache" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)

    # implement LINEAR->SIGMOID. Add "cache" to the "cache" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def forward_propagation(X, parameters):
    """
    Arguments:
    X -- input date of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- the sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost give in equation (13)

    Arguments:
    A2 -- the sigmoid output of the second activation, of shape (1, number of examples)
    Y -- true labels vector of shape (1, number of examples)

    Returns:
    Cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m
    cost = np.squeeze(cost)

    assert (isinstance(cost, float))

    return cost


def compute_cost_deep(AL, Y):
    """
    Implement the cost function defined by equation (7)

    Arguments:
    AL -- probability vector corresponding to your label prediction, shape (1, number of examples)
    Y -- true "label" vector (from example: containing 0 if non-cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # compute loss from al and y
    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)), axis=1, keepdims=True)/m

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer 1)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer 1)
    cache --  tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer 1), same shape as W
    db -- Gradient of the cost with respect to b (current layer 1), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer 1
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation --  the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer 1), same shape as W
    db -- Gradient of the cost with respect to b (current layer 1), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect Z
    """

    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object

    # when z <= 0, you should set dZ to 0 as well
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect Z
    """

    Z = cache

    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instruction above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Update parameters using the gradient descent update rule give above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    # retrieve each parameters from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # update rule for each parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iteration=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iteration -- number of iterations in gradient descent loop
    print_cost -- if true, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iteration):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print("cost after iteration %i:%f" % (i, cost))

    return parameters


def nn_model_2(X, Y, n_h, num_iteration=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iteration -- number of iterations in gradient descent loop
    print_cost -- if true, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    layer_dims = [n_x, n_h, n_y]
    parameters = initialize_parameters_deep(layer_dims)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iteration):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print("cost after iteration %i:%f" % (i, cost))

    return parameters


def nn_predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns:
    prediction -- vector of predictions of our model (red: 0 / blue: 1)
    """
    A2, cache = forward_propagation(X, parameters)
    prediction = (A2 > 0.5)

    return prediction


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)                     # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)             # after this line, Y is the same shape as AL

    # initializing the backpropagation
    dAL = -((np.divide(Y, AL)) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradient. Input: "AL, Y, caches". Output: "grads["dAL"], grads["dWL"], grads["dbL"]"
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    print(L)
    for l in reversed(range(L - 1)):
        print(l)
        # lth layer: (RELU -> LINEAR) gradients.
        # inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)]"
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters_deep(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters


if __name__ == '__main__':
    pass