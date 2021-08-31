import numpy as np
import numpy.random
import pylab

from mlpy import *


def linear_forward_test_case():
    np.random.seed(1)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)

    return A, W, b


def linear_activation_forward_test_case():
    np.random.randn(2)
    A_prev = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.rand(1, 1)

    return A_prev, W, b


def L_model_forward_test_case():
    np.random.seed(1)
    X = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return X, parameters


def linear_backward_test_case():
    np.random.seed(1)
    dZ = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.rand(1, 1)
    linear_cache = (A, W, b)
    return dZ, linear_cache


def compute_cost_deep_test_case():
    Y = np.array([[1, 1, 1]])
    AL = np.array([[.8, .9, 0.4]])

    return Y, AL


def linear_activation_backward_test_case():
    np.random.seed(2)

    dA = np.random.randn(1, 2)
    A = np.random.randn(3, 2)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    Z = np.random.randn(1, 2)

    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    return dA, linear_activation_cache


def L_model_backward_test_case():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    return AL, Y, caches


def update_parameters_deep_test_case():
    np.random.seed(2)
    W1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    W2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3, 4)
    db1 = np.random.randn(3, 1)
    dW2 = np.random.randn(1, 3)
    db2 = np.random.randn(1, 1)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return parameters, grads


"""
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate = " + str(d["learning_rate"])
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is:" + str(i)
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iteratios = 1500, learning_rate = i, print_cost = False)
    print('\n' + "---------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legent = plt.legent(loc='upper center', shadow=True)
frame = legent.get_frame()
frame.set_facecolor('0.90')
plt.show()

my_image = "my_image.jpg"
fname = "images" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["w"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
"""


if __name__ == '__main__':

    # print('-----------------------------------------------------------------------------------------------------------')
    # t_x = 1
    # print("Input x is " + str(t_x))
    # print("Basic sigmoid function of scalar input sigmoid(x) = " + str(basic_sigmoid(t_x)))
    # print("Basic tanh function of scalar input tanh(x) = " + str(basic_tanh(t_x)))

    # print('-----------------------------------------------------------------------------------------------------------')
    # t_x = np.array([1, 2, 3, 4, 5])
    # print("Input x is " + str(t_x))
    # print("Sigmoid function of array input sigmoid(x) = " + str(sigmoid(t_x)))
    # print("Tanh function of array input tanh(x) = " + str(tanh(t_x)))
    #
    # print("The derivative of sigmoid(x) = " + str(sigmoid_derivative(t_x)))
    #
    # t_image = np.array([[[0.67826139, 0.29380381],
    #                      [0.90714982, 0.52835647],
    #                      [0.4215251, 0.45017551]],
    #                     [[0.92814219, 0.96677647],
    #                      [0.85304703, 0.52351845],
    #                      [0.19981397, 0.27417313]],
    #                     [[0.60659855, 0.00533165],
    #                      [0.10820313, 0.49978937],
    #                      [0.34144279, 0.94630077]]])
    # # plt.imshow(t_image)
    # print(image2vector(t_image))
    #
    # t_x = np.array([[0, 3, 4],
    #                 [1, 6, 4]])
    # print("NormalizeRows(x) = " + str(normalize_rows(t_x)))
    #
    # t_x = np.array([[9, 2, 5, 0, 0],
    #                 [7, 5, 0, 0, 0]])
    # print("softmax(x) = " + str(softmax(t_x)))
    #
    # yhat = np.array([.9, 0.2, 0.1, .4, .9])
    # y = np.array([1, 0, 0, 1, 1])
    # print("L1 = " + str(L1(yhat, y)))
    # print("L2 = " + str(L2(yhat, y)))
    #
    # w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1, 2],[3, 4]]), np.array([[1, 0]])
    # grads, cost = propagate(w, b, X, Y)
    # print("dw = " + str(grads["dw"]))
    # print("db = " + str(grads["db"]))
    # print("cost = " + str(cost))
    #
    # params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    # print("w = " + str(params["w"]))
    # print("b = " + str(params["b"]))
    # print("dw = " + str(grads["dw"]))
    # print("db = " + str(grads["db"]))
    #
    # print("predictions = " + str(predict(w, b, X)))

    # print('-----------------------------------------------------------------------------------------------------------')
    # m = 400
    # X, Y = load_planar_dataset(m)
    # print("the shape of X is:" + str(X.shape))
    # print("the shape of Y is:" + str(Y.shape))
    # print("the training examples:" + str(X.shape[1]))
    # # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    # # plt.show()

    # print('-----------------------------------------------------------------------------------------------------------')
    # m = 400
    # X, Y = load_planar_dataset(m)
    # clf = sklearn.linear_model.LogisticRegressionCV()
    # clf.fit(X.T, Y.T)
    # plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    # # print accuracy
    # LR_predictions = clf.predict(X.T)
    # print('Accuracy of logistic regression : %d' % float(
    #     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
    #       '% ' + "(percentage of correctly labelled datapoints)")

    # print('-----------------------------------------------------------------------------------------------------------')
    # m = 400
    # X, Y = load_planar_dataset(m)
    # parameters = nn_model_2(X, Y, n_h=m, num_iteration=10000, print_cost=True)
    # # plot the decision boundary
    # plot_decision_boundary(lambda x: nn_predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    # pylab.show()
    # predictions = nn_predict(parameters, X)
    # print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T))/float(Y.size) * 100) + '%')

    # print('-----------------------------------------------------------------------------------------------------------')
    # m = 400
    # X, Y = load_planar_dataset(m)
    # plt.figure(figsize=(16, 32))
    # hidden_layer_size = [1, 2, 3, 4, 5, 20, 50]
    # for i, n_h in enumerate(hidden_layer_size):
    #     plt.subplot(5, 2, i+1)
    #     plt.title('Hidden layer of size %d' % n_h)
    #     parameters = nn_model(X, Y, n_h, num_iteration=5000,print_cost=True)
    #     plot_decision_boundary(lambda x: nn_predict(parameters, x.T), X, Y)
    #
    #     predictions = nn_predict(parameters, X)
    #     accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T))/float(Y.size) * 100)
    #     print("Accuracy for {} hidden units : {} %".format(n_h, accuracy))
    #     pylab.show()

    # print('-----------------------------------------------------------------------------------------------------------')
    # A, W, b = linear_forward_test_case()
    # Z, linear_cache = linear_forward(A, W, b)
    # print('Z = ' + str(Z))

    # print('-----------------------------------------------------------------------------------------------------------')
    # A_prev, W, b = linear_activation_forward_test_case()
    # A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
    # print("With sigmoid: A = " + str(A))
    # A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
    # print("With relu: A = " + str(A))

    # print('-----------------------------------------------------------------------------------------------------------')
    # X, parameters = L_model_forward_test_case()
    # AL, caches = L_model_forward(X, parameters)
    # print('AL = ' + str(AL))
    # print('Lenght of caches list = ' + str(len(caches)))

    # print('-----------------------------------------------------------------------------------------------------------')
    # Y, AL = compute_cost_deep_test_case()
    # print('cost = ' + str(compute_cost_deep(AL, Y)))

    # print('-----------------------------------------------------------------------------------------------------------')
    # dZ, linear_cache = linear_backward_test_case()
    # dA_prev, dW, db = linear_backward(dZ, linear_cache)
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db))

    # print('-----------------------------------------------------------------------------------------------------------')
    # AL, linear_activation_cache = linear_activation_backward_test_case()
    # dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")
    # print("sigmoid:")
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db) + '\n')
    # dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="relu")
    # print("relu:")
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db))

    # print('-----------------------------------------------------------------------------------------------------------')
    # AL, Y_access, caches = L_model_backward_test_case()
    # grads = L_model_backward(AL, Y_access, caches)
    # print("dW1 = " + str(grads["dW1"]))
    # print("db1 = " + str(grads["db1"]))
    # print("dA1 = " + str(grads["dA1"]))

    # print('-----------------------------------------------------------------------------------------------------------')
    # parameters, grads = update_parameters_deep_test_case()
    # parameters = update_parameters_deep(parameters, grads, 0.1)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # print('-----------------------------------------------------------------------------------------------------------')
    # a = np.arange(10)
    # print(a)
    # a = a.reshape(10, 1)
    # print(np.linalg.norm(a))

    fname = ".\\rawdata\\ex1data1.txt"
    raw_data = np.loadtxt(fname, delimiter=',')
    m = raw_data.shape[0]
    X = raw_data[:, 0].reshape((m, 1))
    y = raw_data[:, 1].reshape((m, 1))

    plt.plot(X, y, 'rx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()