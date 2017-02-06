import numpy as np

from scipy.optimize import fmin_cg

import matplotlib.pyplot as plt


def compute_polynomial_kernel(x1, x2, d):
    """
    Define the polynomial kernel matrix
    :param x1: the original x_train matrix of size mxn where m=#instances & n=#features
    :param x2: the other x matrix (train or test) of size m'xn where m'=#new_instances
    :param d: the degree of the polynomial
    :return: the polynomial kernel matrix of size mxm'
    """
    k = (1+np.dot(x1, x2.T))**d  # (1 + x.z)^d
    # print "original k:", k
    k = (k - k.mean(axis=0)) / k.std(axis=0)
    # print "normalized k:", k
    return k


def sigmoid(z):
    """
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    :param z: Matrix, vector or scalar
    :return: The sigmoid value
    """
    return 1.0 / (1.0 + np.exp(-z))


def avg_cross_entropy(a, k, y):
    """
    Compute the average cross entropy
    :param a: the weight vector of size mx1 where m = #instances
    :param k: the kernel matrix of size mxm
    :param y: the output vector of size mx1
    :return: the average cross entropy
    """
    m = y.shape[0]  # number of instances
    z = np.dot(a.T, k)  # a^T.K of size 1xm
    h = sigmoid(z)  # of size 1xm
    j = (-1. / m) * np.sum(y.T * np.log(h) + (1. - y.T) * np.log(1 - h))
    return j


def compute_cost(a, k, y, lambda_):
    """
    Define the cost function J(a)
    :param a: the weight vector of size mx1 where m = #instances
    :param k: the kernel matrix of size mxm
    :param y: the output vector of size mx1
    :param lambda_: the L2 regularization parameter
    :return: the cost function value
    """
    m = y.shape[0]  # number of instances
    z = np.dot(a.T, k)  # a^T.K of size 1xm
    h = sigmoid(z)  # of size 1xm
    j = (-1. / m) * np.sum(y.T * np.log(h) + (1.-y.T) * np.log(1-h)) + (lambda_/2.)*np.dot(z, a)
    return j


def compute_gradient(a, k, y, lambda_):
    """
    Define the gradient of the cost function with respect to a
    :param a: the weight vector of size mx1 where m = #instances
    :param k: the kernel matrix of size mxm
    :param y: the output vector of size mx1
    :param lambda_: the L2 regularization parameter
    :return:
    """
    m = y.shape[0]  # number of instances
    z = np.dot(a.T, k)  # a^T.K of size 1xm
    h = sigmoid(z)  # of size 1xm

    grad = (-1. / m) * np.dot(y.T - h, k) + lambda_*z
    return grad


def fit_log_reg(x, y, deg, lambd):
    """
    Train a Logistic Regression model with polynomial kernel of degree d
    :param x: the original x matrix of size mxn where m=#instances & n=#features
    :param y: the output vector of size mx1
    :param deg: the polynomial degree
    :param lambd: the L2 regularization parameter
    :return: the kernel matrix, the optimal weight vector, the optimal cost
    """

    m = x.shape[0]  # number of instances
    # n = x.shape[1]  # number of features

    k_train = compute_polynomial_kernel(x, x, deg)  # compute training kernel matrix

    initial_a = np.zeros((m, 1))  # initial parameter vector a=0 for all instances

    # init_cost = compute_cost(initial_a, k_train, y, lambd)  # initial cost
    # init_gradient = compute_gradient(initial_a, k_train, y, lambd)  # initial gradient
    # print 'Cost at initial `a`=(zeros):', cost
    # print 'Gradient at initial `a`=(zeros):', gradient
    # print gradient.shape

    def f(a):
        return compute_cost(a, k_train, y, lambd).flatten()

    def fprime(a):
        return compute_gradient(a, k_train, y, lambd).flatten()

    result = fmin_cg(f, initial_a, fprime, maxiter=40000, disp=True, full_output=True, retall=True)
    a_opt = result[0]  # optimal weights
    f_opt = result[1]  # value of function at optimal weights
    n_f_call = result[2]  # number of function calls made
    n_g_call = result[3]  # number of gradient calls made
    error = result[4]  # error code. 0 is OK.
    all_vecs = result[5]  # list of all weight vectors for all iterations

    if error == 0:
        print "Success"
    elif error == 1:
        print "Maximum number of iterations exceeded"
    elif error == 2:
        print "Gradient and/or function calls not changing"

    # print "optimal weights:", a_opt
    # print "final cost:", f_opt
    # print "number of cost evaluation:", n_f_call
    # print "number of gradient evaluation:", n_g_call
    # print "number of weight updates:", len(all_vecs)

    return k_train, a_opt, f_opt


def predict(k, a, y):
    """
    Compute the class probability.
    :param k: the kernel matrix to compute the probabilities for
    :param a: the optimal weight vector
    :param y: the true labels
    :return: the prediction classes, and the accuracy
    """
    z = np.dot(a.T, k)
    h = np.asarray(sigmoid(z))
    for i in range(len(h)):
        if h[i] >= 0.5:
            h[i] = 1
        else:
            h[i] = 0
    error = np.count_nonzero(y.flatten() - h) / float(len(y.flatten()))
    return h, 1.-error


def log_reg(x_train, y_train, x_test, y_test):
    data = {
        #TODO: 'lambda': [0, 0.1, 1., 10., 100., 1000., 10000.],
        'lambda': [0],
        'train_avg_ce': {1: [], 2: [], 3: []},  # train average cross entropy for each degree
        'test_avg_ce': {1: [], 2: [], 3: []},  # test average cross entropy for each degree
        'train_accuracy': {1: [], 2: [], 3: []},  # train accuracy for each degree
        'test_accuracy': {1: [], 2: [], 3: []},  # test accuracy for each degree
    }

    print "\nKernelized Logistic Regression on data (with L2 regularization)..."
    for l in data['lambda']:
        for deg in [1, 2, 3]:
            print "\nlambda:", l, "& degree:", deg
            k_train, a_opt, train_cost = fit_log_reg(x_train, y_train, deg, l)
            k_test = compute_polynomial_kernel(x_train, x_test, deg)
            print "k_train shape", k_train.shape
            print "k_test shape", k_test.shape

            train_avg_ce = avg_cross_entropy(a_opt, k_train, y_train)
            print "training avg ce:", train_avg_ce
            data['train_avg_ce'][deg].append(train_avg_ce)

            test_avg_ce = avg_cross_entropy(a_opt, k_test, y_test)
            print "testing avg ce:", test_avg_ce
            data['test_avg_ce'][deg].append(test_avg_ce)

            train_predictions, train_acc = predict(k_train, a_opt, y_train)
            # print "train predictions:", train_predictions
            # print "true predictions:", y_train.flatten()
            print "train accuracy:", train_acc
            data['train_accuracy'][deg].append(train_acc)

            test_predictions, test_acc = predict(k_test, a_opt, y_test)
            # print "test predictions:", test_predictions
            # print "true predictions:", y_test.flatten()
            print "test accuracy:", test_acc
            data['test_accuracy'][deg].append(test_acc)

    return data


def plot(data):
    ###
    # PLOT TRAIN & TEST AVERAGE CROSS ENTROPY ERROR
    ###
    plt.title('QUESTION 2.c)\nTrain & Test Average Cross Entropy')
    train_avg_ce = [data['train_avg_ce'][d][0] for d in [1,2,3]]
    print "train avg ce:", train_avg_ce
    train_line, = plt.plot([1, 2, 3], train_avg_ce, 'bo-')
    test_avg_ce = [data['test_avg_ce'][d][0] for d in [1, 2, 3]]
    print "test avg ce:", test_avg_ce
    test_line, = plt.plot([1, 2, 3], test_avg_ce, 'ro-')
    plt.legend([train_line, test_line], ["train", "test"], loc='lower right')
    plt.xlabel('degree')
    plt.ylabel('average cross entropy')
    plt.show()

    ###
    # PLOT TRAIN & TEST ACCURACIES
    ###
    plt.title('QUESTION 2.c)\nTrain & Test Accuracy')
    train_acc = [data['train_accuracy'][d][0] for d in [1, 2, 3]]
    print "train accuracy:", train_acc
    train_line, = plt.plot([1, 2, 3], train_acc, 'bo-')
    test_acc = [data['test_accuracy'][d][0] for d in [1, 2, 3]]
    print "test accuracy:", test_acc
    test_line, = plt.plot([1, 2, 3], test_acc, 'ro-')
    plt.legend([train_line, test_line], ["train", "test"], loc='lower right')
    plt.xlabel('degree')
    plt.ylim(0, 1)
    plt.ylabel('accuracy')
    plt.show()

    """
    # UNCOMMENT TO PLOT WITH REGULARIZATION ON X-AXIS
    # WARNING: REGULARIZATION WITH KERNEL OF DEGREE 2 WILL RESULT IN WEIRD BEHAVIOR LIKE -INF WEIGHTS AND NAN CROSS-ENTROPY

    ###
    # PLOT TRAIN & TEST AVERAGE CROSS ENTROPY ERROR
    ###
    colors = ['b', 'r', 'g']
    assert len(colors) == len(data['train_avg_ce'])
    lines = []
    labels = []
    plt.title('QUESTION 2.c)\nTrain & Test Average Cross Entropy')
    for i, (deg, avg_ce) in enumerate(data['train_avg_ce'].iteritems()):
        print "degree:", deg, "train avg ce:", avg_ce
        line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], avg_ce, colors[i] + 'o-')
        lines.append(line)
        labels.append('train; d=%s' % deg)
    for i, (deg, avg_ce) in enumerate(data['test_avg_ce'].iteritems()):
        print "degree:", deg, "test avg ce:", avg_ce
        line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], avg_ce, colors[i] + 'o--')
        lines.append(line)
        labels.append('test; d=%s' % deg)
    plt.legend(lines, labels)
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('average cross entropy')
    plt.show()

    ###
    # PLOT TRAIN & TEST ACCURACIES
    ###
    colors = ['b', 'r', 'g']
    assert len(colors) == len(data['train_accuracy'])
    lines = []
    labels = []
    plt.title('QUESTION 2.c)\nTrain & Test Accuracies')
    for i, (deg, acc) in enumerate(data['train_accuracy'].iteritems()):
        print "degree:", deg, "train accuracy:", acc
        line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], acc, colors[i] + 'o-')
        lines.append(line)
        labels.append('train; d=%s' % deg)
    for i, (deg, acc) in enumerate(data['test_accuracy'].iteritems()):
        print "degree:", deg, "test accuracy:", acc
        line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], acc, colors[i] + 'o--')
        lines.append(line)
        labels.append('test; d=%s' % deg)
    plt.legend(lines, labels, loc='lower right')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylim(0, 1)
    plt.ylabel('accuracy')
    plt.show()
    """
