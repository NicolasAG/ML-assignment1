import argparse
from datetime import datetime as dt

import numpy as np

import a1q1d
import a1q1e
import a1q1f
import a1q1g
import a1q2


###
# QUESTION 1.a)
# LOADING DATASET
###
def load_data(normalize=True):
    print "\nLoading data..."
    with open('./hw1x.dat', 'rb') as file_x:
        rows = [map(float, line.split()) for line in file_x]
    x = np.array(rows)

    with open('./hw1y.dat', 'rb') as file_y:
        rows = [map(float, line.split()) for line in file_y]
    y = np.array(rows)

    if normalize:  # Normalize the data
        x = (x - x.mean(axis=0)) / x.std(axis=0)

    # Add one extra column for the bias term:
    # x = np.hstack((np.ones((x.shape[0], 1)), x))
    # No need to do it: scikit learn adds a bias column itself!

    print "x =", x.shape
    print "y =", y.shape
    return x, y


###
# QUESTION 1.b)
# SPLIT DATA into 80% train 20% test
###
def split_data(x, y, test=0.2):
    print "\nSplitting data into %.2f train %.2f test..." % (1-test, test)
    seed = dt.now().microsecond
    np.random.seed(seed)
    np.random.shuffle(x)  # shuffle matrix X
    np.random.seed(seed)
    np.random.shuffle(y)  # shuffle vector y with same seed as before to maintain valid labels

    split = int(x.shape[0] * test)
    x_test = x[:split]
    x_train = x[split:]
    y_test = y[:split]
    y_train = y[split:]

    print "x_train =", x_train.shape
    print "x_test =", x_test.shape
    print "y_train =", y_train.shape
    print "y_test =", y_test.shape

    return x_train, y_train, x_test, y_test

###
# QUESTION 1.c)
# Objective of logistic regression with L2 regularization:
# - J_D(W) = - (sum from i=0 to m {y_i log h(x_i) + (1 - y_i) log(1 - h(x_i)) }) + lambda/2 * ||W||^2
# with h(x) being our objective function: h(x) = 1 / (1 + exp{W^t x})
###


def main():
    def my_bool(p):
        return p == '1' or p.lower() == 'true' or p.lower() == 'yes'

    parser = argparse.ArgumentParser(description='COMP 652 - Machine Learning - Assignment 1')
    parser.add_argument('--q1d', action="store_true", help='produce only plots for q1.d)')
    parser.add_argument('--q1f', action="store_true", help='produce only plots for q1.f)')
    parser.add_argument('--q1g', action="store_true", help='produce only plots for q1.g)')
    parser.add_argument('--q2', action="store_true", help='produce only plots for q2.c)')
    parser.add_argument('--normalize', type=my_bool, default=True, help='normalize the X matrix when loading it')
    parser.add_argument('--use_sgd', type=my_bool, default=True, help='run logistic regression using stochastic gradient descent')
    parser.add_argument('--n_iter', type=int, default=10000, help='number of iterations for SGD, or max number of iteration for LogReg')
    args = parser.parse_args()
    print args

    plot_all = not args.q1d and not args.q1f and not args.q1g and not args.q2

    x, y = load_data(normalize=args.normalize)
    x_train, y_train, x_test, y_test = split_data(x, y, 0.2)

    data = None  # data returned after performing logistic regression in q1.d)
    sigmas = [0.1, 0.5, 1, 5, 10]  # used for q1.e)f)
    means = np.linspace(-10, 10, len(sigmas))  # used for q1.e)
    gaussian_data = None  # data returned after performing logistic regression in q1.f)

    if plot_all or args.q1d:
        ###
        # QUESTION 1.d)
        # - Run logistic regression using L2 regularization (lambda = {0, 0.1, 1, 10, 100, 1000})
        # - Plot average (over all instances) cross-entropy for training & testing data as a function of lambda (log scale).
        # - Plot the L2 norm of the weight vector.
        # - Plot the actual values of the weights (one curve per weight).
        # - Plot the accuracy on the training and test set.
        # Explain briefly what you see.
        ###
        data = a1q1d.log_reg(x_train, y_train, x_test, y_test, args.n_iter, args.use_sgd)
        a1q1d.plot(data)

    if plot_all or args.q1f:
        # NOTE: need data produced by q1d for plots in q1f
        if data is None:
            data = a1q1d.log_reg(x_train, y_train, x_test, y_test, args.n_iter, args.use_sgd)
        ###
        # QUESTION 1.e)
        # - Re-format data by feeding each of the input variables (except the bias term) through a set of Gaussian basis functions:
        # - use 5 univariate basis functions with means evenly spaced between -10 and 10 and variance sigma in [0.1, 0.5, 1, 5, 10]
        ###
        gaussian_data = a1q1e.reformat(x_train, x_test, sigmas, means)

        ###
        # QUESTION 1.f)
        # Logistic Regression with no regularization
        # - Plot training and testing error as a function of sigma
        # - Add constant lines showing the training and testing error you had obtained in part d
        # Explain how sigma influences overfitting and the bias-variance trade-off
        ###
        gaussian_data = a1q1f.log_reg(gaussian_data, y_train, y_test, sigmas, args.n_iter, args.use_sgd)
        a1q1f.plot(gaussian_data, sigmas, data)

    if plot_all or args.q1g:
        # NOTE: need gaussian_data produced by q1e and q1f for plots in q1g
        if gaussian_data is None:
            gaussian_data = a1q1e.reformat(x_train, x_test, sigmas, means)
            gaussian_data = a1q1f.log_reg(gaussian_data, y_train, y_test, sigmas, args.n_iter, args.use_sgd)
        ###
        # QUESTION 1.g)
        # Add all basis functions & regularized regression with lambda in [0, 0.1, 1, 10, 100, 1000, 10000]
        # - Plot the average cross-entropy error for training and testing data, as a function of lambda (log scale)
        # - Plot the L2 norm of the weight vector you obtain.
        # - Plot the L2 norm of the weights for the set of basis functions corresponding to each value of sigma,
        #   as a function of lambda (this will be a graph with 5 lines on it)
        # Explain briefly the results.
        ###
        full_x_train, full_x_test = a1q1g.build_feature_matrices(gaussian_data, sigmas)
        data = a1q1g.log_reg(full_x_train, full_x_test, sigmas, y_train, y_test, args.n_iter, args.use_sgd)
        a1q1g.plot(data)

    if plot_all or args.q2:
        ###
        # Implementation your algorithm using a polynomial kernel: K(x, z)=(x.z+1)^d
        # Experiment with plain & polynomial logistic regression with d = 1, 2, 3
        # - Plot the training and testing cross-entropy
        # - Plot the training and testing accuracy
        ###
        data = a1q2.log_reg(x_train, y_train, x_test, y_test)
        a1q2.plot(data)


if __name__ == '__main__':
    main()
