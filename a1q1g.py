import numpy as np

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

###
# QUESTION 1.g)
# Add all basis functions & regularized regression with lambda in [0, 0.1, 1, 10, 100, 1000, 10000]
# - Plot the average cross-entropy error for training and testing data, as a function of lambda (log scale)
# - Plot the L2 norm of the weight vector you obtain.
# - Plot the L2 norm of the weights for the set of basis functions corresponding to each value of sigma,
#   as a function of lambda (this will be a graph with 5 lines on it)
# Explain briefly the results.
###


def build_feature_matrices(gaussian_data, sigmas):
    print "\nBuilding big X matrices..."
    big_x_train = np.asarray([])
    big_x_test = np.asarray([])
    for sigma in sigmas:
        # Concatenate all train matrices on horizontal axis. Note that there is no bias term in X since added by sklearn.
        if big_x_train.size:
            big_x_train = np.hstack((big_x_train, gaussian_data[str(sigma)]['x_train']))
        else:
            big_x_train = gaussian_data[str(sigma)]['x_train']
        # Concatenate all test matrices on horizontal axis (except for 1st  column: bias term)
        if big_x_test.size:
            big_x_test = np.hstack((big_x_test, gaussian_data[str(sigma)]['x_test']))
        else:
            big_x_test = gaussian_data[str(sigma)]['x_test']
    # Insert again bias column. No need!
    # full_x_train = np.ones((big_x_train.shape[0], big_x_train.shape[1] + 1))
    # full_x_test = np.ones((big_x_test.shape[0], big_x_test.shape[1] + 1))
    # full_x_train[:, 1:] = big_x_train
    # full_x_test[:, 1:] = big_x_test
    print "big x_train:", big_x_train.shape
    print "big x_test:", big_x_test.shape

    return big_x_train, big_x_test


def log_reg(full_x_train, full_x_test, sigmas, y_train, y_test, n_iter=10000, use_sgd=True):
    data = {
        'lambda': [0, 0.1, 1., 10., 100., 1000., 10000.],
        'train_avg_ce': [],
        'test_avg_ce': [],
        'l2_norm': [],
        'l2_norm_per_sigma': {},
        'train_accuracy': [],
        'test_accuracy': []
    }

    print "\nLogistic Regression on data applied to gaussian basis functions (with L2 regularization)..."
    for l in data['lambda']:
        if l == 0:
            if use_sgd: logreg = SGDClassifier(loss='log', penalty='none', n_iter=n_iter, shuffle=False)
            else: logreg = LogisticRegression(penalty='l2', C=1.e12, max_iter=n_iter)
        else:
            if use_sgd: logreg = SGDClassifier(loss='log', penalty='l2', alpha=l, n_iter=n_iter, shuffle=False)
            else: logreg = LogisticRegression(penalty='l2', C=1./l, max_iter=n_iter)
        logreg.fit(full_x_train, y_train.flatten())

        # L2 norm of all weights
        data['l2_norm'].append(np.sum(logreg.coef_[0]**2))

        # L2 norm of specific weights for each sigma gaussian function
        for i, sigma in enumerate(sigmas):
            weights = logreg.coef_[0][1+i*1565: 1+(i+1)*1565]
            l2_norm = np.sum(weights**2)
            if str(sigma) in data['l2_norm_per_sigma']:
                data['l2_norm_per_sigma'][str(sigma)].append(l2_norm)
            else:
                data['l2_norm_per_sigma'][str(sigma)] = [l2_norm]

        # TRAINING SET MEASURES:
        score = logreg.score(full_x_train, y_train.flatten())
        data['train_accuracy'].append(score)

        predicted_proba = logreg.predict_proba(full_x_train)
        avg_ce = log_loss(y_train.flatten(), predicted_proba)
        data['train_avg_ce'].append(avg_ce)

        # TEST SET MEASURES:
        score = logreg.score(full_x_test, y_test.flatten())
        data['test_accuracy'].append(score)

        predicted_proba = logreg.predict_proba(full_x_test)
        avg_ce = log_loss(y_test.flatten(), predicted_proba)
        data['test_avg_ce'].append(avg_ce)

    return data


def plot(data):
    ###
    # Q1.g PLOT TRAIN & TEST AVERAGE CROSS ENTROPY ERROR
    ###
    print "\nLambda:", data['lambda']
    # NOTE: replacing lambda = 0 to 0.0001 in order to plot the points at 0 since
    # log scale will not show lambda = 0
    print "Train Average Cross Entropy:", data['train_avg_ce']
    print "Test Average Cross Entropy:", data['test_avg_ce']
    plt.title('QUESTION 1.g)\nTrain & Test Average Cross Entropy')
    train_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], data['train_avg_ce'], 'bo-')
    test_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], data['test_avg_ce'], 'ro-')
    plt.legend([train_line, test_line], ['train', 'test'])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('average cross entropy')
    plt.show()
    ###
    # Q1.g PLOT TRAIN & TEST ACCURACIES
    ###
    print "Train Accuracies:", data['train_accuracy']
    print "Test Accuracies:", data['test_accuracy']
    plt.title('QUESTION 1.g)\nTrain & Test Accuracies')
    train_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], data['train_accuracy'], 'bo-')
    test_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], data['test_accuracy'], 'ro-')
    plt.legend([train_line, test_line], ['train', 'test'])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylim(0, 1)
    plt.ylabel('accuracy')
    plt.show()
    ###
    # Q1.g PLOT L2 NORM
    ###
    print "L2 norm:", data['l2_norm']
    plt.title('QUESTION 1.g)\nL2 norm of Weight vector')
    plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], data['l2_norm'], 'bo-')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.yscale('log')
    plt.ylabel('L2 norm of W')
    plt.show()
    ###
    # Q1.g PLOT L2 NORM OF EACH BASIS FUNCTION WEIGHTS
    ###
    colors = ['b', 'r', 'g', 'm', 'c']
    assert len(colors) == len(data['l2_norm_per_sigma'])
    lines = []
    labels = []
    plt.title('QUESTION 1.g)\nL2 norm of Weights for each gaussian function')
    for i, (sigma, l2_norms) in enumerate(data['l2_norm_per_sigma'].iteritems()):
        print "sigma", sigma, "L2 norm:", l2_norms
        line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000, 10000], l2_norms, colors[i]+'o-')
        lines.append(line)
        labels.append('sigma %s' % sigma)
    plt.legend(lines, labels)
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.yscale('log')
    plt.ylabel('L2 norm of W')
    plt.show()
