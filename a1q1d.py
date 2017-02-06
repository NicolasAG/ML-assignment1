import numpy as np

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

###
# QUESTION 1.d)
# - Run logistic regression using L2 regularization (lambda = {0, 0.1, 1, 10, 100, 1000})
# - Plot average (over all instances) cross-entropy for training & testing data as a function of lambda (log scale).
# - Plot the L2 norm of the weight vector.
# - Plot the actual values of the weights (one curve per weight).
# - Plot the accuracy on the training and test set.
# Explain briefly what you see.
###


def log_reg(x_train, y_train, x_test, y_test, n_iter=10000, use_sgd=True):
    data = {
        'lambda': [0, 0.1, 1., 10., 100., 1000.],
        'train_avg_ce': [],
        'test_avg_ce': [],
        'l2_norm': [],
        'weights': [],
        'train_accuracy': [],
        'test_accuracy': []
    }

    print "\nLogistic Regression on data with L2 regularization..."
    for l in data['lambda']:
        if l == 0:
            if use_sgd: logreg = SGDClassifier(loss='log', penalty='none', n_iter=n_iter, shuffle=False)
            else: logreg = LogisticRegression(penalty='l2', C=1.e12, max_iter=n_iter)
        else:
            if use_sgd: logreg = SGDClassifier(loss='log', penalty='l2', alpha=l, n_iter=n_iter, shuffle=False)
            else: logreg = LogisticRegression(penalty='l2', C=1./l, max_iter=n_iter)
        logreg.fit(x_train, y_train.flatten())

        # print "W:", logreg.coef_[0]
        data['weights'].append(logreg.coef_[0])
        # print "L2 norm of W:", np.sum(logreg.coef[0]**2)
        data['l2_norm'].append(np.sum(logreg.coef_[0]**2))

        # TRAINING SET MEASURES:
        # print "TRAIN prediction:", logreg.predict(x_train)
        # print "true:", y_train.flatten()
        score = logreg.score(x_train, y_train.flatten())
        # print "TRAIN accuracy score:", score
        data['train_accuracy'].append(score)

        predicted_proba = logreg.predict_proba(x_train)
        # print "predict_proba:", predicted_proba
        avg_ce = log_loss(y_train.flatten(), predicted_proba)
        # print "average cross entropy:", avg_ce
        data['train_avg_ce'].append(avg_ce)

        # TEST SET MEASURES:
        # print "TEST prediction:", logreg.predict(x_test)
        # print "true:", y_test.flatten()
        score = logreg.score(x_test, y_test.flatten())
        # print "TEST accuracy score:", score
        data['test_accuracy'].append(score)

        predicted_proba = logreg.predict_proba(x_test)
        # print "predict_proba:", predicted_proba
        avg_ce = log_loss(y_test.flatten(), predicted_proba)
        # print "average cross entropy:", avg_ce
        data['test_avg_ce'].append(avg_ce)

    return data


def plot(data):
    ###
    # Q1.d PLOT TRAIN & TEST AVERAGE CROSS ENTROPY ERROR
    ###
    # NOTE: replacing lambda = 0 to 0.0001 in order to plot the points at 0 since log scale will not show lambda = 0
    print "\nLambda:", data['lambda']
    print "Train Average Cross Entropy:", data['train_avg_ce']
    print "Test Average Cross Entropy:", data['test_avg_ce']
    plt.title('QUESTION 1.d)\nTrain & Test Average Cross Entropy')
    train_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000], data['train_avg_ce'], 'bo-')
    test_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000], data['test_avg_ce'], 'ro-')
    plt.legend([train_line, test_line], ['train', 'test'])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('average cross entropy')
    plt.show()
    ###
    # Q1.d PLOT L2 NORM
    ###
    print "L2 norm:", data['l2_norm']
    plt.title('QUESTION 1.d)\nL2 norm of Weight vector')
    plt.plot([0.0001, 0.1, 1, 10, 100, 1000], data['l2_norm'], 'bo-')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.yscale('log')
    plt.ylabel('L2 norm of W')
    plt.ylim(-0.1, max(data['l2_norm'])+1)
    plt.show()
    ###
    # Q1.d PLOT WEIGHT VALUES
    ###
    plt.title('QUESTION 1.d)\nWeight Values')
    for j in range(len(data['weights'][0])):
        plt.plot(
            [0.0001, 0.1, 1, 10, 100, 1000],
            [data['weights'][i][j] for i in range(len(data['lambda']))],
            'bo-'
        )
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('weight values')
    plt.show()
    ###
    # Q1.d PLOT TRAIN & TEST ACCURACIES
    ###
    print "Train Accuracies:", data['train_accuracy']
    print "Test Accuracies:", data['test_accuracy']
    plt.title('QUESTION 1.d)\nTrain & Test Accuracies')
    train_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000], data['train_accuracy'], 'bo-')
    test_line, = plt.plot([0.0001, 0.1, 1, 10, 100, 1000], data['test_accuracy'], 'ro-')
    plt.legend([train_line, test_line], ['train', 'test'])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylim(0, 1)
    plt.ylabel('accuracy')
    plt.show()
