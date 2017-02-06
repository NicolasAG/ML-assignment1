from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

###
# QUESTION 1.f)
# Logistic Regression with no regularization
# - Plot training and testing error as a function of sigma
# - Add constant lines showing the training and testing error you had obtained in part d
# Explain how sigma influences overfitting and the bias-variance trade-off
###


def log_reg(gaussian_data, y_train, y_test, sigmas, n_iter=10000, use_sgd=True):
    print "\nLogistic Regression on data applied to gaussian basis functions (no regularization)..."
    for sigma in sigmas:
        basis_x_train = gaussian_data[str(sigma)]['x_train']
        basis_x_test = gaussian_data[str(sigma)]['x_test']

        if use_sgd: logreg = SGDClassifier(loss='log', penalty='none', n_iter=n_iter, shuffle=False)
        else: logreg = LogisticRegression(penalty='l2', C=1.e12, max_iter=n_iter)
        logreg.fit(basis_x_train, y_train.flatten())

        # train metrics
        score = logreg.score(basis_x_train, y_train.flatten())
        gaussian_data[str(sigma)]['train_accuracy'] = score  # accuracy

        predicted_proba = logreg.predict_proba(basis_x_train)
        avg_ce = log_loss(y_train.flatten(), predicted_proba)
        gaussian_data[str(sigma)]['train_avg_ce'] = avg_ce  # average cross-entropy error

        # test metrics
        score = logreg.score(basis_x_test, y_test.flatten())
        gaussian_data[str(sigma)]['test_accuracy'] = score  # accuracy

        predicted_proba = logreg.predict_proba(basis_x_test)
        avg_ce = log_loss(y_test.flatten(), predicted_proba)
        gaussian_data[str(sigma)]['test_avg_ce'] = avg_ce  # average cross-entropy error

    return gaussian_data


def plot(gaussian_data, sigmas, old_data):
    ###
    # Q1.f PLOT TRAIN & TEST AVERAGE CROSS ENTROPY ERROR
    ###
    print "Sigma:", sigmas
    train_ce = [gaussian_data[str(sigma)]['train_avg_ce'] for sigma in sigmas]
    test_ce = [gaussian_data[str(sigma)]['test_avg_ce'] for sigma in sigmas]
    print "Train avg. Cross-Entropy:", train_ce
    print "Test avg. Cross-Entropy:", test_ce
    plt.title('QUESTION 1.f)\nTrain & Test Avg. Cross-Entropy')
    train_line, = plt.plot(sigmas, train_ce, 'bo-')  # plot train gaussian basis accuracies.
    test_line, = plt.plot(sigmas, test_ce, 'ro-')  # plot test gaussian basis accuracies.
    # Plot old average Cross-Entropy with no regularization:
    train_line_old, = plt.plot(sigmas, [old_data['train_avg_ce'][0]]*len(sigmas), 'c-')
    test_line_old, = plt.plot(sigmas, [old_data['test_avg_ce'][0]]*len(sigmas), 'm-')
    # Add legend, scale, and show
    plt.legend(
        [train_line, test_line, train_line_old, test_line_old],
        ['train gaussian', 'test gaussian', 'train no regularization', 'test no regularization']
    )
    plt.xlabel('sigma')
    plt.ylabel('average cross-entropy')
    plt.ylim(-0.1, max(train_ce+test_ce+[old_data['train_avg_ce'][0]]+[old_data['test_avg_ce'][0]])+1)
    plt.show()
    ###
    # Q1.f PLOT TRAIN & TEST ACCURACY
    ###
    train_acc = [gaussian_data[str(sigma)]['train_accuracy'] for sigma in sigmas]
    test_acc = [gaussian_data[str(sigma)]['test_accuracy'] for sigma in sigmas]
    print "Train accuracies:", train_acc
    print "Test accuracies:", test_acc
    plt.title('QUESTION 1.f)\nTrain & Test Accuracies')
    train_line, = plt.plot(sigmas, train_acc, 'bo-')  # plot train gaussian basis accuracies.
    test_line, = plt.plot(sigmas, test_acc, 'ro-')  # plot test gaussian basis accuracies.
    # Plot old accuracy with no regularization:
    train_line_old, = plt.plot(sigmas, [old_data['train_accuracy'][0]]*len(sigmas), 'c-')
    test_line_old, = plt.plot(sigmas, [old_data['test_accuracy'][0]]*len(sigmas), 'm-')
    # Add legend, scale, and show
    plt.legend(
        [train_line, test_line, train_line_old, test_line_old],
        ['train gaussian', 'test gaussian', 'train no regularization', 'test no regularization'],
        loc='lower right'
    )
    plt.xlabel('sigma')
    plt.ylim(0, 1.01)
    plt.ylabel('accuracy')
    plt.show()
