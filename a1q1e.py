import numpy as np

###
# QUESTION 1.e)
# - Re-format data by feeding each of the input variables (except the bias term) through a set of Gaussian basis functions:
# - use 5 univariate basis functions with means evenly spaced between -10 and 10 and variance sigma in [0.1, 0.5, 1, 5, 10]
###


def reformat(x_train, x_test, sigmas, means):
    gaussian_data = {}  # map from "sigma" to x_train and x_test

    for sigma in sigmas:
        big_x_train = np.array([])
        big_x_test = np.array([])
        for mean in means:
            # Compute gaussian matrices for all columns. Note that there is no bias term in X since added by sklearn.
            new_train_matrix = np.exp(-(x_train-mean)**2 / (2*sigma**2))
            new_test_matrix = np.exp(-(x_test-mean)**2 / (2*sigma**2))
            # Concatenate columns (train set):
            if big_x_train.size:
                big_x_train = np.hstack((big_x_train, new_train_matrix))
            else:
                big_x_train = new_train_matrix
            # Concatenate columns (test set):
            if big_x_test.size:
                big_x_test = np.hstack((big_x_test, new_test_matrix))
            else:
                big_x_test = new_test_matrix
        # Insert again bias column
        gaussian_data[str(sigma)] = {
            'x_train': big_x_train,
            'x_test': big_x_test
        }
        # print "sigma:", sigma
        # print "  train:", gaussian_data[str(sigma)]['x_train'].shape
        # print "  test:", gaussian_data[str(sigma)]['x_test'].shape

    return  gaussian_data
