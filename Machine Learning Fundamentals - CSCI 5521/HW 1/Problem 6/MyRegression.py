"""
This is the provided pseudo-code and the template that you can implement the regression with different order
 of polynomial regression as well as the evaluation of cross-validation. Last, you can also visualize the
 polynomial regression residual error figure

MyRegression takes the X, y ,split and order as input and return the error_dict that contain the mse of different fold
 of the dataset

VisualizeError is used to plot the figure of the error analysis
"""
# Hints
# the numpy package is useful for calculation
# sklearn.linear_model is another useful tool that you can use to fit the model with the data
# refer https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html to see how to use it
# poly.fit_transform also a useful function to generate the X matrix that can be the input of LinearRegression
# refer https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html to see how to use it


# Header
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Implement the Perceptron algorithm
def MyRegression(X, y, split, order=2):
    # initialize the error dict, where the key is the k-th fold, and the error_dict[k] is the mean square error of
    # the test sample in the k-th fold.

    error_dict = {}
    train_error_dict = {}

    for k in range(10):
        error_dict[k] = -1
        train_error_dict[k] = -1

    # in this for loop
    for k in range(10):

        # Creating training_set and test_set for the current split value

        # get the index i such that split[i] = k
        split_idx = []

        # get the index i such that split[i] != k
        not_split_idx = []

        for i in range(len(split)):
            if split[i][0] == k:
                split_idx.append(i)
            else:
                not_split_idx.append(i)

        train_X = np.delete(X, split_idx, 0)
        train_y = np.delete(y, split_idx, 0)
        test_X = np.delete(X, not_split_idx, 0)
        test_y = np.delete(y, not_split_idx, 0)

        # select the training set where the split value is not k (Hint: poly.fit_transform can be used)
        poly = PolynomialFeatures(order)
        train_X = poly.fit_transform(train_X)

        # select the test set where the split value is k (Hint: poly.fit_transform can be used)
        test_X = poly.fit_transform(test_X)

        # build the regression model (Hint: sklearn.linear_model can be used)
        model = LinearRegression()
        model.fit(train_X, train_y)

        # predict the test_X
        predict_y = model.predict(test_X)  # prdiction on test set by model
        predict_train_y = model.predict(train_X)   # prediction on training by model

        # calculate the mean square error
        # Hint: ((predict - ground_truth) ** 2).mean
        MSE = mean_squared_error(predict_y, test_y)
        train_MSE = mean_squared_error(predict_train_y, train_y)
        # MSE = ((predict_y - test_y) ** 2).mean(axis=0)

        # record the error
        # Hint: error_dict[k] = xxx
        error_dict[k] = MSE
        train_error_dict[k] = train_MSE

    return error_dict #, train_error_dict


def VisualizeError(error_related_to_order): #train_error_related_to_order):
    # collect the order information
    # collect the mse and calculate the mean corresponding of different order of polynomial

    xx = []
    yy = []

    for order, error_list in error_related_to_order.items():
        mean_MSE = sum(error_list) / 10.0
        xx.append(order)
        yy.append(mean_MSE)

    train_xx = []
    train_yy = []

    # Plotting
    plt.xlabel('order')
    plt.ylabel('mean_MSE')
    plt.plot(xx, yy, color="red", label="Test MSE")
    plt.plot(train_xx, train_yy, color="blue", label="Train MSE")
    plt.legend()
    plt.show()
