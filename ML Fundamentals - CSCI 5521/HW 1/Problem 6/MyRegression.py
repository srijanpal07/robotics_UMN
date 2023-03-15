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
import matplotlib.pyplot as plt

# Implement the Perceptron algorithm
def MyRegression(X, y, split, order = 2):
    # initialize the error dict, where the key is the k-th fold, and the error_dict[k] is the mean square error of
    # the test sample in the k-th fold.
    error_dict = {}
    for k in range(10):
        error_dict[k] = -1

    # in this for loop
    for k in range(10):
        pass
        # select the training set where the split value is not k (Hint: poly.fit_transform can be used)

        # select the test set where the split value is k (Hint: poly.fit_transform can be used)

        # build the regression model (Hint: sklearn.linear_model can be used)

        # predict the test_X


        # calculate the mean square error
        # Hint: ((predict - ground_truth) ** 2).mean

        # record the error
        # Hint: error_dict[k] = xxx

    return error_dict


def VisualizeError(error_related_to_order):
    pass
    # collect the order information
    # collect the mse and calculate the mean corresponding of different order of polynimal


    # Plotting


