"""
Refer to the pseudo code in our assignment, you can write code to complete your algorithm.

The following three parameter need to be returned:
    1. the final weight w
    2. the number of iteration that it makes w converge
    3. error rate, it represents the fraction of training samples that are classified to another one.
"""
# Hints
# the numpy package is useful for calculation

# Header
import numpy as np

# Please implement your algorithm
def MyPerceptron(X, y, w0=[0.1,-1.0]):
    # we initialize the variable to record the number of iteration that it makes w converge
    iter_value = 0
    w = w0
    error_rate = 1.00

    # update w

    # compute the error rate

    return (w, iter_value, error_rate)
