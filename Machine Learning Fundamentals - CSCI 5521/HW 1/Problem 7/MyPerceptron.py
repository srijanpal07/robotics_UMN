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
    for t in range(0, len(X), 1):
      if y[t]*np.inner(w, X[t]) <= 0:
        w = w + y[t]*X[t]
        if np.array_equal(w, w0):
          break
      iter_value = iter_value + 1
        
          
    # compute the error rate
    for t in range(0, len(X), 1):
      if y[t]*np.inner(w, X[t]) <= 0:
        error_rate = error_rate + 1
    error_rate = error_rate/len(X)
    
    return (w, iter_value, error_rate)