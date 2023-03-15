"""
HW 1 Skeleton

In this assignment, you don't need to modify this file, but only write the code on "MyRegression.py" instead.
This file read the date, and them call the function from "MyRegression.py" with a specific argument and collect the
cross validation error of each split. Last, you can use the collected validation evaluation metric to plot the error
and learn the overfitting and underfitting concepts.

Use the function "plt.show()", you can see the plot intereactively.
If you omit the "plt.show()" and the program would display the saved PNG files.

"""

# Header
import numpy as np
from matplotlib import pyplot as plt
from MyRegression import MyRegression, VisualizeError

# Data loading
data = np.genfromtxt('RegressionData.csv', delimiter=',')
X = data[:,:1]
y = data[:,1:2]
split = data[:,2:3].astype(np.int32)

# sub problem 1
error_related_to_order = {}
# check for different order
for order in [2, 3, 4, 5, 6]:
    mse = MyRegression(X, y, split, order)
    error_related_to_order[order] = list(mse.values())

print(error_related_to_order)

# sub problem 2
VisualizeError(error_related_to_order)
