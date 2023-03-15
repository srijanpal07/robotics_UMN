import numpy as np

"""
Specify your sigma for RBF kernel in the order of questions (simulated data, digit-49, digit-79)
"""
sigma_pool = [0, 0, 0]


class KernelPerceptron:
    """
    Perceptron Algorithm with RBF Kernel
    """

    def __init__(self, train_x, train_y, sigma_idx):
        self.sigma = sigma_pool[sigma_idx]  # sigma value for RBF kernel
        self.train_x = (
            train_x  # kernel perceptron makes predictions based on training data
        )
        self.train_y = train_y
        self.alpha = np.zeros([len(train_x),]).astype(
            "float32"
        )  # parameters to be optimized

    def RBF_kernel(self, x):
        # Implement the RBF kernel
        pass

    def fit(self, train_x, train_y):
        # set a maximum training iteration
        max_iter = 1000

        # training the model
        for iter in range(max_iter):
            error_count = 0  # use a counter to record number of misclassification

            # loop through all samples and update the parameter accordingly

            # stop training if parameters do not change any more
            if error_count == 0:
                break

    def predict(self, test_x):
        # generate predictions for given data
        pred = np.ones([len(test_x)]).astype("float32")  # placeholder

        return pred

    def param(
        self,
    ):
        return self.alpha
