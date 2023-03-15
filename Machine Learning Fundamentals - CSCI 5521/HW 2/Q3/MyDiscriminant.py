import numpy as np


class GaussianDiscriminant:
    def __init__(self, k=2, d=8, priors=None, shared_cov=False):
        self.mean = np.zeros((k, d))  # mean
        self.shared_cov = (
            shared_cov  # using class-independent covariance or not
        )
        if self.shared_cov:
            self.S = np.zeros((d, d))  # class-independent covariance
        else:
            self.S = np.zeros((k, d, d))  # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [
                1.0 / k for i in range(k)
            ]  # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        self.mean[0] = np.mean(Xtrain[ytrain == 1], axis=0)  #placeholder
        self.mean[1] = np.mean(Xtrain[ytrain == 2], axis=0)  #placeholder

        if self.shared_cov:
            # compute the class-independent covariance
            self.S = np.cov(Xtrain, rowvar = False)
        else:
            # compute the class-dependent covariance
            for i in range(self.k):
                self.S[i] = np.cov(Xtrain[ytrain == i+1], rowvar = False)

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0])  # placeholder

        for i in np.arange(Xtest.shape[0]):  # for each test set example
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    disc_1 = -0.5*np.linalg.multi_dot((np.transpose(Xtest[i]-self.mean[0]), np.linalg.inv(self.S), (Xtest[i]-self.mean[0]))) - 0.5*np.linalg.det(self.S) + np.log(0.3) 
                    disc_2 = -0.5*np.linalg.multi_dot((np.transpose(Xtest[i]-self.mean[1]), np.linalg.inv(self.S), (Xtest[i]-self.mean[1]))) - 0.5*np.linalg.det(self.S) + np.log(0.7)
                    if disc_1>disc_2:
                        predicted_class[i] = 1
                    elif disc_2>disc_1:
                        predicted_class[i] = 2

                else:
                    disc_1 = -0.5*np.linalg.multi_dot((np.transpose(Xtest[i]-self.mean[0]), np.linalg.inv(self.S[c]), (Xtest[i]-self.mean[0]))) - 0.5*np.linalg.det(self.S[c]) + np.log(0.3) 
                    disc_2 = -0.5*np.linalg.multi_dot((np.transpose(Xtest[i]-self.mean[1]), np.linalg.inv(self.S[c]), (Xtest[i]-self.mean[1]))) - 0.5*np.linalg.det(self.S[c]) + np.log(0.7)
                    if disc_1>disc_2:
                        predicted_class[i] = 1
                    elif disc_2>disc_1:
                        predicted_class[i] = 2

        # determine the predicted class based on the values of discriminant function
        # remember to return 1 or 2 for the predicted class

        return predicted_class
