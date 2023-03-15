import numpy as np

class NaiveBayes:
    def __init__(self, X_train, y_train):
        self.n = X_train.shape[0] # size of the dataset
        self.d = X_train.shape[1] # size of the feature vector
        self.K = len(set(y_train)) # size of the class set

        # these are the shapes of the parameters
        self.psis = np.zeros([self.K, self.d]) 
        self.phis = np.zeros([self.K])

    def fit(self, X_train, y_train):

        # we now compute the parameters
        for k in range(self.K):
            X_k = X_train[y_train == k]
            self.phis[k] = self.get_prior_prob(X_k) # prior
            self.psis[k] = self.get_class_likelihood(X_k) # likelihood

        # clip probabilities to avoid log(0)
        self.psis = self.psis.clip(1e-14, 1-1e-14)

    def predict(self, X_test):
        # compute log-probabilities
        # (Hint: Using the bayes rule, and the log-sum-exp trick)
        # (Hint: this should return class for every sample in X_test)
        pass

    def get_prior_prob(self, X):
        # compute the prior probability of class k 
        # (Hint: prior prob is the number of samples in class k divided by the total number of samples)
        # (Hint: this should return a scalar)
        pass

    def get_class_likelihood(self, X):
        # estimate Bernoulli parameter theta for each feature for each class
        # (Hint: parameter is the mean of the features in class k)
        # (Hint: this should return a vector of size d)
        pass