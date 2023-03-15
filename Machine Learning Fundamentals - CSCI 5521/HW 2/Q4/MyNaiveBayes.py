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
        x = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
        psis = np.reshape(self.psis, (self.K, 1, X_test.shape[1]))
        
        logpy = np.log(self.phis).reshape([self.K,1])
        logpxy = x * np.log(psis) + (1-x) * np.log(1-psis)
        logpyx = logpxy.sum(axis=2) + logpy
 
        return logpyx.argmax(axis=0).flatten()

    def get_prior_prob(self, X):
        # compute the prior probability of class k 
        # (Hint: prior prob is the number of samples in class k divided by the total number of samples)
        # (Hint: this should return a scalar)
        prior_prob = len(X)/self.n
        return prior_prob

    def get_class_likelihood(self, X):
        # estimate Bernoulli parameter theta for each feature for each class
        # (Hint: parameter is the mean of the features in class k)
        # (Hint: this should return a vector of size d)
        theta = np.zeros([1,self.d]).astype("float")
        for i in range(X.shape[0]):
            for j in range(self.d):
                if X[i][j] == 1:
                    theta[0][j] += 1
        for m in range(self.d):
                theta[0][m] = theta[0][m]/len(X)
        # (Hint: this should return a vector of size d)
        return theta