import numpy as np

"""
Specify your sigma for RBF kernel in the order of questions (simulated data, digit-49, digit-79)
"""
sigma_pool = [0.001, 1, 5]


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
        self.gamma = None
        self.alpha = np.zeros([len(train_x),]).astype(
            "float32"
        )  # parameters to be optimized

    def RBF_kernel(self, x):
        # Implement the RBF kernel
        dist = np.zeros((self.train_x.shape[0], x.shape[0]))
        K = np.zeros((self.train_x.shape[0], x.shape[0]))
        for i in range(dist.shape[0]):
          for j in range(dist.shape[1]):
            dist[i][j] = (np.linalg.norm(self.train_x[i] - x[j]))**2
            K[i][j] = np.exp(-dist[i][j]/self.sigma) 
        
        return K

    def fit(self, train_x, train_y):
        # set a maximum training iteration
        max_iter = 20

        for iter in range(max_iter):
            error_count = 0  # use a counter to record number of misclassification
            k = self.RBF_kernel(train_x)
            yhat = np.zeros(train_y.shape)

             # loop through all samples and update the parameter accordingly
            for i in range(train_x.shape[0]):
              yhat[i] = (self.alpha.T * train_y) @ k[:, i]
              if yhat[i] > 0:
                yhat[i] = 1 
              else:
                yhat[i] = -1

              if yhat[i] != train_y[i]:
                  error_count = error_count + 1
                  self.alpha[i] = self.alpha[i]+ 1

              # stop training if parameters do not change any more
              if error_count==0:
                break

    def predict(self, test_x):
        # generate predictions for given data
        k = self.RBF_kernel(test_x)
        pred = ((self.alpha.T * self.train_y) @ k).T
        for i in range(len(pred)):
          if pred[i]>0:
            pred[i] = 1
          else:
            pred[i] = -1
        
        return pred

    def param(
        self,
    ):
        return self.alpha