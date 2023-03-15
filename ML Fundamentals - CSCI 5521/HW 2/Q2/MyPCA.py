import numpy as np

def PCA(X,num_dim=None):
    X_pca, num_dim = X, len(X[0]) # placeholder

    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)


    # select the reduced dimensions that keep >90% of the variance
    if num_dim is None:
        pass


    # project the high-dimensional data to low-dimensional one


    return X_pca, num_dim
