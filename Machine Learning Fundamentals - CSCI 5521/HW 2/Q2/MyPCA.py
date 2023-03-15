import numpy as np

def PCA(X,num_dim=None):
    X_pca, num_dim = X, len(X[0]) # placeholder

    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
    X_mean = X - np.mean(X, axis = 0)
    cov_mat = np.cov(X_mean, rowvar = False)
    evals, evecs  = np.linalg.eigh(cov_mat)
    sorted_evals = np.flip(evals)
    
    sum_eval = 0
    var=0
    for i in range(len(sorted_evals)):
        sum_eval += sorted_evals[i]

    # select the reduced dimensions that keep >90% of the variance
    #if num_dim is None:
        #pass
    for i in range(len(sorted_evals)):
        var += sorted_evals[i]
        if var/sum_eval > 0.9:
            num_dim = i
            break

    sorted_evecs = np.flip(evecs, axis=1)
    evec_subset = sorted_evecs[:, :num_dim]
    X_pca = np.dot(X, evec_subset)

    # project the high-dimensional data to low-dimensional one


    return X_pca, num_dim
