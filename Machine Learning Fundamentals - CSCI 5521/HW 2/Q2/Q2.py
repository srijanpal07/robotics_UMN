# import libraries
import time

import numpy as np
from matplotlib import pyplot as plt

from Mykmeans import Kmeans
from MyPCA import PCA

# read in data.
data = np.genfromtxt("Digits089.csv", delimiter=",")
X = data[:, 2:]
y = data[:, 1]

# apply kmeans algorithms to raw data

start = time.time()
clf = Kmeans(k=3)
num_iter, error_history, cluster, center = clf.run_kmeans(X, y)
time_raw = time.time() - start

# plot the history of reconstruction error
fig = plt.figure()
plt.plot(np.arange(len(error_history)), error_history, "b-", linewidth=2)
fig.set_size_inches(10, 10)
fig.savefig("raw_data.png")
plt.show()

# apply kmeans algorithms to low-dimensional data (PCA) that captures > 90% of variance

X_pca_, num_dim_pca = PCA(X)
start = time.time()
clf = Kmeans(k=3)
num_iter_pca, error_history_pca, cluster_pca, center_pca = clf.run_kmeans(X_pca_, y)
time_pca = time.time() - start

# plot the history of reconstruction error
fig1 = plt.figure()
plt.plot(
    np.arange(len(error_history_pca)), error_history_pca, "b-", linewidth=2
)
fig1.set_size_inches(10, 10)
fig1.savefig("pca.png")
plt.show()

# print the number of iterations for convergence
print("#################")
print("Using raw data converged in %d iteration (%.2f seconds)" % (num_iter, time_raw))

print("#################")
print(
    "Project data into %d dimensions with PCA converged in %d iteration (%.2f seconds)"
    % (num_dim_pca, num_iter_pca, time_pca)
)
print("#################")

# print the number of iterations for convergence
X_pca, num_dim = PCA(np.concatenate((X, center), axis=0), 2)
fig = plt.figure()
colors = {0: 'r', 1: 'g', 2: 'b'}
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[colors[i] for i in cluster] + ['k']*3)
fig.savefig("kmeans_cluster.png")
plt.show()

X_pca, num_dim = PCA(np.concatenate((X_pca_, center_pca), axis=0), 2)
fig = plt.figure()
colors = {0: 'r', 1: 'g', 2: 'b'}
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[colors[i] for i in cluster_pca] + ['k']*3)

fig.savefig("kmeans_cluster_pca.png")
plt.show()