# import libraries
import numpy as np


class Kmeans:
    """my implementation of Kmeans algorithm"""
    def __init__(self, k=3):  # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        """ initialize the centers of clutsers as a set of pre-selected samples"""
        init_idx = [
            500, 1500, 2500,
        ] 
        self.center = X[init_idx]
        num_iter = 0  # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype("int")
        cluster_assignment = np.zeros([len(X),]).astype("int")
        is_converged = False

        # iteratively update the centers of clusters till convergence
        while not is_converged:

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                dist = [None]*len(self.center)
                for j in range(len(self.center)):
                    dist[j] = np.linalg.norm(X[i] - self.center[j])
                
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                min_dist_index = dist.index(min(dist))
                cluster_assignment[i] = min_dist_index
                #pass
                
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample

            # update the centers based on cluster assignment (M step)
            new_center = np.zeros((3,X.shape[1])).astype("int")
            num_cluster = [0, 0, 0]
            for i in range(len(X)):
                if cluster_assignment[i] == 0:
                    for j in range(X.shape[1]):
                        new_center[0][j] = new_center[0][j] + X[i][j]
                    num_cluster[0] += 1
                elif cluster_assignment[i] == 1:
                    for j in range(X.shape[1]):
                        new_center[1][j] = new_center[1][j] + X[i][j]
                    num_cluster[1] += 1
                elif cluster_assignment[i] == 2:
                    for j in range(X.shape[1]):
                        new_center[2][j] = new_center[2][j] + X[i][j]
                    num_cluster[2] += 1 
            for m in range(new_center.shape[0]):
                for n in range(new_center.shape[1]):
                    new_center[m][n] = int(float(new_center[m][n])/float(num_cluster[m]))       
            self.center = new_center

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        return num_iter, self.error_history, cluster_assignment, self.center

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
            for j in range(len(self.center)):
                error += np.linalg.norm(X[i] - self.center[j])

        return error

    def params(self):
        return self.center