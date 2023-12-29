import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-6.0, -1.0],
                        [-3.0, +2.0],
                        [+3.0, +2.0],
                        [+6.0, -1.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +4.0]],
                              [[+2.4, -2.0],
                               [-2.0, +2.4]],
                              [[+2.4, +2.0],
                               [+2.0, +2.4]],
                              [[+0.4, +0.0],
                               [+0.0, +4.0]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter=",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description


def initialize_parameters(X, K):
    # your implementation starts below

    means = np.genfromtxt("hw05_initial_centroids.csv", delimiter=',')
    covariances = [np.cov(X.T) for i in range(K)]

    priors = np.zeros(K)
    for i in range(K):
        priors[i] = 1/K

    # your implementation ends above
    return (means, covariances, priors)


means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm


def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below

    for i in range(100):
        hs = [stats.multivariate_normal(means[k], covariances[k]).pdf(
            X) * priors[k] for k in range(K)]
        hs_sum = np.sum(hs, axis=0)

        expt = np.array([h / hs_sum for h in hs])
        means = np.array(
            [np.sum(expt[c][:, None] * X, axis=0) / np.sum(expt[c]) for c in range(K)])

        for c in range(K):
            covariances[c] = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[0]):
                covariances[c] += np.matmul((X[i] - means[c])[:, None],
                                            (X[i] - means[c])[None, :]) * expt[c][i]
            covariances[c] /= np.sum(expt[c], axis=0)

        priors = [np.sum(expt[k], axis=0) / X.shape[0] for k in range(K)]
        priors = np.array(priors)

        assignments = np.argmax(expt, axis=0)

    # your implementation ends above
    return (means, covariances, priors, assignments)


means, covariances, priors, assignments = em_clustering_algorithm(
    X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description


def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

    xmin = X[:, 0].min()
    xmax = X[:, 0].max()
    ymin = X[:, 1].min() - 1
    ymax = X[:, 1].max() + 1

    x, y = np.meshgrid(np.linspace(xmin, xmax),
                       np.linspace(ymin, ymax))

    for c in range(K):
        plt.plot(X[assignments == c, 0], X[assignments == c, 1],
                 ".", markersize=10, color=cluster_colors[c])

        xy_grid = np.column_stack((x.flatten(), y.flatten()))
        pdf_values = stats.multivariate_normal.pdf(
            xy_grid, means[c], covariances[c]).reshape(x.shape)
        plt.contour(x, y, pdf_values, levels=[0.01], colors=cluster_colors[c])

        initial_pdf_values = stats.multivariate_normal.pdf(
            xy_grid, group_means[c], group_covariances[c]).reshape(x.shape)
        plt.contour(x, y, initial_pdf_values, levels=[
                    0.01], colors="black", linestyles="dashed")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    # your implementation ends above


draw_clustering_results(X, K, group_means, group_covariances,
                        means, covariances, assignments)
