import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs



class KMeans:
    def __init__(self, n_clusters, init='k-means++', max_iter=300, tol=0.0001, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.iteration_data = []

    def random_initialization(self):
        rng = np.random.RandomState(self.random_state)
        idxs = rng.choice(self.X.shape[0], size=self.n_clusters)
        return self.X[idxs]

    def initialization(self):
        if self.init == 'random':
            return self.random_initialization()
        elif self.init == 'k-means++':
            return self.kmeans_plus_plus_initialization()
    def visualize(self):
        n_plots = len(self.iteration_data)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten()

        for i, data in enumerate(self.iteration_data):
            X, centroids, labels, itr = data
            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette=['purple', 'orange', 'yellow', 'red', 'blue'], ax=axes[i])
            axes[i].scatter(centroids[:, 0], centroids[:, 1], marker='o', color='black', s=150)
            axes[i].set_title(f'Iteration: {itr}')

        for ax in axes[len(self.iteration_data):]:
            ax.remove()

        plt.tight_layout()
        plt.show()

    def fit(self, X):
        self.X = X.copy()
        self.cluster_centers_ = self.initialization()

        for itr in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy()

            centroid_dist = np.linalg.norm((self.X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]), axis=-1)
            self.labels_ = np.argmin(centroid_dist, axis=-1)

            self.inertia_ = 0
            for idx in range(self.n_clusters):
                self.cluster_centers_[idx] = self.X[self.labels_ == idx].mean(axis=0)
                self.inertia_ += ((self.X[self.labels_ == idx] - self.cluster_centers_[idx][np.newaxis, :])**2).sum()

            centroids_change = old_cluster_centers - self.cluster_centers_
            if np.linalg.norm(centroids_change, ord='fro') >= self.tol:
                self.iteration_data.append((self.X, old_cluster_centers, self.labels_, itr + 1))

            if np.linalg.norm(centroids_change, ord='fro') < self.tol:
                print(f"Converged after {itr} iterations")
                break

        self.labels_ = self.predict(self.X)
        self.visualize()

    def predict(self, X):
        centroid_dist = np.linalg.norm((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]), axis=-1)
        labels = np.argmin(centroid_dist, axis=-1)
        return labels


X, y = make_blobs(n_samples=600, centers=4, random_state=1000, cluster_std=1)
kmeans = KMeans(n_clusters=4, random_state=80, init='random')
kmeans.fit(X)