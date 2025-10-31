import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import AgglomerativeClustering

np.random.seed(42)
X = np.vstack([np.random.randn(20, 2) + a for a in [(0, 0), (5, 5), (10, 0)]])

D = distance_matrix(X, X)
MST = minimum_spanning_tree(D).toarray()

thr = np.percentile(MST[MST > 0], 90)
trim = MST * (MST < thr)

n_clusters = 3
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(X)
labels = clustering.labels_

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
titles = ["Generated Data", "Clustered Data", "Full MST", "Trimmed MST"]

ax[0].scatter(X[:, 0], X[:, 1], color='gray')
ax[0].set_title(titles[0])

ax[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
ax[1].set_title(titles[1])

for i in range(len(X)):
    for j in range(len(X)):
        if MST[i, j] > 0:
            ax[2].plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'r-', alpha=0.6)
ax[2].scatter(X[:, 0], X[:, 1], color='black')
ax[2].set_title(titles[2])

for i in range(len(X)):
    for j in range(len(X)):
        if trim[i, j] > 0:
            ax[3].plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'g-', alpha=0.6)
ax[3].scatter(X[:, 0], X[:, 1], color='black')
ax[3].set_title(titles[3])

plt.tight_layout()
plt.show()
