import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = db.labels_

noise = labels == -1

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], color='gray')
plt.title("Original Data")

plt.subplot(1,2,2)
plt.scatter(X[~noise,0], X[~noise,1], c=labels[~noise], cmap='viridis', edgecolors='k')
plt.scatter(X[noise,0], X[noise,1], c='red', marker='x', label='Noise')
plt.title("DBSCAN Clustering")
plt.legend()
plt.tight_layout()
plt.show()
