import numpy as np
from statsmodels.multivariate.pca import PCA

X = np.array([
    [2.5, 2.4, 3.5],
    [1.0, 0.5, 1.2],
    [2.2, 2.9, 3.1],
    [1.9, 2.2, 2.8],
    [3.1, 3.0, 3.9]
])

model = PCA(X, ncomp=2)
print("PCA Components:\n", model.loadings)
print("Explained Variance:\n", model.eigenvals)

U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
print("\nSVD Singular Values:\n", S)

classes = np.array([0, 0, 0, 1, 1])
means = [np.mean(X[classes == c], axis=0) for c in np.unique(classes)]
S_w = sum([np.cov(X[classes == c].T) * (sum(classes == c) - 1) for c in np.unique(classes)])
diff = means[0] - means[1]
w = np.dot(np.linalg.inv(S_w), diff)
print("\nLDA Direction:\n", w)
