import numpy as np
from scipy.optimize import minimize

def svm_train(X, y):
    n_samples, n_features = X.shape

    K = np.dot(X, X.T)

    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(np.outer(y, y) * K, alpha)) - np.sum(alpha)

    cons = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
    bounds = [(0, None)] * n_samples  # alpha_i >= 0

    res = minimize(objective, np.zeros(n_samples), bounds=bounds, constraints=cons)
    alpha = res.x

    w = np.dot(alpha * y, X)
    b = np.mean(y - np.dot(X, w))

    return w, b

def svm_predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

X = np.array([[1,1], [2,2], [3,3], [6,6], [7,7], [8,8]])
y = np.array([-1,-1,-1,1,1,1])

w, b = svm_train(X, y)
preds = svm_predict(X, w, b)

print("Weights:", w)
print("Bias:", b)
print("Predictions:", preds)
