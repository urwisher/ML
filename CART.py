from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(criterion='gini').fit(X, y)

plt.figure(figsize=(8,5))
plot_tree(model, filled=True, feature_names=load_iris().feature_names)
plt.show()
