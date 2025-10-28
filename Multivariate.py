import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
X1, X2, X3 = np.random.rand(200)*10, np.random.rand(200)*5, np.random.rand(200)*8
y = 3*X1 + 2*X2 - 1.5*X3 + 10 + np.random.randn(200)*2
X = np.column_stack([X1, X2, X3])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = LinearRegression().fit(Xtr, ytr)
yp = m.predict(Xte)

print("Coefficients:", np.round(m.coef_, 2))
print("Intercept:", round(m.intercept_, 2))
print("R²:", round(r2_score(yte, yp), 3))

plt.scatter(yte, yp, c='blue', alpha=0.6)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--')
plt.title("Predicted vs Actual")
plt.show()

plt.scatter(yte, yp, color='green', alpha=0.6, label='Predicted vs Actual')
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--', label='Ideal Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual (Perfect Fit Line in Red)")
plt.legend()
plt.show()
