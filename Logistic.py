import numpy as np
from sklearn.linear_model import LogisticRegression

n = int(input("Enter number of data points: "))
x1, x2, y = [], [], []

for i in range(n):
    a = float(input(f"Enter X1[{i+1}]: "))
    b = float(input(f"Enter X2[{i+1}]: "))
    c = int(input(f"Enter Class Y[{i+1}] (0 or 1): "))
    x1.append(a)
    x2.append(b)
    y.append(c)

X = np.column_stack((x1, x2))
y = np.array(y)

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_[0])

print("\nX1\tX2\tActual\tPredicted\tProbability")
for i in range(n):
    print(f"{x1[i]:.2f}\t{x2[i]:.2f}\t{y[i]}\t{y_pred[i]}\t\t{y_prob[i]:.3f}")

a = float(input("\nEnter new X1: "))
b = float(input("Enter new X2: "))
pred_class = model.predict([[a, b]])[0]
pred_prob = model.predict_proba([[a, b]])[0][1]
print(f"\nPredicted Class = {pred_class}, Probability = {round(pred_prob, 3)}")
