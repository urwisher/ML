import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

n = int(input("Enter number of data points: "))
x1, x2, y = [], [], []

for i in range(n):
    a = float(input(f"Enter X1[{i+1}]: "))
    b = float(input(f"Enter X2[{i+1}]: "))
    c = float(input(f"Enter Y[{i+1}]: "))
    x1.append(a)
    x2.append(b)
    y.append(c)

# Convert to numpy arrays
X = np.column_stack((x1, x2))
y = np.array(y)

# Train model
model = LinearRegression()
model.fit(X, y)

# Display coefficients and intercept
print("\nCoefficients (b1, b2):", model.coef_)
print("Intercept (b0):", model.intercept_)

# Predictions and R2 Score
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("\nR² Score:", round(r2, 4))

# Prediction for new values
a = float(input("\nEnter new X1: "))
b = float(input("Enter new X2: "))
pred = model.predict([[a, b]])
print("Predicted Y =", round(pred[0], 2))

# Display comparison table
print("\nX1\tX2\tActualY\tPredY")
for i in range(n):
    print(f"{x1[i]}\t{x2[i]}\t{y[i]}\t{round(y_pred[i],2)}")
