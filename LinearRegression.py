import numpy as np
from sklearn.linear_model import LinearRegression

# Take input from user
n = int(input("Enter number of data points: "))
x = []
y = []

for i in range(n):
    xi = float(input(f"Enter X[{i+1}]: "))
    yi = float(input(f"Enter Y[{i+1}]: "))
    x.append([xi])
    y.append(yi)

# Convert to numpy arrays
x = np.array(x)
y = np.array(y)

# Train Linear Regression model
model = LinearRegression()
model.fit(x, y)

# Get coefficients
b0 = model.intercept_
b1 = model.coef_[0]

# Display regression equation
print(f"\nRegression Equation: Y = {round(b0, 2)} + {round(b1, 2)} * X")

# Predict new value
new_x = float(input("\nEnter X value to predict Y: "))
pred_y = model.predict([[new_x]])[0]

# Show only prediction
print(f"Predicted Y for X = {new_x} is: {round(pred_y, 2)}")
