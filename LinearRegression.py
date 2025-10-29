import numpy as np
from sklearn.linear_model import LinearRegression

n = int(input("Enter number of data points: "))
x = []
y = []

for i in range(n):
    xi = float(input(f"Enter X{i+1}: "))
    yi = float(input(f"Enter Y{i+1}: "))
    x.append([xi])
    y.append(yi)

x = np.array(x)
y = np.array(y)

model = LinearRegression()
model.fit(x, y)

print("\nCoefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)

pred = float(input("\nEnter X value to predict Y: "))
print("Predicted Y =", model.predict([[pred]])[0])

y_pred = model.predict(x)
for i in range(n):
    print(f"X={x[i][0]}, Actual Y={y[i]}, Predicted Y={round(y_pred[i],2)}")
