import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Position_Salaries.csv"
dataset = pd.read_csv(csv_path)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# visualize linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualize polynomial regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualize polynomial regression results (higher resolution and smoother curve)
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1)) # reshape to a column vector
plt.scatter(X, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predict a new result with linear regression
print(
    f"Linear Regression Prediction for 6.5: {lin_reg.predict([[6.5]])}"
)

# predict a new result with polynomial regression
print(
    f"Polynomial Regression Prediction for 6.5: {lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))}"
)