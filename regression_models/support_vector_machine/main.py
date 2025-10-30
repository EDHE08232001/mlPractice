import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Position_Salaries.csv"

dataset = pd.read_csv(csv_path)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print("X Features:", X)
print("y Target:", y)

y = y.reshape(len(y), 1)
print("Reshaped y Target:", y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print("Scaled X Features:", X)
print("Scaled y Target:", y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

print("Predicting A New Result...")
y_pred = regressor.predict(
    sc_X.transform([[6.5]]).reshape(1, -1)
)
y_restored_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))
print("Predicted Scaled Value:", y_pred)

# visualization of SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# visualization of SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.01)  # step size of 0.01
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()