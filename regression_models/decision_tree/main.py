import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Position_Salaries.csv"

dataset = pd.read_csv(csv_path)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

print("Predicting A New Result...")
y_pred = regressor.predict([[6.5]])
print("Predicted Value:", y_pred)

# visualization of Decision Tree Regression results (higher resolution)
X_grid = np.arange(X.min(), X.max(), 0.01)  # step
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()