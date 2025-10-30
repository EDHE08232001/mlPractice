# Support Vector Regression (SVR) - Study Notes

## Overview
Support Vector Regression (SVR) is a regression algorithm that uses the same principles as Support Vector Machines (SVM) for classification. It aims to find a function that deviates from the actual observed targets by a value no greater than a small amount (ε) while being as flat as possible.

## Key Concepts

### 1. **Kernel Trick**
- SVR uses kernel functions to transform data into higher dimensions
- `kernel='rbf'` (Radial Basis Function) is commonly used for non-linear regression
- Other kernels: linear, polynomial, sigmoid

### 2. **ε-Insensitive Tube**
- SVR tries to fit the best line within a margin of tolerance (ε)
- Points outside this tube are considered errors
- The algorithm minimizes these errors while maximizing the margin

### 3. **Feature Scaling Importance**
- SVR is sensitive to feature scales because it relies on distance calculations
- **Always scale features** before applying SVR

## Code Analysis

### Data Preparation
```python
# Reshape y to 2D array for scaling
y = y.reshape(len(y), 1)

# Standardize features and target
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
```

### Model Training
```python
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
```

### Prediction Process
```python
# 1. Scale the input
scaled_input = sc_X.transform([[6.5]]).reshape(1, -1)

# 2. Make prediction (returns scaled value)
y_pred = regressor.predict(scaled_input)

# 3. Inverse transform to get actual value
y_restored_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))
```

## Important Observations from Your Output

### Data Transformation
- **Original X**: Position levels 1-10
- **Original y**: Salaries from $45,000 to $1,000,000
- **Scaled values**: Centered around 0 with unit variance

### Warning Messages
1. **DataConversionWarning**: 
   - **Cause**: y was passed as 2D array but 1D expected
   - **Solution**: Use `y.ravel()` or ensure y is 1D
   ```python
   regressor.fit(X, y.ravel())  # Recommended fix
   ```

2. **DeprecationWarning**:
   - **Cause**: `min()`/`max()` on arrays returning arrays instead of scalars
   - **Solution**: Use `.item()` or `.flat`
   ```python
   X_grid = np.arange(
       min(sc_X.inverse_transform(X)).item(), 
       max(sc_X.inverse_transform(X)).item(), 
       0.01
   )
   ```

## Improved Code Version

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Data loading and preparation
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Position_Salaries.csv"
dataset = pd.read_csv(csv_path)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).ravel()  # Fix: use ravel()

# Model training
regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, y_scaled)

# Prediction
new_data = sc_X.transform([[6.5]])
y_pred_scaled = regressor.predict(new_data)
y_pred_actual = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

print(f"Predicted salary for level 6.5: ${y_pred_actual[0,0]:,.2f}")

# Visualization (fixed version)
X_actual = sc_X.inverse_transform(X_scaled)
y_actual = sc_y.inverse_transform(y_scaled.reshape(-1, 1))

# Create smooth curve
X_min = sc_X.inverse_transform(X_scaled).min()
X_max = sc_X.inverse_transform(X_scaled).max()
X_grid = np.arange(X_min, X_max, 0.01).reshape(-1, 1)
X_grid_scaled = sc_X.transform(X_grid)
y_grid_pred = regressor.predict(X_grid_scaled)
y_grid_actual = sc_y.inverse_transform(y_grid_pred.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.scatter(X_actual, y_actual, color='red', label='Actual Data')
plt.plot(X_grid, y_grid_actual, color='blue', label='SVR Prediction')
plt.title('Support Vector Regression Results')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()
```

## Key Takeaways

### Advantages of SVR
- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Versatile through kernel functions

### Limitations
- Doesn't perform well with large datasets
- Sensitive to noisy data
- Requires careful parameter tuning

### When to Use SVR
- Small to medium-sized datasets
- Non-linear relationships
- When interpretability is less important than performance

### Best Practices
1. Always scale features before SVR
2. Experiment with different kernels
3. Tune hyperparameters (C, epsilon, gamma)
4. Use cross-validation for parameter optimization

This implementation demonstrates a complete SVR workflow from data loading to visualization, highlighting both the power and nuances of Support Vector Regression.