# 📚 Study Note: Decision Tree Regression in Python

*(with Mathematical Foundations)*

## 1. Objective

Train a **Decision Tree Regressor** to predict salaries from position levels, showcasing non-linear regression and the tree’s characteristic **piecewise-constant** predictions.

---

## 2. Environment & Setup

* **Packages**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
* **Run**:

  ```bash
  python3 -m regression_models.decision_tree.main
  ```
* **Model Output**:

  ```
  Predicting A New Result...
  Predicted Value: [150000.]
  ```

---

## 3. Data Handling

```python
dataset = pd.read_csv(csv_path)
X = dataset.iloc[:, 1:-1].values  # Position level
y = dataset.iloc[:, -1].values    # Salary
```

* `X`: shape `(n_samples, 1)`
* `y`: salary targets (1-D array).

⚠️ Keep `X` two-dimensional when fitting scikit-learn.

---

## 4. Model Training

```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
```

* **DecisionTreeRegressor** splits the input space recursively into **regions** where the target is predicted by a **constant value** (mean of that region’s training samples).
* `random_state=0` → deterministic splits.

---

## 5. Prediction

```python
y_pred = regressor.predict([[6.5]])
```

Predicts **$150 000** salary for position level **6.5**.
🔹 Always pass a 2-D array to `.predict()`.

---

## 6. Visualization

```python
X_grid = np.arange(X.min(), X.max(), 0.01).reshape(-1, 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
```

* **Red points**: actual data.
* **Blue line**: tree’s stepwise prediction.

---

## 7. Mathematics Behind Decision Tree Regression

At the heart of a decision tree is a **recursive binary partitioning** of the feature space.

### 7.1 Problem Setup

Given data:
[
{(\mathbf{x}*i, y_i)}*{i=1}^N
]
with features (\mathbf{x}_i \in \mathbb{R}^d) and target (y_i \in \mathbb{R}).

We want a predictor (f(\mathbf{x})) that minimizes the empirical squared error:
[
\min_{f} \frac{1}{N} \sum_{i=1}^N \big(y_i - f(\mathbf{x}_i)\big)^2.
]

Decision trees restrict (f) to be **piecewise constant**:
[
f(\mathbf{x}) = \sum_{m=1}^M c_m , \mathbf{1}_{{\mathbf{x} \in R_m}},
]
where:

* (R_m) are disjoint rectangular regions (leaves),
* (c_m) is a constant prediction for region (R_m).

The goal is to choose regions (R_m) and constants (c_m) to minimize the squared error.

---

### 7.2 Optimal Constant in a Region

For any region (R),
[
\min_{c} \sum_{x_i \in R} (y_i - c)^2
]
is minimized when
[
c^* = \frac{1}{|R|} \sum_{x_i \in R} y_i,
]
the **mean of targets** inside that region.

---

### 7.3 Splitting Criterion

Starting with the full feature space, the algorithm chooses the best split:

* Pick a feature (j) and threshold (s).
* Divide into left (R_1 = {x: x_j \le s}) and right (R_2 = {x: x_j > s}).

Minimize the **total variance** (sum of squared errors):
[
\text{SSE}(j,s) =
\sum_{x_i \in R_1} (y_i - \bar{y}_{R_1})^2

* \sum_{x_i \in R_2} (y_i - \bar{y}*{R_2})^2.
  ]
  Choose ((j^*, s^*)) that gives the **largest reduction**:
  [
  \Delta \text{SSE} =
  \text{SSE}*{\text{parent}} -
  \big(\text{SSE}(j,s)\big).
  ]
  This is equivalent to maximizing **variance reduction**.

The process repeats recursively until a stopping rule is reached:

* maximum depth,
* minimum samples per leaf,
* or no further variance reduction.

---

### 7.4 Final Predictor

The resulting predictor is:
[
\hat{f}(\mathbf{x}) =
\sum_{m=1}^M \bar{y}*{R_m} , \mathbf{1}*{{\mathbf{x} \in R_m}}.
]
Each region (R_m) corresponds to a leaf node and its prediction is the mean target of its training samples.

---

## 8. Hyper-parameters to Control Complexity

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

Tuning these mitigates overfitting and controls tree size.

---

## 9. Technical Note on NumPy

Use:

```python
np.arange(X.min(), X.max(), 0.01)
```

not `min(X)` / `max(X)` to avoid deprecation warnings in NumPy ≥1.25.

---

## 10. Key Takeaways

* Decision Tree Regression produces **piecewise constant** predictions, capturing abrupt changes but lacking smooth interpolation.
* The algorithm greedily splits features to **maximize variance reduction**, with each leaf predicting the **mean target** of its region.
* For smoother or more robust results, consider ensemble methods like **Random Forests** or **Gradient Boosting**.

---

### Reflection

The decision tree is like a seasoned magistrate: it surveys the realm of data, carves it into provinces (regions), and rules each with a single decree—the mean. No curves, no waffling—just decisive partition and judgment.