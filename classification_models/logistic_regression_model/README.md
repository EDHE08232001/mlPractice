# 📓 Study Note: Logistic Regression with Python (Classification Example)

## 1. Problem Setup

* Dataset: **Social_Network_Ads.csv**
* Features: `Age`, `Estimated Salary`
* Target: `Purchased` (0 = No, 1 = Yes)
* Goal: Predict whether a person will purchase based on age & salary.

---

## 2. Workflow

### Step 1: Import libraries

* `numpy`, `pandas`, `matplotlib` → handling data & visualization.
* `sklearn` modules → splitting, scaling, modeling, evaluation.

---

### Step 2: Data Preprocessing

1. **Load data**

   ```python
   dataset = pd.read_csv(csv_path)
   X = dataset.iloc[:, :-1].values
   y = dataset.iloc[:, -1].values
   ```

   * `X` = features (age, salary)
   * `y` = labels (0/1 purchase)

2. **Train-Test Split**

   ```python
   train_test_split(X, y, test_size=0.25, random_state=0)
   ```

   * 25% of data is test, 75% is training.

3. **Feature Scaling**

   ```python
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   ```

   * Standardizes features to mean 0, variance 1.
   * 🚩 **Why important?** Logistic regression is distance-based; large-scale features (like salary) would dominate without scaling.

---

### Step 3: Model Training

* Logistic regression classifier:

  ```python
  classifier = LogisticRegression(random_state=0)
  classifier.fit(X_train, y_train)
  ```

* Learns a **decision boundary** between classes.

---

### Step 4: Predictions & Evaluation

1. Predict test set:

   ```python
   y_pred = classifier.predict(X_test)
   ```

2. **Confusion Matrix & Accuracy**

   ```python
   confusion_matrix(y_test, y_pred)
   accuracy_score(y_test, y_pred)
   ```

   Example output:

   ```
   [[65  3]
    [ 8 24]]
   Accuracy = 0.89
   ```

   * 65 true negatives
   * 24 true positives
   * 3 false positives
   * 8 false negatives

   → Accuracy = 89%

---

### Step 5: Visualization

* Decision boundaries are shown using `plt.contourf`.
* Training & test sets plotted separately.
* Classes are visualized in **red** (0) and **green** (1).
* Fixed scatter plotting:

  ```python
  plt.scatter(..., color=('red' if i == 0 else 'green'))
  ```

---

## 3. Key Concepts Recap

* **Logistic Regression**

  * A linear model that outputs probabilities using the **sigmoid function**.
  * Decision rule: predict 1 if `p ≥ 0.5`, else 0.

* **Why Scaling?**

  * Prevents features like “salary” (thousands) from overshadowing “age” (tens).
  * Ensures smoother convergence.

* **Confusion Matrix** helps us see not just accuracy, but also the breakdown of errors.

---

## 4. Takeaway

* Logistic regression works well for **binary classification**.
* Visualization is helpful to **see the decision boundary**.
* Accuracy isn’t everything → look at false positives/negatives too.