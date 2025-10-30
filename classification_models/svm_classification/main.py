import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap

# --- Load dataset ---
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Social_Network_Ads.csv"
dataset = pd.read_csv(csv_path)
X = dataset.iloc[:, :-1].values   # Age, Salary
y = dataset.iloc[:, -1].values    # Purchased (0 or 1)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# --- Feature Scaling ---
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --- Grid Search for Best Hyperparameters ---
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# --- Predictions ---
y_pred = best_model.predict(X_test)

# --- Evaluation ---
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Predict a new sample ---
sample = np.array([[30, 87000]])
print("Prediction for Age=30, Salary=87000:", best_model.predict(sc.transform(sample)))
print("Probabilities:", best_model.predict_proba(sc.transform(sample)))

# --- Visualization Function ---
def plot_decision_boundary(X_set, y_set, model, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
        np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
    )
    plt.contourf(
        X1, X2,
        model.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            c=ListedColormap(('red', 'green'))(i), label=j
        )
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

# --- Visualize Training Set ---
plot_decision_boundary(sc.inverse_transform(X_train), y_train, best_model, 'SVM (Training set)')

# --- Visualize Test Set ---
plot_decision_boundary(sc.inverse_transform(X_test), y_test, best_model, 'SVM (Test set)')