import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap

# -------------------
# 1. Load Data
# -------------------
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Social_Network_Ads.csv"
dataset = pd.read_csv(csv_path)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# -------------------
# 2. Train/Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# -------------------
# 3. Feature Scaling
# -------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# -------------------
# 4. Train Model
# -------------------
K = 5
classifier = KNeighborsClassifier(n_neighbors=K, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# -------------------
# 5. Predictions & Evaluation
# -------------------
y_pred = classifier.predict(X_test)

print("y_pred vs y_test:\n", np.c_[y_pred, y_test])
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------
# 6. Helper Function for Visualization
# -------------------
def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
        np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=250)
    )
    plt.contourf(
        X1, X2,
        classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)

    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Training set visualization
plot_decision_boundary(sc.inverse_transform(X_train), y_train, 'K-NN (Training set)')

# Test set visualization
plot_decision_boundary(sc.inverse_transform(X_test), y_test, 'K-NN (Test set)')

# -------------------
# 7. Hyperparameter Tuning (try different k)
# -------------------
accuracies = []
k_values = range(1, 21)
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    model.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test)))

plt.plot(k_values, accuracies, marker='o')
plt.title("K vs Accuracy")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()