import numpy as np
import pandas as pd
from pathlib import Path

# ===============================
# 📂 Load Dataset
# ===============================
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Social_Network_Ads.csv"
dataset = pd.read_csv(csv_path)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ===============================
# ✂️ Split Dataset
# ===============================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# ===============================
# ⚖️ Feature Scaling
# ===============================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ===============================
# 🌲 Model Training
# ===============================
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    random_state=0,
    n_jobs=-1
)
classifier.fit(X_train, y_train)

# ===============================
# 🔮 Make Single Prediction
# ===============================
sample = np.array([[30, 87000]])
scaled_sample = sc.transform(sample)
prediction = classifier.predict(scaled_sample)[0]
print(f"Prediction for [30, 87000]: {'Will Buy' if prediction == 1 else 'Will Not Buy'}")

# ===============================
# 🧠 Evaluate Model
# ===============================
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 🎨 Visualization Function
# ===============================
def visualize_decision_boundary(classifier, sc, X_data, y_data, title):
    """Reusable function for visualizing decision boundaries."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    X_set, y_set = sc.inverse_transform(X_data), y_data
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
        np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
    )

    plt.contourf(
        X1, X2,
        classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(('red', 'green'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            color=ListedColormap(('red', 'green'))(i),
            label=f'Class {j}',
            edgecolor='k'
        )

    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

# ===============================
# 🧾 Visualize Both Sets
# ===============================
visualize_decision_boundary(classifier, sc, X_train, y_train, 'Random Forest (Training Set)')
visualize_decision_boundary(classifier, sc, X_test, y_test, 'Random Forest (Test Set)')