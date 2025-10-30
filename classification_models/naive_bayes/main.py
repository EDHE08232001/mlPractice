# =========================================================
# 🧭 Naive Bayes Classification — Social Network Ads
# =========================================================
# Author: Edward He
# Purpose:
#   • Train a Gaussian Naive Bayes classifier on user age & salary
#   • Predict purchase decision
#   • Visualize training/test results with decision boundaries
# =========================================================

# === 1. Imports ==============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path


# === 2. Helper Functions =====================================================
def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load Social_Network_Ads dataset safely using project-relative path."""
    print("📥 Loading dataset...")

    csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Social_Network_Ads.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found at: {csv_path}\n", "Please ensure the file exists in /dataset/")

    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, :-1].values   # Age, Estimated Salary
    y = dataset.iloc[:, -1].values    # Purchased (0/1)

    print(f"✅ Dataset loaded from: {csv_path}")
    print(f"   → {dataset.shape[0]} samples, {dataset.shape[1]} features")
    return X, y


def plot_decision_boundary(title: str, X_set, y_set, classifier, sc):
    """Visualize decision boundary for a trained classifier."""
    cmap = ListedColormap(("red", "green"))
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
        np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25),
    )
    plt.contourf(
        X1, X2,
        classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
        alpha=0.75, cmap=cmap
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=cmap(i), label=j, edgecolor="k", s=25)

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.tight_layout()
    plt.show()


# === 3. Main Workflow ========================================================
def main():
    # Load and preprocess
    X, y = load_dataset()
    print("✂️ Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    print("⚙️ Scaling features...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train model
    print("🧠 Training Gaussian Naive Bayes classifier...")
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predictions
    print("🔮 Predicting new result [Age=30, Salary=87000] ...")
    new_pred = classifier.predict(sc.transform([[30, 87000]]))
    print(f"   → {'Will Purchase' if new_pred[0] else 'Will Not Purchase'}")

    print("📊 Predicting test set results...")
    y_pred = classifier.predict(X_test)

    # Evaluation
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("\n📈 Evaluation Summary")
    print("----------------------------")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.4f}")
    print("----------------------------")

    # Visualization
    print("🎨 Visualizing results...")
    plot_decision_boundary("Naive Bayes (Training set)", sc.inverse_transform(X_train), y_train, classifier, sc)
    plot_decision_boundary("Naive Bayes (Test set)", sc.inverse_transform(X_test), y_test, classifier, sc)


# === 4. Entry Point ==========================================================
if __name__ == "__main__":
    main()