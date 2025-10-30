"""
===============================================================================
🎯 Kernel SVM Classification — Social Network Ads Dataset
Author: Edward He, University of Ottawa
Environment: macOS (Apple Silicon, Python 3.12, scikit-learn)
===============================================================================
"""

# === 1. Imports ==============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap


# === 2. Data Loading =========================================================
def load_dataset():
    """Load Social Network Ads dataset from ../dataset/."""
    csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Social_Network_Ads.csv"
    dataset = pd.read_csv(csv_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y


# === 3. Data Preprocessing ===================================================
def preprocess_data(X, y, test_size=0.25, random_state=0):
    """Split, scale, and return train-test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, sc


# === 4. Model Training & Prediction ==========================================
def train_and_predict(X_train, y_train, X_test):
    """Train SVM classifier (RBF kernel) and predict on test set."""
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return classifier, y_pred


# === 5. Evaluation ===========================================================
def evaluate_model(y_test, y_pred):
    """Compute and print confusion matrix and accuracy."""
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("🔹 Confusion Matrix:\n", cm)
    print(f"🔹 Accuracy: {acc:.4f}\n")

    # Optional: print predictions side by side
    print("🔹 Predictions vs Actuals:")
    print(np.column_stack((y_pred, y_test)))


# === 6. Visualization ========================================================
def visualize_results(X, y, classifier, sc, title):
    """Plot decision boundary for SVM classifier."""
    X_set, y_set = sc.inverse_transform(X), y
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
        np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25)
    )

    # Decision boundary
    plt.contourf(
        X1, X2,
        classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(("red", "green"))
    )

    # Plot points
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=ListedColormap(("red", "green"))(i), label=f"Class {j}", edgecolor="k", s=40)
    
    plt.title(f"Kernel SVM ({title})")
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.show()


# === 7. Main Execution =======================================================
def main():
    """Main routine: load, preprocess, train, evaluate, visualize."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test, sc = preprocess_data(X, y)

    classifier, y_pred = train_and_predict(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred)

    visualize_results(X_train, y_train, classifier, sc, "Training Set")
    visualize_results(X_test, y_test, classifier, sc, "Test Set")


if __name__ == "__main__":
    main()