import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

def load_dataset(csv_filename: str):
    """Load dataset from CSV file."""
    csv_path = Path(__file__).resolve().parent.parent / "dataset" / csv_filename
    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

def preprocess_data(X, y, test_size=0.25, random_state=0):
    """Split data into training and test sets and apply feature scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, sc

def train_decision_tree(X_train, y_train, criterion='entropy', random_state=0):
    """Train a Decision Tree Classifier."""
    classifier = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_model(classifier, X_test, y_test):
    """Evaluate model performance and print metrics."""
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 Evaluation Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)
    return y_pred

def visualize_results(classifier, sc, X_scaled, y, title):
    """Visualize classification results for given dataset."""
    X_set, y_set = sc.inverse_transform(X_scaled), y
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
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            color=ListedColormap(('red', 'green'))(i),
            label=j, edgecolor='k'
        )
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

def main():
    X, y = load_dataset("Social_Network_Ads.csv")
    X_train, X_test, y_train, y_test, sc = preprocess_data(X, y)

    print("🧠 Training Decision Tree Classifier...")
    classifier = train_decision_tree(X_train, y_train)

    print("\n🔮 Predicting a new example [Age=30, Salary=87000]:")
    prediction = classifier.predict(sc.transform([[30, 87000]]))
    print("Prediction:", "Will Buy" if prediction[0] == 1 else "Will Not Buy")

    print("\n✅ Evaluating model on test data...")
    y_pred = evaluate_model(classifier, X_test, y_test)

    # Visualization
    print("\n📈 Visualizing training set results...")
    visualize_results(classifier, sc, X_train, y_train, "Decision Tree Classification (Training Set)")
    print("\n📈 Visualizing test set results...")
    visualize_results(classifier, sc, X_test, y_test, "Decision Tree Classification (Test Set)")

if __name__ == "__main__":
    main()