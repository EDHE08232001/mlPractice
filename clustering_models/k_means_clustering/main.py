# =========================
# 📁 1. Path Setup
# =========================
import os

BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.abspath(os.path.join(BASE_DIR, "../dataset/Mall_Customers.csv"))


# =========================
# 📊 2. Data Loading
# =========================
import pandas as pd

try:
    dataset = pd.read_csv(dataset_path)
    print(f"✅ Loaded dataset from: {dataset_path}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Dataset not found at {dataset_path}")

# Select only the relevant features (Annual Income and Spending Score)
X = dataset.iloc[:, [3, 4]].values


# =========================
# ⚙️ 3. Find Optimal k (Elbow Method)
# =========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=10,              # explicitly set for reproducibility
        max_iter=300,           # safe default
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True, linestyle=':')
plt.show()


# =========================
# 🤖 4. Train K-Means Model
# =========================
optimal_k = 5  # determined visually from elbow curve
kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
y_kmeans = kmeans.fit_predict(X)


# =========================
# 🎨 5. Visualize Clusters
# =========================
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = [f'Cluster {i+1}' for i in range(optimal_k)]

for i, color, label in zip(range(optimal_k), colors, labels):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=color, label=label)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    edgecolors='black',
    label='Centroids'
)

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()