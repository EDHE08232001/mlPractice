import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load the dataset
# -----------------------------
csv_path = Path(__file__).resolve().parent.parent / "dataset" / "50_Startups.csv"
dataset = pd.read_csv(csv_path)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# -----------------------------
# Encode the categorical column (State)
# -----------------------------
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [3])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# -----------------------------
# Split the data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------------
# Train the model
# -----------------------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# -----------------------------
# Predict and inspect
# -----------------------------
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

print("Model intercept:", regressor.intercept_)
print("Model coefficients:", regressor.coef_)