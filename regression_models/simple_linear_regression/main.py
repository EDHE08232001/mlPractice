"""
Train and visualise a simple linear regression model on the Salary dataset.
Run with:  python -m regression_models.simple_linear_regression.main
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def main() -> None:
    """Main pipeline for training and visualising a simple linear regression."""
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Salary_Data.csv"
    logging.info("Loading dataset from %s", csv_path)
    dataset = pd.read_csv(csv_path)

    # Independent (X) and dependent (y) variables
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    logging.info("Dataset loaded: %d samples", len(y))

    # ------------------------------------------------------------------
    # Split into train/test
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3, random_state=0
    )
    logging.info("Training set size: %d, Test set size: %d",
                 len(y_train), len(y_test))

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    regressor = LinearRegression()
    logging.info("Training the linear regression model...")
    regressor.fit(X_train, y_train)
    logging.info("Training complete. Intercept: %.2f, Slope: %.2f",
                 regressor.intercept_, regressor.coef_[0])

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = regressor.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    logging.info("Mean Squared Error on test set: %.2f", mse)

if __name__ == "__main__":
    main()