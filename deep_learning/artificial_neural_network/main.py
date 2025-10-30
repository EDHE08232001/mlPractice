# -*- coding: utf-8 -*-
"""
Enhanced Artificial Neural Network for Churn Prediction
--------------------------------------------------------
Author: Edward He
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# ==============================
# 1. Setup Paths and Load Dataset
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', 'knowledge', 'Churn_Modelling.csv')

print(f"TensorFlow version: {tf.__version__}")
dataset = pd.read_csv(dataset_path)

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# ==============================
# 2. Encoding Categorical Data
# ==============================
# Label encode 'Gender'
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One-hot encode 'Geography'
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# ==============================
# 3. Split and Scale Data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(f"Training features: {X_train.shape}, Testing features: {X_test.shape}")

# ==============================
# 4. Build ANN Model
# ==============================
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=12, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'])

# ==============================
# 5. Train Model with Callbacks
# ==============================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history = ann.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# 6. Evaluate Model
# ==============================
y_pred = (ann.predict(X_test) > 0.5)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)
print("\nAccuracy: {:.4f}".format(acc))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 7. Visualize Training Performance
# ==============================
plt.figure(figsize=(10, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==============================
# 8. Save the Model
# ==============================
model_path = os.path.join(BASE_DIR, 'churn_ann_model.keras')
ann.save(model_path)
print(f"\n✅ Model saved at: {model_path}")