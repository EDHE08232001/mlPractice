# -*- coding: utf-8 -*-
"""
Restaurant Review Sentiment Analysis
------------------------------------
A classic NLP pipeline demonstrating:
    1. Data loading and preprocessing
    2. Text cleaning and stemming
    3. Feature extraction using Bag of Words
    4. Training a Naive Bayes classifier
    5. Evaluating model performance

Author: Edward He
Environment: macOS (zsh, Python 3.x)
"""

# ============================================================
# 1️⃣ IMPORT LIBRARIES
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# 2️⃣ PATH SETUP
# ============================================================

# Define base directory and dataset path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'knowledge', 'Restaurant_Reviews.tsv')


# ============================================================
# 3️⃣ LOAD DATA
# ============================================================

# Load the dataset (tab-separated file, no quoted strings)
dataset = pd.read_csv(dataset_path, delimiter='\t', quoting=3)
print(f"✅ Dataset loaded successfully: {dataset.shape[0]} reviews found.")


# ============================================================
# 4️⃣ TEXT CLEANING AND PREPROCESSING
# ============================================================

# Download stopwords if not already available
nltk.download('stopwords', quiet=True)

# Initialize objects
ps = PorterStemmer()
all_stopwords = stopwords.words('english')

# Keep the word "not" since it affects sentiment polarity
if 'not' in all_stopwords:
    all_stopwords.remove('not')

corpus = []  # will contain all cleaned reviews

# Clean each review
for i in range(len(dataset)):
    review = dataset['Review'][i]
    review = re.sub('[^a-zA-Z]', ' ', review)  # keep letters only
    review = review.lower().split()             # convert to lowercase and split
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    corpus.append(' '.join(review))

# Preview the cleaned data
print("\n🧠 Corpus sample (first 5 cleaned reviews):")
for i, example in enumerate(corpus[:5]):
    print(f"{i+1}. {example}")


# ============================================================
# 5️⃣ FEATURE EXTRACTION (BAG OF WORDS MODEL)
# ============================================================

# Convert text corpus into numerical feature vectors
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📈 Labels shape: {y.shape}")


# ============================================================
# 6️⃣ SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

print(f"\n🧩 Training samples: {len(X_train)}")
print(f"🧪 Testing samples: {len(X_test)}")


# ============================================================
# 7️⃣ TRAIN MODEL (NAIVE BAYES)
# ============================================================

classifier = GaussianNB()
classifier.fit(X_train, y_train)
print("\n🚀 Model training complete.")


# ============================================================
# 8️⃣ PREDICTION AND EVALUATION
# ============================================================

y_pred = classifier.predict(X_test)

# Combine predictions and true values for inspection
results = np.concatenate(
    (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1
)
print("\n🔮 Predictions vs Actuals (sample):")
print(results[:10])  # show first 10 comparisons

# Confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\n📋 Confusion Matrix:\n", cm)
print(f"🎯 Accuracy Score: {acc:.4f}")


# ============================================================
# 9️⃣ CONCLUSION
# ============================================================

"""
Summary:
---------
✔ Successfully trained a Naive Bayes sentiment classifier.
✔ Text cleaned, stemmed, and vectorized.
✔ Accuracy typically ~70–80% for this dataset.
✔ Foundation for experimenting with TF-IDF, Word2Vec, or BERT embeddings.
"""