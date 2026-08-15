"""
train_model.py
---------------
This is the "brain building" step. We:
1. Load the labeled messages from sample_data.csv
2. Turn each message into numbers (features) using features.py
3. ALSO turn the raw text into numbers a different way (TF-IDF - explained below)
4. Combine both, and show a model lots of examples of "forward" vs "normal"
5. Test how well it learned, on messages it hasn't seen before
6. Save the trained model to a file so the demo app can use it later

Run this with: python3 train_model.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib  # used to save/load trained models to disk

from features import extract_features

# ---------------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("sample_data.csv")
print(f"Loaded {len(df)} messages ({(df['label']=='forward').sum()} forward, "
      f"{(df['label']=='normal').sum()} normal)")

# ---------------------------------------------------------------
# STEP 2: Extract hand-crafted features for every message
# ---------------------------------------------------------------
# This turns each message into a row of numbers like:
# {exclamation_count: 7, capital_ratio: 0.26, urgency_word_count: 6, word_count: 13}
print("\nExtracting features...")
feature_dicts = df["message"].apply(extract_features)
feature_df = pd.DataFrame(list(feature_dicts))
print(feature_df.head())

# ---------------------------------------------------------------
# STEP 3: Turn raw text into numbers using TF-IDF
# ---------------------------------------------------------------
# TF-IDF (Term Frequency - Inverse Document Frequency) scores each word by
# how important it is: words that appear a lot in ONE message but rarely
# elsewhere get a high score. This captures the actual words used, on top
# of the hand-crafted features above.
print("\nBuilding TF-IDF text vectors...")
vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["message"])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (messages x vocabulary words)")

# ---------------------------------------------------------------
# STEP 4: Combine hand-crafted features + TF-IDF into one dataset
# ---------------------------------------------------------------
# Scale the hand-crafted features so they're on a similar range to TF-IDF values.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_df)

# hstack "glues" the two feature sets together side by side for each message.
X = hstack([tfidf_matrix, scaled_features])
y = df["label"]

# ---------------------------------------------------------------
# STEP 5: Split into training data and test data
# ---------------------------------------------------------------
# We train on 80% of messages and test on the other 20% the model has
# NEVER seen, to check if it actually learned patterns (not just memorized).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining on {X_train.shape[0]} messages, testing on {X_test.shape[0]} messages")

# ---------------------------------------------------------------
# STEP 6: Train the model
# ---------------------------------------------------------------
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# STEP 7: Evaluate - how good is it on messages it hasn't seen?
# ---------------------------------------------------------------
predictions = model.predict(X_test)
print("\n--- Results on unseen test messages ---")
print(classification_report(y_test, predictions))
print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, predictions, labels=["forward", "normal"]))
print("Labels order: [forward, normal]")

# ---------------------------------------------------------------
# STEP 8: Save everything the demo app will need
# ---------------------------------------------------------------
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(scaler, "scaler.joblib")
print("\nSaved model.joblib, vectorizer.joblib, scaler.joblib")
print("Done! You can now run the demo app with: streamlit run app.py")
