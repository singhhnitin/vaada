"""
baseline.py — TF-IDF + Logistic Regression baseline intent classifier.
Establishes the benchmark F1 score that VAADA fine-tuned model will beat.
"""

import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)
from sklearn.pipeline import Pipeline
import pickle
import os

# ── Load data ─────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"\nIntent distribution:\n{train_df['intent'].value_counts()}")

# ── Feature: combine reminder + reply ─────────────────────────
def combine_text(df):
    return df['reminder'].fillna('') + ' [SEP] ' + df['reply'].fillna('')

X_train = combine_text(train_df)
y_train = train_df['intent']

X_val   = combine_text(val_df)
y_val   = val_df['intent']

X_test  = combine_text(test_df)
y_test  = test_df['intent']

# ── Build pipeline ────────────────────────────────────────────
print("\nTraining baseline classifier...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        sublinear_tf=True,
        analyzer='char_wb',
        min_df=2,
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        C=5.0,
        class_weight='balanced',
        random_state=42,
    ))
])

pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
print("\n=== Validation Results ===")
y_val_pred = pipeline.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy : {val_acc:.4f}")
print(f"F1 Score : {val_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\n=== Test Results ===")
y_test_pred = pipeline.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy : {test_acc:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_test_pred))

# ── Save results ──────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

results = {
    "model": "TF-IDF + Logistic Regression (Baseline)",
    "val_accuracy":  round(val_acc, 4),
    "val_f1":        round(val_f1, 4),
    "test_accuracy": round(test_acc, 4),
    "test_f1":       round(test_f1, 4),
    "per_class": classification_report(
        y_test, y_test_pred, output_dict=True
    )
}

with open("outputs/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("models/baseline_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nResults saved to outputs/baseline_results.json")
print("Model saved to models/baseline_pipeline.pkl")
print(f"\nBaseline F1: {test_f1:.4f} — VAADA fine-tuned model must beat this")
