"""
recovery_predictor.py — Recovery likelihood predictor.
Predicts whether a customer will actually pay based on conversation features.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import pickle

# ── Feature engineering ───────────────────────────────────────
INTENT_SCORE = {
    "promise_to_pay":  0.8,
    "partial_payment": 0.6,
    "needs_more_time": 0.3,
    "dispute":         0.2,
    "refusal":         0.0,
}

TONE_SCORE = {
    "cooperative":  0.9,
    "polite":       0.8,
    "worried":      0.6,
    "desperate":    0.5,
    "evasive":      0.3,
    "angry":        0.2,
    "aggressive":   0.1,
    "resigned":     0.1,
    "neutral":      0.5,
}

REGION_SCORE = {
    "bangalore": 0.7,
    "mumbai":    0.65,
    "delhi":     0.6,
    "hyderabad": 0.6,
}

LOAN_SCORE = {
    "personal loan EMI":   0.7,
    "BNPL payment":        0.6,
    "credit line EMI":     0.65,
    "bike loan EMI":       0.7,
    "mobile loan EMI":     0.65,
    "business loan EMI":   0.5,
    "education loan EMI":  0.75,
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()

    # Intent score
    features['intent_score'] = df['intent'].map(INTENT_SCORE).fillna(0.4)

    # Tone score
    features['tone_score'] = df['tone'].map(TONE_SCORE).fillna(0.5)

    # Region score
    features['region_score'] = df['region'].map(REGION_SCORE).fillna(0.6)

    # Loan score
    features['loan_score'] = df.get(
        'loan', pd.Series(['personal loan EMI'] * len(df))
    ).map(LOAN_SCORE).fillna(0.6)

    # DPD features
    dpd = df.get('dpd', pd.Series([15] * len(df))).fillna(15).astype(float)
    features['dpd_normalized'] = (dpd / 90).clip(0, 1)
    features['is_soft_dpd']    = (dpd <= 15).astype(int)
    features['is_mid_dpd']     = ((dpd > 15) & (dpd <= 30)).astype(int)
    features['is_hard_dpd']    = ((dpd > 30) & (dpd <= 60)).astype(int)
    features['is_severe_dpd']  = (dpd > 60).astype(int)

    # Amount features
    amount = df.get(
        'amount', pd.Series([5000] * len(df))
    ).fillna(5000).astype(float)
    features['amount_normalized'] = (amount / 50000).clip(0, 1)
    features['is_small_amount']   = (amount <= 5000).astype(int)
    features['is_large_amount']   = (amount > 20000).astype(int)

    # PTP features
    features['has_ptp_date'] = df.get(
        'ptp_date', pd.Series([None] * len(df))
    ).apply(lambda x: 0 if pd.isna(x) or str(x) in
            ['null', 'nan', 'none', ''] else 1)

    features['has_ptp_amount'] = df.get(
        'ptp_amount', pd.Series([None] * len(df))
    ).apply(lambda x: 0 if pd.isna(x) or str(x) in
            ['null', 'nan', 'none', ''] else 1)

    # CIBIL and legal mentions
    features['cibil_mentioned'] = df.get(
        'cibil_mentioned', pd.Series([False] * len(df))
    ).astype(int)

    features['legal_mentioned'] = df.get(
        'legal_mentioned', pd.Series([False] * len(df))
    ).astype(int)

    # Text features
    reply_len = df['reply'].fillna('').str.len()
    features['reply_length']    = (reply_len / 200).clip(0, 1)
    features['has_emoji']       = df['reply'].fillna('').str.contains(
        '🙏|😔|😢|✅|👍', regex=True
    ).astype(int)
    features['has_apology']     = df['reply'].fillna('').str.lower().str.contains(
        'sorry|maaf|galti', regex=True
    ).astype(int)
    features['has_strong_word'] = df['reply'].fillna('').str.lower().str.contains(
        'pakka|definitely|zaroor|promise|confirm', regex=True
    ).astype(int)

    # Composite score
    features['composite_score'] = (
        features['intent_score'] * 0.4 +
        features['tone_score']   * 0.2 +
        features['region_score'] * 0.1 +
        (1 - features['dpd_normalized']) * 0.2 +
        features['has_strong_word'] * 0.1
    )

    return features

def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create recovery likelihood labels from intent + DPD.
    high   = likely to pay
    medium = uncertain
    low    = unlikely to pay
    """
    labels = []
    for _, row in df.iterrows():
        intent = row.get('intent', 'unknown')
        dpd    = float(row.get('dpd', 15))
        tone   = row.get('tone', 'neutral')

        if intent == 'promise_to_pay' and dpd <= 30:
            label = 'high'
        elif intent == 'promise_to_pay' and dpd > 30:
            label = 'medium'
        elif intent == 'partial_payment':
            label = 'medium'
        elif intent == 'needs_more_time' and dpd <= 15:
            label = 'medium'
        elif intent == 'needs_more_time' and dpd > 15:
            label = 'low'
        elif intent == 'dispute':
            label = 'low'
        elif intent == 'refusal':
            label = 'low'
        else:
            label = 'medium'

        # Tone adjustment
        if tone in ['cooperative', 'polite'] and label == 'medium':
            label = 'high'
        elif tone in ['aggressive', 'resigned'] and label == 'medium':
            label = 'low'

        labels.append(label)

    return pd.Series(labels)

if __name__ == "__main__":
    print("=== VAADA Recovery Predictor ===\n")

    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Engineer features
    print("Engineering features...")
    X_train = engineer_features(train_df)
    X_val   = engineer_features(val_df)
    X_test  = engineer_features(test_df)

    # Create labels
    y_train = create_labels(train_df)
    y_val   = create_labels(val_df)
    y_test  = create_labels(test_df)

    print(f"\nLabel distribution (train):\n{y_train.value_counts()}")

    # Train model
    print("\nTraining recovery predictor...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("\n=== Validation Results ===")
    print(classification_report(y_val, y_val_pred))

    print("=== Test Results ===")
    print(classification_report(y_test, y_test_pred))

    # Feature importance
    importance = pd.DataFrame({
        'feature':    X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("=== Top Features ===")
    print(importance.head(10).to_string(index=False))

    # Save
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    with open("models/recovery_predictor.pkl", "wb") as f:
        pickle.dump(model, f)

    results = {
        "model": "GradientBoostingClassifier",
        "val_report":  classification_report(
            y_val, y_val_pred, output_dict=True
        ),
        "test_report": classification_report(
            y_test, y_test_pred, output_dict=True
        ),
        "top_features": importance.head(10).to_dict('records'),
    }

    with open("outputs/recovery_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nModel saved to models/recovery_predictor.pkl")
    print("Results saved to outputs/recovery_results.json")

    # Demo prediction
    print("\n=== Demo Predictions ===")
    demo_cases = [
        {"intent": "promise_to_pay", "tone": "polite",
         "dpd": 5,  "reply": "kal pakka kar dunga 🙏"},
        {"intent": "needs_more_time", "tone": "desperate",
         "dpd": 45, "reply": "bhai paisa nahi hai abhi"},
        {"intent": "refusal", "tone": "aggressive",
         "dpd": 70, "reply": "nahi karunga court mein milte hain"},
    ]

    for case in demo_cases:
        demo_df = pd.DataFrame([{
            **case,
            "reminder": "EMI pending hai",
            "amount": 5000,
            "region": "delhi",
        }])
        feat = engineer_features(demo_df)
        pred = model.predict(feat)[0]
        prob = model.predict_proba(feat)[0]
        print(f"Intent: {case['intent']:20} | DPD: {case['dpd']:3} | "
              f"Prediction: {pred:6} | Proba: {dict(zip(model.classes_, prob.round(2)))}")
