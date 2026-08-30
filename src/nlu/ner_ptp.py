"""
ner_ptp.py — CRF-based PTP date and amount extractor.
Replaces regex baseline. Target: 80%+ recall on both.
"""

import re
import json
import pickle
import os
import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_val_score

# ── Feature extraction ────────────────────────────────────────
DATE_KEYWORDS = {
    "kal","parso","aaj","shaam","friday","saturday","monday",
    "tuesday","wednesday","thursday","sunday","shukravar","somvar",
    "week","hafte","tarikh","mahine","weekend","tomorrow","today",
    "next","agla","din","parson","evening","morning","night"
}

AMOUNT_KEYWORDS = {
    "rs","rupees","rupaye","paisa","amount","payment","emi",
    "half","aadha","poora","total","baaki","remaining","rest"
}

HINDI_COMMIT = {
    "pakka","definitely","zaroor","promise","confirm","bilkul","100"
}

def word2features(sent, i):
    word = sent[i][0].lower()
    features = {
        "word.lower":        word,
        "word.isdigit":      word.isdigit(),
        "word.hasdigit":     any(c.isdigit() for c in word),
        "word.length":       len(word),
        "word.isdate":       word in DATE_KEYWORDS,
        "word.isamount":     word in AMOUNT_KEYWORDS,
        "word.iscommit":     word in HINDI_COMMIT,
        "word.hasrupee":     "₹" in word or "rs" in word.lower(),
        "word.hasnumber":    bool(re.search(r'\d', word)),
        "word.prefix3":      word[:3] if len(word) >= 3 else word,
        "word.suffix3":      word[-3:] if len(word) >= 3 else word,
        "word.hastarikh":    "tarikh" in word,
        "word.hasdin":       "din" in word,
        "word.hashafte":     "hafte" in word,
    }

    if i > 0:
        prev = sent[i-1][0].lower()
        features.update({
            "prev.lower":    prev,
            "prev.isdigit":  prev.isdigit(),
            "prev.isdate":   prev in DATE_KEYWORDS,
            "prev.isamount": prev in AMOUNT_KEYWORDS,
        })
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        nxt = sent[i+1][0].lower()
        features.update({
            "next.lower":    nxt,
            "next.isdigit":  nxt.isdigit(),
            "next.isdate":   nxt in DATE_KEYWORDS,
            "next.isamount": nxt in AMOUNT_KEYWORDS,
        })
    else:
        features["EOS"] = True

    if i > 1:
        features["prev2.lower"] = sent[i-2][0].lower()
    if i < len(sent) - 2:
        features["next2.lower"] = sent[i+2][0].lower()

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

# ── Tokenizer ─────────────────────────────────────────────────
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'₹\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?|[a-zA-Z\u0900-\u097F]+|[^\s]', text)
    return tokens

# ── Label creator ─────────────────────────────────────────────
def label_tokens(tokens, ptp_date, ptp_amount):
    labels = ["O"] * len(tokens)
    text = " ".join(tokens)

    # Label date tokens
    if ptp_date and str(ptp_date) not in ["null","nan","none",""]:
        date_str = str(ptp_date).lower().strip()
        for i, tok in enumerate(tokens):
            if tok in DATE_KEYWORDS or (len(tok) >= 2 and tok.isdigit()):
                if "tarikh" in text[max(0,text.find(tok)-10):text.find(tok)+20]:
                    labels[i] = "B-DATE"
                elif tok in DATE_KEYWORDS:
                    labels[i] = "B-DATE"

    # Label amount tokens
    if ptp_amount and str(ptp_amount) not in ["null","nan","none",""]:
        try:
            amt_val = float(str(ptp_amount).replace(",",""))
            amt_str = str(int(amt_val))
            for i, tok in enumerate(tokens):
                tok_clean = tok.replace("₹","").replace(",","")
                if tok_clean == amt_str or tok.startswith("₹"):
                    labels[i] = "B-AMOUNT"
                    if i > 0 and tokens[i-1].lower() in ["rs","rs.","rupees"]:
                        labels[i-1] = "B-AMOUNT"
        except Exception:
            pass

    return labels

# ── Build dataset ─────────────────────────────────────────────
def build_dataset(df):
    sentences = []
    for _, row in df.iterrows():
        text = str(row.get("reply",""))
        if not text or len(text) < 3:
            continue

        tokens = tokenize(text)
        if not tokens:
            continue

        ptp_date   = row.get("ptp_date","")
        ptp_amount = row.get("ptp_amount","")

        labels = label_tokens(tokens, ptp_date, ptp_amount)
        sent   = list(zip(tokens, labels))
        sentences.append(sent)

    return sentences

# ── Train ─────────────────────────────────────────────────────
def train(train_sents):
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s)   for s in train_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf

# ── Predict ───────────────────────────────────────────────────
def predict_ptp(crf, text):
    tokens = tokenize(text.lower())
    if not tokens:
        return {"date": None, "amount": None}

    sent     = [(tok, "O") for tok in tokens]
    features = sent2features(sent)
    labels   = crf.predict([features])[0]

    date_tokens   = [tokens[i] for i, l in enumerate(labels) if "DATE" in l]
    amount_tokens = [tokens[i] for i, l in enumerate(labels) if "AMOUNT" in l]

    date   = " ".join(date_tokens) if date_tokens else None
    amount = None
    for tok in amount_tokens:
        clean = tok.replace("₹","").replace(",","")
        if clean.replace(".","").isdigit():
            try:
                amount = float(clean)
                break
            except Exception:
                pass

    return {"date": date, "amount": amount}

# ── Evaluate ──────────────────────────────────────────────────
def evaluate(crf, test_df):
    date_tp = date_fp = date_fn = 0
    amt_tp  = amt_fp  = amt_fn  = 0

    for _, row in test_df.iterrows():
        text = str(row.get("reply",""))
        if not text:
            continue

        result = predict_ptp(crf, text)

        gt_date = str(row.get("ptp_date","")).lower()
        gt_amt  = row.get("ptp_amount","")

        # Date
        if gt_date and gt_date not in ["null","nan","none",""]:
            if result["date"]:
                date_tp += 1
            else:
                date_fn += 1
        else:
            if result["date"]:
                date_fp += 1

        # Amount
        if pd.notna(gt_amt) and str(gt_amt) not in ["null","nan","none",""]:
            if result["amount"]:
                amt_tp += 1
            else:
                amt_fn += 1
        else:
            if result["amount"]:
                amt_fp += 1

    date_rec  = date_tp / (date_tp + date_fn) if (date_tp + date_fn) > 0 else 0
    date_prec = date_tp / (date_tp + date_fp) if (date_tp + date_fp) > 0 else 0
    amt_rec   = amt_tp  / (amt_tp  + amt_fn)  if (amt_tp  + amt_fn)  > 0 else 0
    amt_prec  = amt_tp  / (amt_tp  + amt_fp)  if (amt_tp  + amt_fp)  > 0 else 0

    return {
        "date_precision": round(date_prec, 4),
        "date_recall":    round(date_rec,  4),
        "amt_precision":  round(amt_prec,  4),
        "amt_recall":     round(amt_rec,   4),
    }

if __name__ == "__main__":
    print("=== VAADA CRF PTP Extractor ===\n")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df  = pd.read_csv("data/processed/test.csv")

    ptp_train = train_df[train_df["intent"].isin(["promise_to_pay","partial_payment"])]
    ptp_test  = test_df[test_df["intent"].isin(["promise_to_pay","partial_payment"])]

    print(f"PTP train: {len(ptp_train)} | PTP test: {len(ptp_test)}")

    print("Building dataset...")
    train_sents = build_dataset(ptp_train)
    print(f"Training sentences: {len(train_sents)}")

    print("Training CRF model...")
    crf = train(train_sents)
    print("Training complete.")

    print("\nEvaluating...")
    metrics = evaluate(crf, ptp_test)

    print("\n=== RESULTS ===")
    print(f"Date  — Precision: {metrics['date_precision']:.4f} | Recall: {metrics['date_recall']:.4f}")
    print(f"Amount — Precision: {metrics['amt_precision']:.4f}  | Recall: {metrics['amt_recall']:.4f}")
    print("\n=== vs Baseline Regex ===")
    print(f"Regex  — Date recall: 0.6246 | Amount recall: 0.3754")
    print(f"CRF    — Date recall: {metrics['date_recall']:.4f} | Amount recall: {metrics['amt_recall']:.4f}")

    os.makedirs("models", exist_ok=True)
    with open("models/ptp_crf.pkl","wb") as f:
        pickle.dump(crf, f)
    print("\nSaved to models/ptp_crf.pkl")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ner_results.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("Results saved to outputs/ner_results.json")

    # Demo
    print("\n=== Demo ===")
    tests = [
        "bhai kal pakka kar dunga 🙏",
        "friday tak transfer ho jayega",
        "aadha abhi de sakta hun 6000 baaki 15 tarikh ko",
        "2-3 din aur do please",
        "next week pakka 5000 bhej dunga",
    ]
    for t in tests:
        r = predict_ptp(crf, t)
        print(f"  '{t}'")
        print(f"   → date={r['date']} amount={r['amount']}")
