"""
pipeline.py — VAADA End-to-End Pipeline
Input: raw Hinglish WhatsApp conversation
Output: intent, PTP extraction, recovery likelihood, agent action
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlu.ptp_extractor import extract_ptp
from src.nlu.recovery_predictor import engineer_features, create_labels

# ── Load models ───────────────────────────────────────────────
def load_models():
    models = {}

    # Baseline intent classifier
    baseline_path = "models/baseline_pipeline.pkl"
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            models["intent_classifier"] = pickle.load(f)
        print("Intent classifier loaded.")
    else:
        print("WARNING: baseline_pipeline.pkl not found.")
        models["intent_classifier"] = None

    # Recovery predictor
    recovery_path = "models/recovery_predictor.pkl"
    if os.path.exists(recovery_path):
        with open(recovery_path, "rb") as f:
            models["recovery_predictor"] = pickle.load(f)
        print("Recovery predictor loaded.")
    else:
        print("WARNING: recovery_predictor.pkl not found.")
        models["recovery_predictor"] = None

    return models

# ── Intent classification ─────────────────────────────────────
def classify_intent(classifier, reminder: str, reply: str) -> dict:
    if classifier is None:
        return {"intent": "unknown", "confidence": 0.0, "all_probs": {}}

    text = reminder + " [SEP] " + reply
    intent = classifier.predict([text])[0]
    probs  = classifier.predict_proba([text])[0]
    classes = classifier.classes_

    return {
        "intent":     intent,
        "confidence": float(round(max(probs), 4)),
        "all_probs":  {
            c: float(round(p, 4))
            for c, p in zip(classes, probs)
        }
    }

# ── Recovery prediction ───────────────────────────────────────
def predict_recovery(predictor, intent: str, tone: str,
                     dpd: int, amount: float,
                     region: str, reply: str) -> dict:
    if predictor is None:
        return {"likelihood": "unknown", "confidence": 0.0}

    row = {
        "intent":  intent,
        "tone":    tone,
        "dpd":     dpd,
        "amount":  amount,
        "region":  region,
        "reply":   reply,
        "reminder": "",
        "cibil_mentioned": False,
        "legal_mentioned": False,
    }
    df   = pd.DataFrame([row])
    feat = engineer_features(df)

    likelihood = predictor.predict(feat)[0]
    probs      = predictor.predict_proba(feat)[0]
    confidence = float(round(max(probs), 4))

    return {
        "likelihood": likelihood,
        "confidence": confidence,
        "proba": {
            c: float(round(p, 4))
            for c, p in zip(predictor.classes_, probs)
        }
    }

# ── Agent action ──────────────────────────────────────────────
def get_agent_action(intent: str, ptp: dict,
                     dpd: int, amount: float,
                     recovery: str) -> dict:
    import random

    def make_link(amt, partial=False):
        lid = f"rzp_{'p' if partial else 'f'}_{random.randint(10000,99999)}"
        return {"link_id": lid, "url": f"https://rzp.io/l/{lid}", "amount": amt}

    if intent == "promise_to_pay":
        ptp_amt    = ptp["ptp_amount"]["amount"] or amount
        is_partial = ptp["ptp_amount"]["is_partial"]
        days_out   = ptp["ptp_date"].get("days_from_now", 1) or 1

        if is_partial:
            link = make_link(ptp_amt, partial=True)
            return {
                "action":       "SEND_PARTIAL_PAYMENT_LINK",
                "payment_link": link,
                "message":      f"Partial payment link of ₹{ptp_amt:.0f} bhej rahe hain.",
                "follow_up_in": days_out + 1,
                "risk":         "medium",
            }
        else:
            link = make_link(ptp_amt)
            return {
                "action":       "SEND_FULL_PAYMENT_LINK",
                "payment_link": link,
                "message":      f"Payment link of ₹{ptp_amt:.0f} bhej rahe hain.",
                "follow_up_in": days_out + 1,
                "risk":         "low",
            }

    elif intent == "partial_payment":
        ptp_amt = ptp["ptp_amount"]["amount"] or amount * 0.5
        link    = make_link(ptp_amt, partial=True)
        return {
            "action":       "SEND_PARTIAL_PAYMENT_LINK",
            "payment_link": link,
            "message":      f"₹{ptp_amt:.0f} ka partial payment accept kar rahe hain.",
            "follow_up_in": 7,
            "risk":         "medium",
        }

    elif intent == "needs_more_time":
        days_out = ptp["ptp_date"].get("days_from_now", 3) or 3
        if dpd > 60:
            return {
                "action":       "SEND_SETTLEMENT_OFFER",
                "message":      "Settlement offer bhej rahe hain.",
                "follow_up_in": 1,
                "risk":         "high",
            }
        return {
            "action":       "SCHEDULE_FOLLOWUP",
            "message":      f"{days_out} din ka time de rahe hain.",
            "follow_up_in": days_out,
            "risk":         "medium",
        }

    elif intent == "dispute":
        ticket_id = f"VAADA-{random.randint(1000,9999)}"
        return {
            "action":       "FLAG_FOR_HUMAN_REVIEW",
            "ticket_id":    ticket_id,
            "message":      "Complaint note kar li. Agent 24 ghante mein contact karega.",
            "follow_up_in": 1,
            "risk":         "medium",
        }

    elif intent == "refusal":
        if dpd > 60:
            return {
                "action":       "TRIGGER_LEGAL_NOTICE",
                "message":      "Legal notice process initiate kar rahe hain.",
                "follow_up_in": 0,
                "risk":         "critical",
            }
        return {
            "action":       "ESCALATE_TO_SENIOR_TEAM",
            "message":      "Senior collections team ko escalate kar diya.",
            "follow_up_in": 1,
            "risk":         "high",
        }

    return {
        "action":       "MANUAL_REVIEW",
        "message":      "Manual review required.",
        "follow_up_in": 1,
        "risk":         "medium",
    }

# ── Main pipeline ─────────────────────────────────────────────
class VAADAPipeline:
    def __init__(self):
        print("Initializing VAADA Pipeline...")
        self.models = load_models()
        print("VAADA Pipeline ready.\n")

    def process(self, reminder: str, reply: str,
                dpd: int = 15, amount: float = 5000,
                region: str = "delhi",
                tone: str = "neutral") -> dict:

        # Step 1: Intent classification
        intent_result = classify_intent(
            self.models["intent_classifier"],
            reminder, reply
        )
        intent = intent_result["intent"]

        # Step 2: PTP extraction
        ptp = extract_ptp(reminder, reply, amount)

        # Step 3: Recovery prediction
        recovery = predict_recovery(
            self.models["recovery_predictor"],
            intent, tone, dpd, amount, region, reply
        )

        # Step 4: Agent action
        action = get_agent_action(
            intent, ptp, dpd, amount,
            recovery["likelihood"]
        )

        return {
            "timestamp":   datetime.now().isoformat(),
            "input": {
                "reminder": reminder,
                "reply":    reply,
                "dpd":      dpd,
                "amount":   amount,
                "region":   region,
            },
            "intent":    intent_result,
            "ptp":       ptp,
            "recovery":  recovery,
            "action":    action,
        }

    def process_batch(self, df: pd.DataFrame) -> list:
        results = []
        for _, row in df.iterrows():
            result = self.process(
                reminder = str(row.get("reminder", "")),
                reply    = str(row.get("reply", "")),
                dpd      = int(row.get("dpd", 15)),
                amount   = float(row.get("amount", 5000)),
                region   = str(row.get("region", "delhi")),
                tone     = str(row.get("tone", "neutral")),
            )
            results.append(result)
        return results

# ── Evaluation ────────────────────────────────────────────────
def evaluate():
    pipeline = VAADAPipeline()
    test_df  = pd.read_csv("data/processed/test.csv")

    print(f"Running VAADA on {len(test_df)} test samples...\n")
    results = pipeline.process_batch(test_df)

    # Metrics
    correct       = 0
    total         = len(results)
    ptp_detected  = 0
    links_sent    = 0
    risk_dist     = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    action_dist   = {}

    for i, (result, (_, row)) in enumerate(
        zip(results, test_df.iterrows())
    ):
        # Intent accuracy
        if result["intent"]["intent"] == row["intent"]:
            correct += 1

        # PTP detection
        if result["ptp"]["has_ptp"]:
            ptp_detected += 1

        # Payment links
        if "payment_link" in result["action"]:
            links_sent += 1

        # Risk distribution
        risk = result["action"].get("risk", "medium")
        if risk in risk_dist:
            risk_dist[risk] += 1

        # Action distribution
        action = result["action"]["action"]
        action_dist[action] = action_dist.get(action, 0) + 1

    intent_acc = correct / total
    ptp_rate   = ptp_detected / total
    link_rate  = links_sent / total

    print("=" * 50)
    print("VAADA END-TO-END EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples processed : {total}")
    print(f"Intent accuracy         : {intent_acc:.4f}")
    print(f"PTP detection rate      : {ptp_rate:.4f}")
    print(f"Payment links generated : {links_sent} ({link_rate:.1%})")
    print(f"\nAction distribution:")
    for action, count in sorted(
        action_dist.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {action:35} : {count:4} ({count/total:.1%})")
    print(f"\nRisk distribution:")
    for risk, count in risk_dist.items():
        print(f"  {risk:10} : {count:4} ({count/total:.1%})")

    # Save
    os.makedirs("outputs", exist_ok=True)
    summary = {
        "total":          total,
        "intent_accuracy": round(intent_acc, 4),
        "ptp_rate":        round(ptp_rate, 4),
        "link_rate":       round(link_rate, 4),
        "links_sent":      links_sent,
        "action_dist":     action_dist,
        "risk_dist":       risk_dist,
        "sample_outputs":  results[:5],
    }
    with open("outputs/pipeline_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved to outputs/pipeline_results.json")
    return summary

# ── Demo ──────────────────────────────────────────────────────
def demo():
    pipeline = VAADAPipeline()

    cases = [
        {
            "reminder": "Rahul ji ₹5000 EMI 8 din overdue. "
                       "Please aaj payment karein: rzp.io/pay",
            "reply":    "bhai kal pakka kar dunga 🙏",
            "dpd": 8, "amount": 5000, "region": "delhi",
        },
        {
            "reminder": "Arre ₹12000 business loan 20 din overdue. "
                       "Settlement offer hai aapke liye.",
            "reply":    "aadha abhi de sakta hun 6000 baaki 15 tarikh ko",
            "dpd": 20, "amount": 12000, "region": "mumbai",
        },
        {
            "reminder": "₹8000 EMI miss ho gayi. Kripya payment karein.",
            "reply":    "maine toh 3 din pehle UPI kar diya tha",
            "dpd": 7, "amount": 8000, "region": "hyderabad",
        },
        {
            "reminder": "₹18000 bahut time overdue. Legal action lena padega.",
            "reply":    "nahi karunga court mein milte hain",
            "dpd": 65, "amount": 18000, "region": "bangalore",
        },
    ]

    print("=" * 60)
    print("VAADA PIPELINE DEMO")
    print("=" * 60)

    for i, case in enumerate(cases, 1):
        print(f"\n--- Case {i} ---")
        print(f"Reminder : {case['reminder'][:60]}...")
        print(f"Reply    : {case['reply']}")
        result = pipeline.process(**case)
        print(f"Intent   : {result['intent']['intent']} "
              f"({result['intent']['confidence']:.0%})")
        print(f"PTP      : {result['ptp']['has_ptp']} | "
              f"Date: {result['ptp']['ptp_date']['raw']} | "
              f"Amount: {result['ptp']['ptp_amount']['amount']}")
        print(f"Recovery : {result['recovery']['likelihood']} "
              f"({result['recovery']['confidence']:.0%})")
        print(f"Action   : {result['action']['action']}")
        print(f"Message  : {result['action']['message']}")
        if "payment_link" in result["action"]:
            print(f"Link     : {result['action']['payment_link']['url']}")

if __name__ == "__main__":
    print("VAADA — Vernacular Agentic AI for Debt & Arrears\n")
    demo()
    print("\n" + "=" * 60)
    print("Running full evaluation...")
    evaluate()
