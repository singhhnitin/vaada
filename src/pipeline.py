"""
pipeline.py — VAADA End-to-End Pipeline with Real Razorpay API
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
from src.nlu.recovery_predictor import engineer_features
from src.agent.razorpay_client import create_payment_link, create_partial_payment_link

# ── Load models ───────────────────────────────────────────────
def load_models():
    models = {}

    baseline_path = "models/baseline_pipeline.pkl"
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            models["intent_classifier"] = pickle.load(f)
        print("Intent classifier loaded.")
    else:
        models["intent_classifier"] = None

    recovery_path = "models/recovery_predictor.pkl"
    if os.path.exists(recovery_path):
        with open(recovery_path, "rb") as f:
            models["recovery_predictor"] = pickle.load(f)
        print("Recovery predictor loaded.")
    else:
        models["recovery_predictor"] = None

    return models

# ── Intent classification ─────────────────────────────────────
def classify_intent(classifier, reminder: str, reply: str) -> dict:
    if classifier is None:
        return {"intent": "unknown", "confidence": 0.0, "all_probs": {}}

    text    = reminder + " [SEP] " + reply
    intent  = classifier.predict([text])[0]
    probs   = classifier.predict_proba([text])[0]
    classes = classifier.classes_

    return {
        "intent":     intent,
        "confidence": float(round(max(probs), 4)),
        "all_probs":  {c: float(round(p, 4)) for c, p in zip(classes, probs)}
    }

# ── Recovery prediction ───────────────────────────────────────
def predict_recovery(predictor, intent, tone, dpd, amount, region, reply) -> dict:
    if predictor is None:
        return {"likelihood": "unknown", "confidence": 0.0}

    row = {
        "intent": intent, "tone": tone, "dpd": dpd,
        "amount": amount, "region": region, "reply": reply,
        "reminder": "", "cibil_mentioned": False, "legal_mentioned": False,
    }
    df   = pd.DataFrame([row])
    feat = engineer_features(df)

    likelihood = predictor.predict(feat)[0]
    probs      = predictor.predict_proba(feat)[0]

    return {
        "likelihood": likelihood,
        "confidence": float(round(max(probs), 4)),
        "proba":      {c: float(round(p, 4)) for c, p in zip(predictor.classes_, probs)}
    }

# ── Agent action with REAL Razorpay API ───────────────────────
def get_agent_action(intent: str, ptp: dict, dpd: int,
                     amount: float, customer_name: str = "Customer") -> dict:

    if intent == "promise_to_pay":
        ptp_amt    = ptp["ptp_amount"]["amount"] or amount
        is_partial = ptp["ptp_amount"]["is_partial"]
        days_out   = ptp["ptp_date"].get("days_from_now", 1) or 1

        if is_partial and ptp_amt < amount:
            result = create_partial_payment_link(
                total_amount   = amount,
                partial_amount = ptp_amt,
                customer_name  = customer_name,
                intent         = "partial_payment",
                dpd            = dpd
            )
            return {
                "action":        "SEND_PARTIAL_PAYMENT_LINK",
                "payment_link":  result,
                "message":       f"Partial payment link of Rs{ptp_amt:.0f} sent via Razorpay.",
                "follow_up_in":  days_out + 1,
                "risk":          "medium",
            }
        else:
            result = create_payment_link(
                amount        = ptp_amt,
                customer_name = customer_name,
                description   = "EMI Recovery - VAADA Collections",
                intent        = "promise_to_pay",
                dpd           = dpd
            )
            return {
                "action":       "SEND_FULL_PAYMENT_LINK",
                "payment_link": result,
                "message":      f"Full payment link of Rs{ptp_amt:.0f} sent via Razorpay.",
                "follow_up_in": days_out + 1,
                "risk":         "low",
            }

    elif intent == "partial_payment":
        ptp_amt = ptp["ptp_amount"]["amount"] or amount * 0.5
        result  = create_partial_payment_link(
            total_amount   = amount,
            partial_amount = ptp_amt,
            customer_name  = customer_name,
            intent         = "partial_payment",
            dpd            = dpd
        )
        return {
            "action":       "SEND_PARTIAL_PAYMENT_LINK",
            "payment_link": result,
            "message":      f"Rs{ptp_amt:.0f} partial payment link sent via Razorpay.",
            "follow_up_in": 7,
            "risk":         "medium",
        }

    elif intent == "needs_more_time":
        days_out = ptp["ptp_date"].get("days_from_now", 3) or 3
        if dpd > 60:
            return {
                "action":       "SEND_SETTLEMENT_OFFER",
                "message":      "Settlement offer sent.",
                "follow_up_in": 1,
                "risk":         "high",
            }
        return {
            "action":       "SCHEDULE_FOLLOWUP",
            "message":      f"Follow-up scheduled in {days_out} days.",
            "follow_up_in": days_out,
            "risk":         "medium",
        }

    elif intent == "dispute":
        import random
        return {
            "action":       "FLAG_FOR_HUMAN_REVIEW",
            "ticket_id":    f"VAADA-{random.randint(1000,9999)}",
            "message":      "Flagged for human review.",
            "follow_up_in": 1,
            "risk":         "medium",
        }

    elif intent == "refusal":
        if dpd > 60:
            return {
                "action":       "TRIGGER_LEGAL_NOTICE",
                "message":      "Legal notice triggered.",
                "follow_up_in": 0,
                "risk":         "critical",
            }
        return {
            "action":       "ESCALATE_TO_SENIOR_TEAM",
            "message":      "Escalated to senior collections team.",
            "follow_up_in": 1,
            "risk":         "high",
        }

    return {
        "action":       "MANUAL_REVIEW",
        "message":      "Manual review required.",
        "follow_up_in": 1,
        "risk":         "medium",
    }

# ── Full pipeline ─────────────────────────────────────────────
class VAADAPipeline:
    def __init__(self):
        print("Initializing VAADA Pipeline...")
        self.models = load_models()
        print("VAADA Pipeline ready.\n")

    def process(self, reminder: str, reply: str,
                dpd: int = 15, amount: float = 5000,
                region: str = "delhi", tone: str = "neutral",
                customer_name: str = "Customer") -> dict:

        intent_result = classify_intent(
            self.models["intent_classifier"], reminder, reply
        )
        intent = intent_result["intent"]

        ptp = extract_ptp(reminder, reply, amount)

        recovery = predict_recovery(
            self.models["recovery_predictor"],
            intent, tone, dpd, amount, region, reply
        )

        action = get_agent_action(intent, ptp, dpd, amount, customer_name)

        return {
            "timestamp":  datetime.now().isoformat(),
            "input": {
                "reminder": reminder,
                "reply":    reply,
                "dpd":      dpd,
                "amount":   amount,
                "region":   region,
            },
            "intent":   intent_result,
            "ptp":      ptp,
            "recovery": recovery,
            "action":   action,
        }

    def process_batch(self, df: pd.DataFrame) -> list:
        results = []
        for _, row in df.iterrows():
            result = self.process(
                reminder      = str(row.get("reminder", "")),
                reply         = str(row.get("reply", "")),
                dpd           = int(row.get("dpd", 15)),
                amount        = float(row.get("amount", 5000)),
                region        = str(row.get("region", "delhi")),
                tone          = str(row.get("tone", "neutral")),
                customer_name = str(row.get("name", "Customer")),
            )
            results.append(result)
        return results

# ── Demo ──────────────────────────────────────────────────────
def demo():
    pipeline = VAADAPipeline()

    cases = [
        {
            "reminder":      "Rahul ji Rs5000 EMI 8 din overdue. Aaj payment karein.",
            "reply":         "bhai kal pakka kar dunga 🙏",
            "dpd": 8, "amount": 5000, "region": "delhi",
            "customer_name": "Rahul Singh",
        },
        {
            "reminder":      "Rs12000 business loan 20 din overdue.",
            "reply":         "aadha abhi de sakta hun 6000 baaki 15 tarikh ko",
            "dpd": 20, "amount": 12000, "region": "mumbai",
            "customer_name": "Amit Kumar",
        },
        {
            "reminder":      "Rs18000 bahut time overdue. Legal action lena padega.",
            "reply":         "nahi karunga court mein milte hain",
            "dpd": 65, "amount": 18000, "region": "delhi",
            "customer_name": "Suresh Yadav",
        },
    ]

    print("=" * 60)
    print("VAADA PIPELINE DEMO — REAL RAZORPAY INTEGRATION")
    print("=" * 60)

    for i, case in enumerate(cases, 1):
        print(f"\n--- Case {i} ---")
        result = pipeline.process(**case)
        print(f"Reply    : {case['reply']}")
        print(f"Intent   : {result['intent']['intent']} ({result['intent']['confidence']:.0%})")
        print(f"Recovery : {result['recovery']['likelihood']}")
        print(f"Action   : {result['action']['action']}")

        if "payment_link" in result["action"]:
            pl = result["action"]["payment_link"]
            if pl.get("success"):
                print(f"REAL URL : {pl['short_url']}")
                print(f"Link ID  : {pl['link_id']}")
            else:
                print(f"Link err : {pl.get('error')}")

if __name__ == "__main__":
    demo()
