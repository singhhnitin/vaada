"""
proactive_risk.py — VAADA Proactive Default Risk Detection
Predicts default probability BEFORE it happens using early signals.
This addresses the 'Detect' requirement of Revenue Recovery track.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from datetime import datetime

# ── Risk signals ──────────────────────────────────────────────
EARLY_WARNING_SIGNALS = {
    "response_delay":     {"weight": 0.20, "desc": "Customer takes >2 days to reply"},
    "evasive_language":   {"weight": 0.25, "desc": "Vague timelines, non-committal words"},
    "broken_promise":     {"weight": 0.30, "desc": "Previously promised but didn't pay"},
    "dpd_trend":          {"weight": 0.15, "desc": "DPD increasing across reminders"},
    "tone_deterioration": {"weight": 0.10, "desc": "Tone getting more aggressive/resigned"},
}

EVASIVE_WORDS = [
    "dekhta hun", "try karunga", "shayad", "maybe",
    "koshish", "hopefully", "pata nahi", "difficult",
    "abhi nahi", "baad mein", "kal tak", "soon"
]

STRONG_COMMIT_WORDS = [
    "pakka", "definitely", "zaroor", "promise",
    "confirm", "100%", "abhi", "turant"
]

def compute_evasion_score(text: str) -> float:
    text_lower = text.lower()
    evasive  = sum(1 for w in EVASIVE_WORDS if w in text_lower)
    committed = sum(1 for w in STRONG_COMMIT_WORDS if w in text_lower)
    if evasive + committed == 0:
        return 0.5
    return evasive / (evasive + committed)

def compute_tone_risk(tone: str) -> float:
    tone_risk = {
        "cooperative": 0.1,
        "polite":      0.2,
        "neutral":     0.3,
        "desperate":   0.5,
        "evasive":     0.6,
        "worried":     0.5,
        "angry":       0.7,
        "aggressive":  0.8,
        "resigned":    0.9,
    }
    return tone_risk.get(tone.lower(), 0.4)

class ProactiveRiskScorer:
    """
    Scores default risk from early conversation signals.
    Called on Day 1-3 conversations to predict who will default.
    """

    def score(self, conversation_history: list,
              dpd_history: list = None,
              loan_amount: float = 5000) -> dict:
        """
        conversation_history: list of dicts with keys:
          day, reminder, reply, intent, tone
        dpd_history: list of DPD values across days
        """

        if not conversation_history:
            return {"risk_score": 0.5, "risk_level": "medium",
                    "signals": [], "recommendation": "Insufficient data"}

        signals_triggered = []
        score = 0.0

        # Signal 1: Response delay
        if len(conversation_history) > 1:
            days = [t["day"] for t in conversation_history]
            avg_gap = np.diff(days).mean() if len(days) > 1 else 1
            if avg_gap > 2:
                score += EARLY_WARNING_SIGNALS["response_delay"]["weight"]
                signals_triggered.append({
                    "signal":  "response_delay",
                    "value":   f"{avg_gap:.1f} days avg response",
                    "weight":  EARLY_WARNING_SIGNALS["response_delay"]["weight"]
                })

        # Signal 2: Evasive language
        last_reply = conversation_history[-1].get("reply", "")
        evasion    = compute_evasion_score(last_reply)
        if evasion > 0.4:
            contribution = EARLY_WARNING_SIGNALS["evasive_language"]["weight"] * evasion
            score += contribution
            signals_triggered.append({
                "signal":  "evasive_language",
                "value":   f"Evasion score: {evasion:.2f}",
                "weight":  contribution
            })

        # Signal 3: Broken promise detection
        promises = [t for t in conversation_history if t.get("intent") == "promise_to_pay"]
        if len(promises) > 1:
            score += EARLY_WARNING_SIGNALS["broken_promise"]["weight"]
            signals_triggered.append({
                "signal":  "broken_promise",
                "value":   f"{len(promises)} promises made — pattern detected",
                "weight":  EARLY_WARNING_SIGNALS["broken_promise"]["weight"]
            })

        # Signal 4: DPD trend
        if dpd_history and len(dpd_history) > 1:
            dpd_increase = dpd_history[-1] - dpd_history[0]
            if dpd_increase > 10:
                contribution = EARLY_WARNING_SIGNALS["dpd_trend"]["weight"]
                score += contribution
                signals_triggered.append({
                    "signal":  "dpd_trend",
                    "value":   f"DPD increased by {dpd_increase} days",
                    "weight":  contribution
                })

        # Signal 5: Tone deterioration
        tones = [t.get("tone", "neutral") for t in conversation_history]
        if len(tones) > 1:
            first_risk = compute_tone_risk(tones[0])
            last_risk  = compute_tone_risk(tones[-1])
            if last_risk > first_risk + 0.2:
                contribution = EARLY_WARNING_SIGNALS["tone_deterioration"]["weight"]
                score += contribution
                signals_triggered.append({
                    "signal":  "tone_deterioration",
                    "value":   f"Tone: {tones[0]} → {tones[-1]}",
                    "weight":  contribution
                })

        score = float(np.clip(score, 0, 1))

        # Risk level
        if score >= 0.25:
            risk_level     = "HIGH"
            recommendation = "Immediate escalation. Send senior agent. Consider settlement."
            action         = "ESCALATE_NOW"
        elif score >= 0.15:
            risk_level     = "MEDIUM"
            recommendation = "Send firm reminder with CIBIL warning. Monitor daily."
            action         = "MONITOR_CLOSELY"
        else:
            risk_level     = "LOW"
            recommendation = "Continue standard follow-up. Auto-send payment link."
            action         = "STANDARD_FOLLOWUP"

        return {
            "customer_risk_score":    round(score, 3),
            "risk_level":             risk_level,
            "default_probability":    f"{score*100:.1f}%",
            "signals_triggered":      signals_triggered,
            "recommendation":         recommendation,
            "suggested_action":       action,
            "loan_amount":            loan_amount,
            "amount_at_risk":         round(loan_amount * score, 2),
            "evaluated_at":           datetime.now().isoformat(),
        }

    def batch_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score an entire batch of customers."""
        results = []
        for _, row in df.iterrows():
            conv = [{
                "day":    int(row.get("dpd", 15)),
                "reply":  str(row.get("reply", "")),
                "intent": str(row.get("intent", "")),
                "tone":   str(row.get("tone", "neutral")),
            }]
            score = self.score(
                conversation_history = conv,
                dpd_history          = [int(row.get("dpd", 15))],
                loan_amount          = float(row.get("amount", 5000))
            )
            results.append({
                "intent":          row.get("intent"),
                "tone":            row.get("tone"),
                "dpd":             row.get("dpd"),
                "amount":          row.get("amount"),
                "risk_score":      score["customer_risk_score"],
                "risk_level":      score["risk_level"],
                "amount_at_risk":  score["amount_at_risk"],
                "action":          score["suggested_action"],
            })
        return pd.DataFrame(results)

def run_demo():
    scorer = ProactiveRiskScorer()

    print("=" * 60)
    print("VAADA Proactive Risk Detection")
    print("Predicting default BEFORE it happens")
    print("=" * 60)

    scenarios = [
        {
            "name": "Rahul Singh — Low Risk",
            "history": [
                {"day":1,"reply":"bhai kal pakka kar dunga 🙏","intent":"promise_to_pay","tone":"polite"},
            ],
            "dpd_history": [8],
            "amount": 5000
        },
        {
            "name": "Amit Kumar — Medium Risk",
            "history": [
                {"day":1,"reply":"kal kar dunga","intent":"promise_to_pay","tone":"polite"},
                {"day":3,"reply":"2-3 din aur do yaar","intent":"needs_more_time","tone":"evasive"},
            ],
            "dpd_history": [8, 11],
            "amount": 12000
        },
        {
            "name": "Suresh Yadav — HIGH RISK",
            "history": [
                {"day":1,"reply":"kal kar dunga","intent":"promise_to_pay","tone":"polite"},
                {"day":4,"reply":"dekhta hun koshish karunga","intent":"needs_more_time","tone":"evasive"},
                {"day":8,"reply":"abhi mushkil hai","intent":"needs_more_time","tone":"resigned"},
            ],
            "dpd_history": [10, 13, 18],
            "amount": 18000
        },
    ]

    total_at_risk = 0
    for sc in scenarios:
        result = scorer.score(sc["history"], sc["dpd_history"], sc["amount"])
        print(f"\n--- {sc['name']} ---")
        print(f"Risk Score   : {result['customer_risk_score']}")
        print(f"Risk Level   : {result['risk_level']}")
        print(f"Default Prob : {result['default_probability']}")
        print(f"Amount at Risk: Rs{result['amount_at_risk']:.0f}")
        print(f"Action       : {result['suggested_action']}")
        for sig in result["signals_triggered"]:
            print(f"  WARNING: {sig['signal']} — {sig['value']}")
        total_at_risk += result["amount_at_risk"]

    print(f"\nTotal amount at risk: Rs{total_at_risk:.0f}")

    # Batch evaluation
    print("\n--- Batch Evaluation on Test Set ---")
    try:
        df = pd.read_csv("data/processed/test.csv")
        results_df = scorer.batch_score(df.head(100))

        high_risk = results_df[results_df["risk_level"] == "HIGH"]
        print(f"Processed: 100 customers")
        print(f"High risk: {len(high_risk)} ({len(high_risk)}%)")
        print(f"Total amount at risk: Rs{results_df['amount_at_risk'].sum():.0f}")
        print(f"Avg risk score: {results_df['risk_score'].mean():.3f}")

        os.makedirs("outputs", exist_ok=True)
        results_df.to_csv("outputs/proactive_risk_scores.csv", index=False)
        print("Saved to outputs/proactive_risk_scores.csv")
    except Exception as e:
        print(f"Batch eval error: {e}")

    print("\nDone.")

if __name__ == "__main__":
    run_demo()
