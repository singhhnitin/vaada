"""
multi_turn_tracker.py — VAADA Multi-Turn Conversation Tracker
Tracks promise-to-pay commitments across a collections thread over multiple days.
"""

import json
import pickle
import os
from datetime import datetime, timedelta
from src.nlu.ptp_extractor import extract_ptp

# ── Load intent classifier ────────────────────────────────────
def load_classifier():
    try:
        with open("models/baseline_pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

CLASSIFIER = load_classifier()

def classify(reminder: str, reply: str) -> dict:
    if CLASSIFIER:
        text   = reminder + " [SEP] " + reply
        intent = CLASSIFIER.predict([text])[0]
        probs  = CLASSIFIER.predict_proba([text])[0]
        return {
            "intent":     intent,
            "confidence": float(max(probs))
        }
    return {"intent": "unknown", "confidence": 0.0}

# ── Conversation state ────────────────────────────────────────
class ConversationState:
    def __init__(self, customer_name: str, loan_amount: float,
                 customer_id: str = None):
        self.customer_name  = customer_name
        self.loan_amount    = loan_amount
        self.customer_id    = customer_id or f"cust_{id(self)}"
        self.turns          = []
        self.promises       = []
        self.broken_promises= []
        self.current_status = "pending"
        self.escalation_level = 0
        self.started_at     = datetime.now()
        self.last_updated   = datetime.now()

    def add_turn(self, day: int, reminder: str,
                 reply: str, dpd: int = 0):
        intent_result = classify(reminder, reply)
        ptp           = extract_ptp(reminder, reply, self.loan_amount)

        turn = {
            "day":        day,
            "reminder":   reminder,
            "reply":      reply,
            "dpd":        dpd,
            "intent":     intent_result["intent"],
            "confidence": intent_result["confidence"],
            "ptp":        ptp,
            "timestamp":  datetime.now().isoformat(),
        }

        self.turns.append(turn)
        self.last_updated = datetime.now()
        self._update_state(turn, day)
        return turn

    def _update_state(self, turn: dict, day: int):
        intent = turn["intent"]
        ptp    = turn["ptp"]

        if intent == "promise_to_pay" and ptp["has_ptp"]:
            promise = {
                "day_promised": day,
                "ptp_date":     ptp["ptp_date"]["raw"],
                "ptp_amount":   ptp["ptp_amount"]["amount"] or self.loan_amount,
                "days_from_now":ptp["ptp_date"].get("days_from_now"),
                "strength":     ptp.get("commitment_strength", "moderate"),
                "kept":         None,
            }
            self.promises.append(promise)
            self.current_status = "promised"

        elif intent == "partial_payment":
            self.current_status = "partial_negotiation"

        elif intent == "dispute":
            self.current_status = "disputed"
            self.escalation_level = max(self.escalation_level, 1)

        elif intent == "refusal":
            self.current_status = "refused"
            self.escalation_level = max(self.escalation_level, 2)

        elif intent == "needs_more_time":
            if self.current_status == "promised":
                # Promise was made but now asking for more time = broken promise
                if self.promises:
                    self.promises[-1]["kept"] = False
                    self.broken_promises.append(self.promises[-1])
                self.current_status = "promise_broken"
                self.escalation_level = max(self.escalation_level, 1)

        # Check if promise was due and not fulfilled
        self._check_broken_promises(day)

    def _check_broken_promises(self, current_day: int):
        for promise in self.promises:
            if promise["kept"] is None:
                days_from_promise = current_day - promise["day_promised"]
                days_promised     = promise.get("days_from_now") or 1
                if days_from_promise > days_promised + 1:
                    promise["kept"] = False
                    self.broken_promises.append(promise)
                    self.current_status  = "promise_broken"
                    self.escalation_level = max(self.escalation_level, 1)

    def mark_paid(self, amount: float = None):
        self.current_status = "paid"
        if self.promises:
            self.promises[-1]["kept"] = True
        return {"status": "paid", "amount": amount or self.loan_amount}

    def get_next_action(self) -> dict:
        from src.agent.razorpay_client import create_payment_link

        status = self.current_status
        level  = self.escalation_level
        n_broken = len(self.broken_promises)

        if status == "promised":
            days_out = 1
            if self.promises:
                days_out = self.promises[-1].get("days_from_now") or 1
            result = create_payment_link(
                amount        = self.loan_amount,
                customer_name = self.customer_name,
                description   = "EMI Recovery - VAADA",
                intent        = "promise_to_pay",
                dpd           = self.turns[-1]["dpd"] if self.turns else 0
            )
            return {
                "action":       "SEND_PAYMENT_LINK",
                "link":         result.get("short_url") if result.get("success") else None,
                "follow_up_in": days_out + 1,
                "message":      f"Payment link sent. Follow up in {days_out+1} days.",
                "risk":         "low",
            }

        elif status == "promise_broken":
            if n_broken >= 2:
                return {
                    "action":       "ESCALATE_TO_SENIOR",
                    "follow_up_in": 1,
                    "message":      f"{n_broken} broken promises. Escalating.",
                    "risk":         "high",
                }
            return {
                "action":       "SEND_FIRM_REMINDER",
                "follow_up_in": 1,
                "message":      "Promise broken. Send firm reminder with CIBIL warning.",
                "risk":         "medium",
            }

        elif status == "disputed":
            return {
                "action":       "FLAG_FOR_HUMAN",
                "follow_up_in": 1,
                "message":      "Dispute flagged for human review.",
                "risk":         "medium",
            }

        elif status == "refused":
            return {
                "action":       "TRIGGER_LEGAL" if level >= 2 else "ESCALATE",
                "follow_up_in": 0,
                "message":      "Refusal. Legal action triggered." if level >= 2 else "Escalating.",
                "risk":         "critical" if level >= 2 else "high",
            }

        elif status == "partial_negotiation":
            from src.agent.razorpay_client import create_partial_payment_link
            partial_amt = self.loan_amount * 0.5
            result = create_partial_payment_link(
                total_amount   = self.loan_amount,
                partial_amount = partial_amt,
                customer_name  = self.customer_name,
                intent         = "partial_payment",
                dpd            = self.turns[-1]["dpd"] if self.turns else 0
            )
            return {
                "action":       "SEND_PARTIAL_LINK",
                "link":         result.get("short_url") if result.get("success") else None,
                "follow_up_in": 7,
                "message":      f"Partial link of Rs{partial_amt:.0f} sent.",
                "risk":         "medium",
            }

        return {
            "action":       "SCHEDULE_FOLLOWUP",
            "follow_up_in": 3,
            "message":      "Follow up scheduled.",
            "risk":         "medium",
        }

    def summary(self) -> dict:
        return {
            "customer":        self.customer_name,
            "loan_amount":     self.loan_amount,
            "total_turns":     len(self.turns),
            "status":          self.current_status,
            "escalation_level":self.escalation_level,
            "promises_made":   len(self.promises),
            "promises_broken": len(self.broken_promises),
            "intent_sequence": [t["intent"] for t in self.turns],
            "next_action":     self.get_next_action(),
        }


# ── Demo: 5-day collections thread ───────────────────────────
def run_demo():
    print("=" * 60)
    print("VAADA Multi-Turn Conversation Tracker")
    print("Simulating a 5-day collections thread")
    print("=" * 60)

    conv = ConversationState(
        customer_name = "Rahul Singh",
        loan_amount   = 8000,
        customer_id   = "CUST_001"
    )

    thread = [
        {
            "day": 1, "dpd": 8,
            "reminder": "Rahul ji Rs8000 EMI 8 din overdue. Aaj payment karein.",
            "reply":    "bhai kal pakka kar dunga, aaj thoda busy tha 🙏",
        },
        {
            "day": 3, "dpd": 10,
            "reminder": "Rahul ji, kal payment nahi aaya. Please aaj karein.",
            "reply":    "yaar 2-3 din aur chahiye, paisa aa raha hai",
        },
        {
            "day": 6, "dpd": 13,
            "reminder": "3rd reminder - Rs8000 still pending. CIBIL affect hoga.",
            "reply":    "bhai client ne payment nahi ki mujhe, Friday pakka",
        },
        {
            "day": 9, "dpd": 16,
            "reminder": "Friday bhi nahi kiya. This is the last reminder.",
            "reply":    "aadha abhi de sakta hun 4000 baaki next week",
        },
        {
            "day": 12, "dpd": 19,
            "reminder": "Rs4000 partial payment accepted. When for remaining?",
            "reply":    "bhai honestly abhi possible nahi court mein milte hain",
        },
    ]

    for turn_data in thread:
        print(f"\n{'─'*50}")
        print(f"DAY {turn_data['day']} | DPD: {turn_data['dpd']}")
        print(f"REMINDER : {turn_data['reminder'][:60]}")
        print(f"REPLY    : {turn_data['reply']}")

        turn = conv.add_turn(
            day      = turn_data["day"],
            reminder = turn_data["reminder"],
            reply    = turn_data["reply"],
            dpd      = turn_data["dpd"],
        )

        print(f"INTENT   : {turn['intent']} ({turn['confidence']:.0%})")
        print(f"STATUS   : {conv.current_status}")
        print(f"ESCALATION: Level {conv.escalation_level}")

        action = conv.get_next_action()
        print(f"ACTION   : {action['action']}")
        if "link" in action and action["link"]:
            print(f"LINK     : {action['link']}")

    print(f"\n{'='*60}")
    print("CONVERSATION SUMMARY")
    print(f"{'='*60}")
    summary = conv.summary()
    print(f"Customer        : {summary['customer']}")
    print(f"Loan Amount     : Rs{summary['loan_amount']}")
    print(f"Total Turns     : {summary['total_turns']}")
    print(f"Final Status    : {summary['status']}")
    print(f"Escalation Level: {summary['escalation_level']}")
    print(f"Promises Made   : {summary['promises_made']}")
    print(f"Promises Broken : {summary['promises_broken']}")
    print(f"Intent Sequence : {' → '.join(summary['intent_sequence'])}")
    print(f"\nFINAL ACTION    : {summary['next_action']['action']}")
    print(f"RISK LEVEL      : {summary['next_action']['risk']}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/multi_turn_demo.json", "w") as f:
        json.dump({
            "summary": summary,
            "turns": conv.turns
        }, f, indent=2, ensure_ascii=False)
    print("\nSaved to outputs/multi_turn_demo.json")

if __name__ == "__main__":
    run_demo()
