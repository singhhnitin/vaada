"""
workflow.py — VAADA Agent Workflow
Takes intent + PTP extraction → triggers right recovery action.
Simulates Razorpay payment link generation and collections actions.
"""

import json
import random
import pandas as pd
from datetime import datetime, timedelta
from src.nlu.ptp_extractor import extract_ptp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Action types ──────────────────────────────────────────────
ACTIONS = {
    "SEND_PAYMENT_LINK":    "Send UPI payment link to customer",
    "SEND_PARTIAL_LINK":    "Send partial payment link",
    "SCHEDULE_FOLLOWUP":    "Schedule follow-up message",
    "FLAG_DISPUTE":         "Flag for human review — dispute",
    "ESCALATE":             "Escalate to collections team",
    "SEND_SETTLEMENT":      "Send settlement offer",
    "MARK_HIGH_RISK":       "Mark account as high risk",
    "SEND_LEGAL_NOTICE":    "Trigger legal notice workflow",
}

# ── Mock Razorpay API ─────────────────────────────────────────
def generate_payment_link(amount: float, customer_name: str,
                           partial: bool = False) -> dict:
    link_id = f"rzp_{'partial' if partial else 'full'}_{random.randint(10000,99999)}"
    return {
        "link_id":    link_id,
        "url":        f"https://rzp.io/l/{link_id}",
        "amount":     amount,
        "partial":    partial,
        "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
        "status":     "created",
    }

def schedule_followup(days: int, reason: str) -> dict:
    followup_date = datetime.now() + timedelta(days=days)
    return {
        "scheduled_at": followup_date.isoformat(),
        "reason":       reason,
        "channel":      "whatsapp",
    }

def flag_for_human(reason: str, priority: str = "medium") -> dict:
    return {
        "ticket_id": f"VAADA-{random.randint(1000,9999)}",
        "reason":    reason,
        "priority":  priority,
        "assigned":  "collections_team",
    }

# ── Core decision engine ──────────────────────────────────────
def decide_action(intent: str, ptp: dict, dpd: int,
                  loan_amount: float, tone: str = "neutral") -> dict:
    action_taken = []
    messages     = []
    next_steps   = []

    if intent == "promise_to_pay":
        ptp_amount = ptp['ptp_amount']['amount'] or loan_amount
        is_partial = ptp['ptp_amount']['is_partial']
        days_out   = ptp['ptp_date'].get('days_from_now', 1) or 1

        if is_partial and ptp_amount < loan_amount:
            link = generate_payment_link(ptp_amount, "customer", partial=True)
            action_taken.append(ACTIONS["SEND_PARTIAL_LINK"])
            messages.append(
                f"Partial payment link of ₹{ptp_amount:.0f} bhej rahe hain. "
                f"Baaki ₹{loan_amount - ptp_amount:.0f} baad mein."
            )
            followup = schedule_followup(days_out + 1, "Check partial payment")
            next_steps.append(f"Follow up on {followup['scheduled_at'][:10]}")
        else:
            link = generate_payment_link(ptp_amount, "customer")
            action_taken.append(ACTIONS["SEND_PAYMENT_LINK"])
            messages.append(
                f"Payment link of ₹{ptp_amount:.0f} bhej rahe hain. "
                f"Please {days_out} din mein complete karein."
            )
            followup = schedule_followup(days_out + 1, "Verify payment")
            next_steps.append(f"Verify payment on {followup['scheduled_at'][:10]}")

        strength = ptp.get('commitment_strength', 'moderate')
        if strength == "weak":
            action_taken.append(ACTIONS["SCHEDULE_FOLLOWUP"])
            next_steps.append("Send reminder if not paid in 24 hours")

        return {
            "action":       action_taken,
            "payment_link": link,
            "message":      messages,
            "next_steps":   next_steps,
            "risk_level":   "low" if strength == "strong" else "medium",
        }

    elif intent == "needs_more_time":
        days_out = ptp['ptp_date'].get('days_from_now') or 3

        if dpd > 60:
            action_taken.append(ACTIONS["SEND_SETTLEMENT"])
            messages.append(
                "Aapke liye special settlement offer hai. "
                "Ek baar baat karte hain?"
            )
            next_steps.append("Send settlement offer within 24 hours")
            risk = "high"
        elif dpd > 30:
            action_taken.append(ACTIONS["SCHEDULE_FOLLOWUP"])
            link = generate_payment_link(loan_amount * 0.5, "customer", partial=True)
            action_taken.append(ACTIONS["SEND_PARTIAL_LINK"])
            messages.append(
                f"Aadha payment ₹{loan_amount*0.5:.0f} abhi kar sakte ho? "
                f"Baaki {days_out} din mein."
            )
            next_steps.append(f"Follow up in {days_out} days")
            risk = "medium"
        else:
            followup = schedule_followup(days_out, "Payment follow-up")
            action_taken.append(ACTIONS["SCHEDULE_FOLLOWUP"])
            messages.append(
                f"Theek hai, {days_out} din ka time de rahe hain. "
                f"{followup['scheduled_at'][:10]} tak payment karein."
            )
            next_steps.append(f"Follow up on {followup['scheduled_at'][:10]}")
            risk = "medium"

        return {
            "action":     action_taken,
            "message":    messages,
            "next_steps": next_steps,
            "risk_level": risk,
        }

    elif intent == "partial_payment":
        partial_amount = ptp['ptp_amount']['amount'] or loan_amount * 0.5
        link = generate_payment_link(partial_amount, "customer", partial=True)
        action_taken.append(ACTIONS["SEND_PARTIAL_LINK"])
        messages.append(
            f"Partial payment of ₹{partial_amount:.0f} accept kar rahe hain. "
            f"Link bhej rahe hain."
        )
        next_steps.append("Follow up for remaining amount in 7 days")

        return {
            "action":       action_taken,
            "payment_link": link,
            "message":      messages,
            "next_steps":   next_steps,
            "risk_level":   "medium",
        }

    elif intent == "dispute":
        ticket = flag_for_human(
            "Customer disputes payment amount or prior payment",
            priority="high" if dpd > 30 else "medium"
        )
        action_taken.append(ACTIONS["FLAG_DISPUTE"])
        messages.append(
            "Aapki complaint note kar li gayi hai. "
            "Ek agent 24 ghante mein contact karega."
        )
        next_steps.append(f"Ticket {ticket['ticket_id']} assigned to team")

        return {
            "action":     action_taken,
            "ticket":     ticket,
            "message":    messages,
            "next_steps": next_steps,
            "risk_level": "medium",
        }

    elif intent == "refusal":
        action_taken.append(ACTIONS["MARK_HIGH_RISK"])

        if dpd > 60:
            action_taken.append(ACTIONS["SEND_LEGAL_NOTICE"])
            messages.append(
                "Legal notice process initiate kar rahe hain."
            )
            next_steps.append("Legal team notification sent")
            risk = "critical"
        else:
            action_taken.append(ACTIONS["ESCALATE"])
            messages.append(
                "Account escalate kar diya gaya hai senior team ko."
            )
            next_steps.append("Senior collections team will contact in 24 hours")
            risk = "high"

        return {
            "action":     action_taken,
            "message":    messages,
            "next_steps": next_steps,
            "risk_level": risk,
        }

    return {
        "action":     ["UNKNOWN"],
        "message":    ["Unable to determine action"],
        "next_steps": [],
        "risk_level": "medium",
    }

# ── Full pipeline ─────────────────────────────────────────────
def run_pipeline(reminder: str, reply: str,
                 intent: str, dpd: int,
                 loan_amount: float,
                 tone: str = "neutral") -> dict:
    ptp    = extract_ptp(reminder, reply, loan_amount)
    result = decide_action(intent, ptp, dpd, loan_amount, tone)

    return {
        "input": {
            "reminder":    reminder,
            "reply":       reply,
            "intent":      intent,
            "dpd":         dpd,
            "loan_amount": loan_amount,
        },
        "ptp_extraction": ptp,
        "agent_action":   result,
        "timestamp":      datetime.now().isoformat(),
    }

# ── Evaluate on test set ──────────────────────────────────────
def evaluate_pipeline():
    test_df = pd.read_csv("data/processed/test.csv")

    results    = []
    action_map = {}
    risk_dist  = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    print(f"Running agent pipeline on {len(test_df)} test samples...")

    for _, row in test_df.iterrows():
        output = run_pipeline(
            reminder    = str(row['reminder']),
            reply       = str(row['reply']),
            intent      = str(row['intent']),
            dpd         = int(row.get('dpd', 15)),
            loan_amount = float(row.get('amount', 5000)),
            tone        = str(row.get('tone', 'neutral')),
        )

        actions = output['agent_action']['action']
        for a in actions:
            action_map[a] = action_map.get(a, 0) + 1

        risk = output['agent_action']['risk_level']
        if risk in risk_dist:
            risk_dist[risk] += 1

        results.append(output)

    total = len(test_df)
    print(f"\n=== Agent Workflow Results ===")
    print(f"Total processed: {total}")
    print(f"\nAction distribution:")
    for action, count in sorted(action_map.items(),
                                key=lambda x: x[1], reverse=True):
        print(f"  {action}: {count} ({count/total*100:.1f}%)")

    print(f"\nRisk distribution:")
    for risk, count in risk_dist.items():
        print(f"  {risk}: {count} ({count/total*100:.1f}%)")

    payment_links = sum(
        1 for r in results
        if 'payment_link' in r['agent_action']
    )
    print(f"\nPayment links generated: {payment_links} ({payment_links/total*100:.1f}%)")

    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/agent_results.json", "w") as f:
        json.dump({
            "total":           total,
            "action_dist":     action_map,
            "risk_dist":       risk_dist,
            "payment_links":   payment_links,
            "sample_outputs":  results[:10],
        }, f, indent=2, ensure_ascii=False)

    print("\nSaved to outputs/agent_results.json")
    return results

if __name__ == "__main__":
    print("=== VAADA Agent Workflow Test ===\n")

    test_cases = [
        {
            "reminder":    "Rahul ji ₹5000 EMI 8 din overdue",
            "reply":       "bhai kal pakka kar dunga 🙏",
            "intent":      "promise_to_pay",
            "dpd":         8,
            "loan_amount": 5000,
        },
        {
            "reminder":    "₹12000 EMI bahut time se pending",
            "reply":       "aadha abhi de sakta hun 6000 baaki 15 tarikh",
            "intent":      "partial_payment",
            "dpd":         20,
            "loan_amount": 12000,
        },
        {
            "reminder":    "₹8000 overdue legal notice aa sakta hai",
            "reply":       "maine toh payment kar diya tha 3 din pehle",
            "intent":      "dispute",
            "dpd":         15,
            "loan_amount": 8000,
        },
        {
            "reminder":    "₹18000 bahut time overdue legal action",
            "reply":       "nahi karunga court mein milte hain",
            "intent":      "refusal",
            "dpd":         65,
            "loan_amount": 18000,
        },
    ]

    for tc in test_cases:
        result = run_pipeline(**tc)
        print(f"Intent     : {tc['intent']}")
        print(f"Actions    : {result['agent_action']['action']}")
        print(f"Message    : {result['agent_action']['message'][0][:80]}")
        print(f"Risk Level : {result['agent_action']['risk_level']}")
        print(f"Next Steps : {result['agent_action']['next_steps']}")
        print()

    print("\nRunning full evaluation...")
    evaluate_pipeline()
