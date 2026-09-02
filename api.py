"""
api.py — Flask API wrapping VAADA's existing pipeline for the static frontend.
Run alongside app.py; does not modify or interfere with the Streamlit app.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

from src.nlu.ptp_extractor import extract_ptp
from src.nlu.recovery_predictor import engineer_features

app = Flask(__name__)
CORS(app)  # allow requests from GitHub Pages / any origin

# ── Load models once at startup ─────────────────────────────
def load_models():
    m = {}
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base, "models", "baseline_pipeline.pkl"), "rb") as f:
            m["intent"] = pickle.load(f)
    except Exception as e:
        print(f"Intent model load error: {e}")
        m["intent"] = None
    try:
        with open(os.path.join(base, "models", "recovery_predictor.pkl"), "rb") as f:
            m["recovery"] = pickle.load(f)
    except Exception as e:
        print(f"Recovery model load error: {e}")
        m["recovery"] = None
    return m

MODELS = load_models()

import random

def run_pipeline(reminder, reply, dpd, amount, region, tone):
    res = {}
    if MODELS["intent"]:
        text = reminder + " [SEP] " + reply
        intent = MODELS["intent"].predict([text])[0]
        probs = MODELS["intent"].predict_proba([text])[0]
        res["intent"] = intent
        res["conf"] = float(max(probs))
        res["probs"] = {c: float(p) for c, p in zip(MODELS["intent"].classes_, probs)}
    else:
        res["intent"] = "promise_to_pay"
        res["conf"] = 0.95
        res["probs"] = {}

    ptp = extract_ptp(reminder, reply, amount)
    res["ptp"] = ptp

    if MODELS["recovery"]:
        row = {"intent": res["intent"], "tone": tone, "dpd": dpd, "amount": amount,
               "region": region, "reply": reply, "reminder": reminder,
               "cibil_mentioned": False, "legal_mentioned": False}
        feat = engineer_features(pd.DataFrame([row]))
        rec = MODELS["recovery"].predict(feat)[0]
        probs_r = MODELS["recovery"].predict_proba(feat)[0]
        res["recovery"] = rec
        res["recovery_conf"] = float(max(probs_r))
    else:
        res["recovery"] = "high"
        res["recovery_conf"] = 1.0

    intent = res["intent"]
    ptp_amt = ptp["ptp_amount"]["amount"] or amount
    partial = ptp["ptp_amount"]["is_partial"]
    days_out = ptp["ptp_date"].get("days_from_now", 1) or 1
    lid = "rzp_{}_{}".format("p" if partial else "f", random.randint(10000, 99999))

    if intent == "promise_to_pay":
        res["action"] = "SEND_PARTIAL_PAYMENT_LINK" if partial else "SEND_FULL_PAYMENT_LINK"
        res["link"] = "https://rzp.io/l/" + lid
        res["link_amount"] = ptp_amt
        res["follow_up"] = days_out + 1
    elif intent == "partial_payment":
        res["action"] = "SEND_PARTIAL_PAYMENT_LINK"
        res["link"] = "https://rzp.io/l/" + lid
        res["link_amount"] = ptp["ptp_amount"]["amount"] or amount * 0.5
        res["follow_up"] = 7
    elif intent == "needs_more_time":
        res["action"] = "SEND_SETTLEMENT_OFFER" if dpd > 60 else "SCHEDULE_FOLLOWUP"
        res["follow_up"] = 1 if dpd > 60 else days_out
    elif intent == "dispute":
        res["action"] = "FLAG_FOR_HUMAN_REVIEW"
        res["ticket"] = "VAADA-{}".format(random.randint(1000, 9999))
        res["follow_up"] = 1
    elif intent == "refusal":
        res["action"] = "TRIGGER_LEGAL_NOTICE" if dpd > 60 else "ESCALATE_TO_SENIOR_TEAM"
        res["follow_up"] = 0 if dpd > 60 else 1
    else:
        res["action"] = "SCHEDULE_FOLLOWUP"
        res["follow_up"] = 3

    return res


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "intent_model_loaded": MODELS["intent"] is not None,
        "recovery_model_loaded": MODELS["recovery"] is not None,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    reminder = data.get("reminder", "")
    reply = data.get("reply", "")
    dpd = int(data.get("dpd", 8))
    amount = float(data.get("amount", 5000))
    region = data.get("region", "delhi")
    tone = data.get("tone", "polite")

    if not reply:
        return jsonify({"error": "reply is required"}), 400

    try:
        result = run_pipeline(reminder, reply, dpd, amount, region, tone)
        # convert numpy/non-serializable types safely
        clean = {
            "intent": str(result.get("intent")),
            "confidence": round(result.get("conf", 0), 4),
            "probs": {k: round(v, 4) for k, v in result.get("probs", {}).items()},
            "ptp_date": result["ptp"]["ptp_date"].get("raw"),
            "ptp_amount": result["ptp"]["ptp_amount"].get("amount"),
            "has_ptp": result["ptp"].get("has_ptp", False),
            "recovery": str(result.get("recovery")),
            "recovery_confidence": round(result.get("recovery_conf", 0), 4),
            "action": result.get("action"),
            "link": result.get("link"),
            "link_amount": result.get("link_amount"),
            "ticket": result.get("ticket"),
            "follow_up_days": result.get("follow_up"),
        }
        return jsonify(clean)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
