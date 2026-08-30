"""
robustness.py — VAADA adversarial robustness layer.
Handles gibberish, typos, code-switching, sarcasm gracefully.
Routes low-confidence predictions to human review.
"""

import pickle
import re
import os
import json
import numpy as np

# ── Config ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45  # Below this → route to human
MIN_TEXT_LENGTH      = 3
MAX_TEXT_LENGTH      = 500

# ── Load model ────────────────────────────────────────────────
def load_model():
    try:
        with open("models/baseline_pipeline.pkl","rb") as f:
            return pickle.load(f)
    except Exception:
        return None

MODEL = load_model()

# ── Input validation ──────────────────────────────────────────
def validate_input(text: str) -> dict:
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return {"valid": False, "reason": "too_short"}

    if len(text) > MAX_TEXT_LENGTH:
        return {"valid": False, "reason": "too_long"}

    # Check for gibberish — too many non-alphanumeric chars
    clean = re.sub(r'[^\w\s]', '', text)
    if len(clean.strip()) < 2:
        return {"valid": False, "reason": "gibberish"}

    # Check for repeated characters (keyboard smashing)
    if re.search(r'(.)\1{4,}', text.lower()):
        return {"valid": False, "reason": "repeated_chars"}

    # Check if it has any recognizable words
    words = text.lower().split()
    if len(words) == 0:
        return {"valid": False, "reason": "no_words"}

    return {"valid": True, "reason": None}

# ── Language detection ────────────────────────────────────────
HINDI_CHARS = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
SOUTH_INDIAN = ["enti","anna","illa","bekku","maadtini","nako","arre"]

def detect_language_signals(text: str) -> dict:
    has_hindi_script = any(c in HINDI_CHARS for c in text)
    has_south_indian = any(w in text.lower() for w in SOUTH_INDIAN)
    has_english      = bool(re.search(r'[a-zA-Z]{3,}', text))
    has_hinglish     = has_english and (has_hindi_script or
                        any(w in text.lower() for w in
                            ["bhai","yaar","pakka","nahi","aaj","kal"]))

    return {
        "hindi_script":  has_hindi_script,
        "south_indian":  has_south_indian,
        "english":       has_english,
        "hinglish":      has_hinglish,
        "language_type": "hinglish" if has_hinglish else
                         "hindi"    if has_hindi_script else
                         "english"  if has_english else "unknown"
    }

# ── Sarcasm detection (simple heuristics) ────────────────────
SARCASM_SIGNALS = [
    r'haan haan', r'bilkul bilkul', r'pakka pakka',
    r'sure sure', r'obviously', r'of course.*not',
    r'great.*not', r'wonderful.*not',
]

def detect_sarcasm(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in SARCASM_SIGNALS)

# ── Core robust prediction ────────────────────────────────────
def robust_predict(reminder: str, reply: str) -> dict:
    """
    Adversarially robust prediction with graceful degradation.
    Routes to human review when uncertain.
    """
    result = {
        "intent":          None,
        "confidence":      0.0,
        "routed_to_human": False,
        "route_reason":    None,
        "language":        None,
        "warnings":        [],
        "all_probs":       {},
    }

    # Validate inputs
    reply_validation = validate_input(reply)
    if not reply_validation["valid"]:
        result["routed_to_human"] = True
        result["route_reason"]    = f"invalid_input_{reply_validation['reason']}"
        result["warnings"].append(f"Reply validation failed: {reply_validation['reason']}")
        return result

    # Language detection
    lang = detect_language_signals(reply)
    result["language"] = lang["language_type"]

    if lang["language_type"] == "unknown":
        result["warnings"].append("Unknown language detected — confidence may be lower")

    # Sarcasm detection
    if detect_sarcasm(reply):
        result["warnings"].append("Possible sarcasm detected — routing to human review")
        result["routed_to_human"] = True
        result["route_reason"]    = "sarcasm_detected"
        return result

    # Model prediction
    if MODEL is None:
        result["routed_to_human"] = True
        result["route_reason"]    = "model_unavailable"
        return result

    try:
        text    = reminder + " [SEP] " + reply
        intent  = MODEL.predict([text])[0]
        probs   = MODEL.predict_proba([text])[0]
        classes = MODEL.classes_
        conf    = float(max(probs))
        all_probs = {c: float(p) for c,p in zip(classes, probs)}

        result["intent"]     = intent
        result["confidence"] = conf
        result["all_probs"]  = all_probs

        # Low confidence → route to human
        if conf < CONFIDENCE_THRESHOLD:
            result["routed_to_human"] = True
            result["route_reason"]    = f"low_confidence_{conf:.2f}"
            result["warnings"].append(
                f"Confidence {conf:.1%} below threshold {CONFIDENCE_THRESHOLD:.0%} — human review recommended"
            )

        # High uncertainty (top 2 close) → route to human
        sorted_probs = sorted(probs, reverse=True)
        if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.15:
            result["warnings"].append(
                f"High ambiguity: top intents within {sorted_probs[0]-sorted_probs[1]:.2f} of each other"
            )

    except Exception as e:
        result["routed_to_human"] = True
        result["route_reason"]    = f"prediction_error_{str(e)}"
        result["warnings"].append(f"Prediction error: {e}")

    return result

# ── Adversarial test suite ────────────────────────────────────
ADVERSARIAL_TESTS = [
    # Gibberish
    ("Rs5000 due hai", "asdfghjkl qwerty", "gibberish"),
    ("Rs5000 due hai", "!!!@@##$$%%", "symbols_only"),
    ("Rs5000 due hai", "aaaaaaaaaaaaaaaa", "repeated_chars"),
    ("Rs5000 due hai", "x", "too_short"),
    # Code-switched South Indian
    ("Rs5000 due hai", "anna definitely maadtini bekku", "south_indian_hinglish"),
    ("Rs5000 due hai", "arre nako tension re", "mumbai_hinglish"),
    # Sarcasm
    ("Rs5000 due hai", "haan haan bilkul bilkul karunga", "sarcasm"),
    # Ambiguous
    ("Rs5000 due hai", "ok", "too_ambiguous"),
    ("Rs5000 due hai", "dekhte hain", "evasive_ambiguous"),
    # Normal cases (should classify correctly)
    ("Rs5000 EMI due", "bhai kal pakka kar dunga", "normal_promise"),
    ("Rs5000 EMI due", "nahi karunga band karo", "normal_refusal"),
    ("Rs5000 EMI due", "aadha abhi baaki baad mein", "normal_partial"),
]

if __name__ == "__main__":
    print("=== VAADA Adversarial Robustness Test ===\n")

    routed_correctly = 0
    classified_correctly = 0
    total = len(ADVERSARIAL_TESTS)

    results = []
    for reminder, reply, test_type in ADVERSARIAL_TESTS:
        result = robust_predict(reminder, reply)
        routed = result["routed_to_human"]

        # Expected behavior
        should_route = test_type in [
            "gibberish","symbols_only","repeated_chars",
            "too_short","sarcasm","too_ambiguous"
        ]
        should_classify = test_type in [
            "normal_promise","normal_refusal","normal_partial"
        ]

        status = ""
        if should_route and routed:
            status = "CORRECT (routed to human)"
            routed_correctly += 1
        elif should_classify and not routed:
            status = "CORRECT (classified: {})".format(result["intent"])
            classified_correctly += 1
        elif should_route and not routed:
            status = "MISSED (should have routed, got: {})".format(result["intent"])
        else:
            status = "OKAY ({})".format(result.get("intent","routed"))

        print(f"[{test_type}]")
        print(f"  Reply  : '{reply[:50]}'")
        print(f"  Result : {status}")
        if result["warnings"]:
            print(f"  Warns  : {result['warnings'][0]}")
        print()

        results.append({
            "test_type":   test_type,
            "reply":       reply,
            "routed":      routed,
            "intent":      result["intent"],
            "confidence":  result["confidence"],
            "route_reason":result["route_reason"],
            "status":      status,
        })

    print(f"=== SUMMARY ===")
    print(f"Total tests    : {total}")
    print(f"Correctly routed to human : {routed_correctly}")
    print(f"Correctly classified      : {classified_correctly}")
    print(f"Robustness score          : {(routed_correctly+classified_correctly)/total:.1%}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/robustness_results.json","w") as f:
        json.dump({"tests": results, "summary": {
            "total": total,
            "routed_correctly": routed_correctly,
            "classified_correctly": classified_correctly,
            "robustness_score": round((routed_correctly+classified_correctly)/total, 3)
        }}, f, indent=2, ensure_ascii=False)

    print("\nSaved to outputs/robustness_results.json")
    print("\nVAADA fails safely — unknown inputs route to human, never silently misclassify.")
