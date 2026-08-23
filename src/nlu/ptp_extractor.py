"""
ptp_extractor.py — Promise-to-Pay extractor.
Extracts structured commitment data from Hinglish customer replies.
"""

import re
import json
import pandas as pd
from datetime import datetime, timedelta

# ── Date patterns in Hinglish ─────────────────────────────────
DATE_PATTERNS = [
    (r'\bkal\b',                    1,   "tomorrow"),
    (r'\bparso\b',                  2,   "day after tomorrow"),
    (r'\baaj\b',                    0,   "today"),
    (r'\bshaam tak\b',              0,   "today evening"),
    (r'\bfriday\b|\bshukravar\b',   None, "friday"),
    (r'\bsaturday\b|\bshanivar\b',  None, "saturday"),
    (r'\bmonday\b|\bsomvar\b',      None, "monday"),
    (r'\btuesday\b|\bmangalvar\b',  None, "tuesday"),
    (r'\bwednesday\b|\bbudhvar\b',  None, "wednesday"),
    (r'\bthursday\b|\bgurvar\b',    None, "thursday"),
    (r'\b(\d+)\s*din\b',            None, "n_days"),
    (r'\b(\d+)\s*hafte\b',          None, "n_weeks"),
    (r'\bnext week\b|\bagla hafte\b', 7,  "next week"),
    (r'\b(\d{1,2})\s*tarikh\b',     None, "specific_date"),
    (r'\bmahine end\b|\bmonth end\b', None, "month end"),
    (r'\b(\d{1,2})[/-](\d{1,2})\b', None, "date_format"),
]

# ── Amount patterns ───────────────────────────────────────────
AMOUNT_PATTERNS = [
    r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)',
    r'(\d+(?:,\d+)*)\s*(?:rupees?|rs\.?|रुपए)',
    r'(\d+(?:,\d+)*)\s*(?:ka|ke|ki)\s*(?:payment|amount)',
    r'(?:aadha|half)\s*(?:yani|matlab|means?)?\s*₹?\s*(\d+)',
    r'(\d+)\s*(?:abhi|now|pehle)',
    r'(?:total|kul)\s*₹?\s*(\d+)',
]

# ── Partial payment indicators ────────────────────────────────
PARTIAL_INDICATORS = [
    'aadha', 'half', '50%', 'fifty percent',
    'kuch', 'thoda', 'partial', 'pehle',
    'abhi itna', 'baaki baad'
]

def extract_date(text: str) -> dict:
    text_lower = text.lower()
    result = {"raw": None, "days_from_now": None, "confidence": 0.0}

    for pattern, days, label in DATE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            result["raw"] = label
            result["confidence"] = 0.9

            if label == "n_days":
                try:
                    result["days_from_now"] = int(match.group(1))
                except:
                    pass

            elif label == "n_weeks":
                try:
                    result["days_from_now"] = int(match.group(1)) * 7
                except:
                    pass

            elif label == "specific_date":
                try:
                    day = int(match.group(1))
                    today = datetime.now()
                    target = today.replace(day=day)
                    if target < today:
                        if today.month == 12:
                            target = target.replace(
                                year=today.year + 1, month=1
                            )
                        else:
                            target = target.replace(month=today.month + 1)
                    result["days_from_now"] = (target - today).days
                    result["raw"] = f"{day}th"
                except:
                    pass

            elif days is not None:
                result["days_from_now"] = days

            break

    return result

def extract_amount(text: str, loan_amount: float = None) -> dict:
    result = {"amount": None, "is_partial": False, "confidence": 0.0}
    text_lower = text.lower()

    for indicator in PARTIAL_INDICATORS:
        if indicator in text_lower:
            result["is_partial"] = True
            break

    for pattern in AMOUNT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            try:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                result["amount"] = amount
                result["confidence"] = 0.9

                if loan_amount and amount < loan_amount:
                    result["is_partial"] = True

                break
            except:
                pass

    if result["amount"] is None and loan_amount and result["is_partial"]:
        result["amount"] = loan_amount * 0.5
        result["confidence"] = 0.5

    return result

def extract_ptp(reminder: str, reply: str,
                loan_amount: float = None) -> dict:
    date_info   = extract_date(reply)
    amount_info = extract_amount(reply, loan_amount)

    ptp = {
        "has_ptp": False,
        "ptp_date": date_info,
        "ptp_amount": amount_info,
        "commitment_strength": "none",
        "follow_up_in_days": None,
    }

    # Determine if there is a PTP
    has_date   = date_info["raw"] is not None
    has_amount = amount_info["amount"] is not None

    if has_date or has_amount:
        ptp["has_ptp"] = True

    # Commitment strength
    strong_words = [
        'pakka', 'definitely', 'zaroor', 'promise',
        'guarantee', '100%', 'confirm'
    ]
    weak_words = [
        'shayad', 'maybe', 'try karunga', 'dekhta hun',
        'koshish', 'hopefully'
    ]

    reply_lower = reply.lower()
    if any(w in reply_lower for w in strong_words):
        ptp["commitment_strength"] = "strong"
    elif any(w in reply_lower for w in weak_words):
        ptp["commitment_strength"] = "weak"
    elif ptp["has_ptp"]:
        ptp["commitment_strength"] = "moderate"

    # Follow-up timing
    if date_info["days_from_now"] is not None:
        ptp["follow_up_in_days"] = date_info["days_from_now"] + 1
    elif has_date:
        ptp["follow_up_in_days"] = 3

    return ptp

def evaluate_on_test_set():
    test_df = pd.read_csv("data/processed/test.csv")
    ptp_df  = test_df[test_df['intent'] == 'promise_to_pay'].copy()

    print(f"Evaluating PTP extraction on {len(ptp_df)} promise_to_pay samples...")

    results = []
    date_hits  = 0
    amount_hits = 0

    for _, row in ptp_df.iterrows():
        ptp = extract_ptp(
            reminder    = str(row['reminder']),
            reply       = str(row['reply']),
            loan_amount = row.get('amount', None)
        )

        # Check against ground truth
        gt_date   = str(row.get('ptp_date', '')).lower()
        gt_amount = row.get('ptp_amount', None)

        date_extracted   = ptp['ptp_date']['raw'] is not None
        amount_extracted = ptp['ptp_amount']['amount'] is not None

        if date_extracted and gt_date not in ['null', 'nan', '', 'none']:
            date_hits += 1

        if amount_extracted and pd.notna(gt_amount):
            amount_hits += 1

        results.append({
            "reply":              row['reply'],
            "extracted_date":     ptp['ptp_date']['raw'],
            "gt_date":            gt_date,
            "extracted_amount":   ptp['ptp_amount']['amount'],
            "gt_amount":          gt_amount,
            "commitment":         ptp['commitment_strength'],
            "has_ptp":            ptp['has_ptp'],
        })

    total = len(ptp_df)
    date_recall   = date_hits / total if total > 0 else 0
    amount_recall = amount_hits / total if total > 0 else 0

    print(f"\n=== PTP Extractor Results ===")
    print(f"Date extraction recall   : {date_recall:.4f}")
    print(f"Amount extraction recall : {amount_recall:.4f}")

    # Show samples
    print("\n=== Sample Extractions ===")
    for r in results[:5]:
        print(f"\nReply   : {r['reply'][:80]}")
        print(f"Date    : {r['extracted_date']} (GT: {r['gt_date']})")
        print(f"Amount  : {r['extracted_amount']} (GT: {r['gt_amount']})")
        print(f"Commit  : {r['commitment']}")

    import json, os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ptp_results.json", "w") as f:
        json.dump({
            "date_recall":   round(date_recall, 4),
            "amount_recall": round(amount_recall, 4),
            "samples":       results[:20]
        }, f, indent=2, ensure_ascii=False)

    print("\nSaved to outputs/ptp_results.json")
    return date_recall, amount_recall

if __name__ == "__main__":
    print("Testing PTP extractor on samples...\n")

    test_cases = [
        {
            "reminder": "Rahul ji ₹5000 EMI overdue hai",
            "reply":    "bhai kal pakka kar dunga 🙏",
            "amount":   5000
        },
        {
            "reminder": "₹8000 pending hai please clear karo",
            "reply":    "friday tak transfer ho jayega",
            "amount":   8000
        },
        {
            "reminder": "₹12000 EMI miss ho gayi",
            "reply":    "aadha abhi de sakta hun 6000 baaki 15 tarikh ko",
            "amount":   12000
        },
        {
            "reminder": "₹3500 overdue hai",
            "reply":    "2-3 din aur do please paisa aa raha hai",
            "amount":   3500
        },
        {
            "reminder": "₹9000 pending",
            "reply":    "nahi kar sakta abhi",
            "amount":   9000
        },
    ]

    for tc in test_cases:
        result = extract_ptp(tc["reminder"], tc["reply"], tc["amount"])
        print(f"Reply  : {tc['reply']}")
        print(f"PTP    : {result['has_ptp']}")
        print(f"Date   : {result['ptp_date']['raw']}")
        print(f"Amount : {result['ptp_amount']['amount']}")
        print(f"Commit : {result['commitment_strength']}")
        print()

    print("\nRunning evaluation on test set...")
    evaluate_on_test_set()
