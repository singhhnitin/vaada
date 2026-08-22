from openai import OpenAI
import json
import pandas as pd
import time
import os
import random
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/generation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

# ─────────────────────────────────────────────
# REAL FEW-SHOT EXAMPLES
# Grounded in actual Indian NBFC/bank collection
# patterns, RBI guidelines, ICICI/HDFC scripts
# ─────────────────────────────────────────────

FEW_SHOTS = {
    "promise_to_pay": [
        {
            "dpd": 5,
            "reminder": "Hi Rahul ji, aapki EMI ₹4500 ka due date kal tha. Abhi tak payment nahi aayi. Kripya aaj hi UPI se kar dein. Link: pay.rzp.io/xxx 🙏",
            "reply": "haan bhai kal pakka kar dunga, aaj thoda busy tha office mein",
            "ptp_date": "tomorrow",
            "ptp_amount": 4500,
            "tone": "polite"
        },
        {
            "dpd": 12,
            "reminder": "Amit ji, 2nd reminder - ₹8000 EMI 12 din se overdue hai. Aapka CIBIL score affect ho sakta hai. Aaj payment karein: pay.rzp.io/yyy",
            "reply": "bhai friday ko salary aayegi pakka usse pehle transfer kar dunga 🙏🙏",
            "ptp_date": "friday",
            "ptp_amount": 8000,
            "tone": "desperate"
        },
        {
            "dpd": 3,
            "reminder": "Priya mam, gentle reminder - ₹2200 EMI due hai. Please clear kar dein.",
            "reply": "yes abhi kar deti hun 2 min mein",
            "ptp_date": "immediate",
            "ptp_amount": 2200,
            "tone": "cooperative"
        }
    ],
    "needs_more_time": [
        {
            "dpd": 8,
            "reminder": "Suresh ji, ₹6500 overdue hai 8 din se. Please aaj settle karein warna late fees lagegi.",
            "reply": "bhai sach mein paisa nahi h abhi, mummy hospital mein hain, 10 din aur do please 🙏",
            "tone": "desperate",
            "excuse_type": "family_emergency"
        },
        {
            "dpd": 15,
            "reminder": "Vikram ji, aapki ₹12000 EMI bahut overdue ho gayi hai. Turant contact karein.",
            "reply": "yaar business mein thoda problem chal raha hai, next month pakka double kar dunga",
            "tone": "evasive",
            "excuse_type": "business_problem"
        },
        {
            "dpd": 6,
            "reminder": "Deepak bhai ₹3500 pending hai. Kab tak kar sakte ho?",
            "reply": "kal pe deadline hai mere client ka, wo paise aate hi turant kar dunga bhai",
            "tone": "polite",
            "excuse_type": "waiting_for_payment"
        }
    ],
    "partial_payment": [
        {
            "dpd": 10,
            "reminder": "Neha ji, ₹9000 EMI overdue hai. Kripya full payment karein.",
            "reply": "poora ek baar mein possible nahi hai, 4500 abhi kar sakti hun baaki 15 tarikh ko",
            "ptp_amount_now": 4500,
            "ptp_amount_later": 4500,
            "ptp_date_later": "15th",
            "tone": "cooperative"
        },
        {
            "dpd": 20,
            "reminder": "Ravi ji, ₹15000 bahut time se pending hai. Legal notice bhejni pad sakti hai.",
            "reply": "bhai 5000 abhi bhejta hun baaki 2 hafte mein, settle ho jayega",
            "ptp_amount_now": 5000,
            "ptp_amount_later": 10000,
            "tone": "worried"
        }
    ],
    "dispute": [
        {
            "dpd": 7,
            "reminder": "Anjali ji, ₹5500 EMI miss ho gayi hai. Please payment karein.",
            "reply": "maine toh 3 din pehle hi UPI kar diya tha, aapke system mein kuch gadbad hai",
            "dispute_type": "payment_already_made",
            "tone": "angry"
        },
        {
            "dpd": 4,
            "reminder": "Amit ji, ₹7000 due hai aapka.",
            "reply": "yeh amount galat hai, mera loan 6000 ka tha, extra charges kyun laga rahe ho",
            "dispute_type": "wrong_amount",
            "tone": "aggressive"
        },
        {
            "dpd": 9,
            "reminder": "Pooja ji, EMI pending hai please clear karein.",
            "reply": "mujhe koi reminder nahi aaya tha, mujhe bataya hi nahi due date ke baare mein",
            "dispute_type": "no_prior_notice",
            "tone": "confused"
        }
    ],
    "refusal": [
        {
            "dpd": 30,
            "reminder": "Rajesh ji, ₹18000 bahut time se overdue hai. Legal action lena pad sakta hai.",
            "reply": "band karo yeh messages, main court mein milta hun",
            "tone": "aggressive"
        },
        {
            "dpd": 25,
            "reminder": "Sunita ji, please ₹11000 clear karein.",
            "reply": "nahi kar sakta abhi, jo karna ho karo",
            "tone": "resigned"
        }
    ]
}

# ─────────────────────────────────────────────
# REAL SCENARIO PARAMETERS
# Based on actual DPD stages used by Indian NBFCs
# ─────────────────────────────────────────────

DPD_STAGES = {
    "soft":   {"range": (1, 15),  "tone": "polite, friendly"},
    "mid":    {"range": (15, 30), "tone": "firm, urgent, mentions CIBIL"},
    "hard":   {"range": (30, 60), "tone": "serious, legal notice warning"},
    "severe": {"range": (60, 90), "tone": "legal action, settlement offer"},
}

REGIONS = {
    "delhi": {
        "style": "Direct, uses 'yaar', 'bhai', 'pakka', 'turant'",
        "example_words": ["yaar", "bhai", "pakka", "turant", "sun", "bol"]
    },
    "mumbai": {
        "style": "Casual, uses 'arre', 'nako', 're', 'kay', mix of Marathi",
        "example_words": ["arre", "nako", "re", "kay", "thoda", "lagech"]
    },
    "hyderabad": {
        "style": "Formal-ish, uses 'boss', 'anna', mix of Telugu words",
        "example_words": ["boss", "anna", "enti", "okay ra", "definitely"]
    },
    "bangalore": {
        "style": "Tech-savvy, mix of Kannada, formal English",
        "example_words": ["sir", "illa", "bekku", "UPI maadtini", "transaction"]
    }
}

LOAN_TYPES = [
    "personal loan EMI",
    "BNPL (Buy Now Pay Later)",
    "credit line EMI",
    "bike loan EMI",
    "mobile loan EMI",
    "business loan EMI",
    "education loan EMI"
]

AMOUNTS = [1500, 2200, 3500, 4500, 5500, 6000,
           7500, 8000, 9000, 10000, 12000, 15000,
           18000, 22000, 25000]

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SINGLE_TURN_PROMPT = """You are generating training data for an AI payment 
collections system for Indian fintech (Razorpay-like platform).

Generate {n} realistic WhatsApp conversations between a collection agent 
and a loan defaulter in India. These must reflect REAL Indian collections 
scenarios, not generic templates.

CONTEXT:
- Loan type: {loan_type}
- DPD (Days Past Due): {dpd} days
- Stage tone: {stage_tone}
- Region/dialect: {region} — {region_style}
- Intent to generate: {intent}

FEW-SHOT EXAMPLES of this intent (use as reference for style, NOT copy):
{few_shots}

STRICT RULES:
1. Natural Hinglish code-switching — not forced, feels like real WhatsApp
2. Agent messages must follow RBI rules: no threats, no calls to family,
   no harassment — polite but firm
3. Customer replies must feel psychologically real:
   - stressed borrowers use more emoji and apologies
   - angry customers are blunt and short
   - evasive ones change topic or give vague timelines
4. Include real Indian context: CIBIL score, UPI links, EMI, 
   late fees, settlement offers where appropriate
5. Vary the vocabulary — do not repeat same phrases across samples
6. Typos and abbreviations are okay in customer messages

Return ONLY valid JSON array. No markdown. No explanation:
[
  {{
    "reminder": "agent WhatsApp message",
    "reply": "customer WhatsApp reply",
    "intent": "{intent}",
    "dpd": {dpd},
    "amount": <realistic amount>,
    "loan_type": "{loan_type}",
    "tone": "<polite|desperate|angry|evasive|cooperative|worried|resigned>",
    "region": "{region}",
    "ptp_date": "<tomorrow|specific day|next week|null>",
    "ptp_amount": <amount or null>,
    "dispute_type": "<wrong_amount|already_paid|no_notice|null>",
    "excuse_type": "<family_emergency|job_loss|business_problem|
                    waiting_for_payment|medical|null>",
    "cibil_mentioned": <true|false>,
    "legal_mentioned": <true|false>,
    "settlement_offered": <true|false>
  }}
]"""

MULTI_TURN_PROMPT = """Generate {n} realistic multi-turn WhatsApp collection 
conversations for Indian fintech. Each conversation spans multiple days 
and shows a realistic collections journey.

CONTEXT:
- Region: {region}
- DPD start: {dpd} days overdue
- Loan type: {loan_type}
- Final outcome: {outcome}

REALISTIC ESCALATION PATTERN to follow:
Day 1: Soft reminder (polite, payment link)
Day 3-4: Follow-up (firmer, mentions consequences)
Day 7-10: Serious notice (CIBIL, legal warning)
Day 15+: Settlement offer or legal action

PSYCHOLOGICAL REALISM:
- Customers who will pay: initially excuse, then commit, then pay
- Customers who default: give vague promises, go silent, 
  then dispute or refuse
- Partial payers: negotiate terms across turns
- Disputers: escalate complaint across turns

Return ONLY valid JSON array:
[
  {{
    "conversation_id": "conv_{idx}",
    "loan_type": "{loan_type}",
    "region": "{region}",
    "total_amount": <amount>,
    "final_outcome": "{outcome}",
    "turns": [
      {{
        "day": <day number>,
        "sender": "agent",
        "message": "message text",
        "dpd": <days past due at this point>
      }},
      {{
        "day": <day number>,
        "sender": "customer",
        "message": "reply text",
        "intent": "<intent>",
        "tone": "<tone>",
        "dpd": <same day>
      }}
    ],
    "ptp_extracted": {{
      "date_mentioned": "<date or null>",
      "amount_mentioned": <amount or null>,
      "kept": <true|false>
    }},
    "recovery_likelihood": "<high|medium|low>"
  }}
]"""


def build_few_shot_str(intent, n=2):
    examples = FEW_SHOTS.get(intent, [])
    selected = random.sample(examples, min(n, len(examples)))
    return json.dumps(selected, ensure_ascii=False, indent=2)


def call_api(prompt, retries=3, wait=2):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content.strip()

            # Clean markdown if model wraps in code block
            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    try:
                        return json.loads(part)
                    except Exception:
                        continue

            return json.loads(raw)

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(wait)

        except Exception as e:
            log.error(f"API error attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))

    log.error("All retries failed for this batch")
    return []


def generate_single_turn(per_intent=400):
    all_data = []
    intents = list(FEW_SHOTS.keys())
    batch_size = 20

    for intent in intents:
        log.info(f"\n=== Intent: {intent} ===")
        intent_data = []
        batches_needed = per_intent // batch_size

        for i in range(batches_needed):
            region = random.choice(list(REGIONS.keys()))
            stage = random.choice(list(DPD_STAGES.keys()))
            dpd = random.randint(*DPD_STAGES[stage]["range"])
            loan_type = random.choice(LOAN_TYPES)

            prompt = SINGLE_TURN_PROMPT.format(
                n=batch_size,
                intent=intent,
                dpd=dpd,
                stage_tone=DPD_STAGES[stage]["tone"],
                loan_type=loan_type,
                region=region,
                region_style=REGIONS[region]["style"],
                few_shots=build_few_shot_str(intent)
            )

            batch = call_api(prompt)
            if batch:
                intent_data.extend(batch)
                log.info(
                    f"  Batch {i+1}/{batches_needed} "
                    f"— {len(batch)} samples "
                    f"[{region}, DPD:{dpd}, {stage}]"
                )
            else:
                log.warning(f"  Batch {i+1} returned empty")

            time.sleep(1.2)

        all_data.extend(intent_data)
        log.info(f"  Subtotal for {intent}: {len(intent_data)}")

    return pd.DataFrame(all_data)


def generate_multi_turn(total=500):
    all_convos = []
    outcomes = ["paid_full", "paid_partial",
                "defaulted", "disputed", "settled"]
    batch_size = 10

    log.info("\n=== Multi-turn generation ===")

    for i in range(total // batch_size):
        region = random.choice(list(REGIONS.keys()))
        loan_type = random.choice(LOAN_TYPES)
        outcome = random.choice(outcomes)
        stage = random.choice(list(DPD_STAGES.keys()))
        dpd = random.randint(*DPD_STAGES[stage]["range"])

        prompt = MULTI_TURN_PROMPT.format(
            n=batch_size,
            idx=i * batch_size,
            region=region,
            loan_type=loan_type,
            outcome=outcome,
            dpd=dpd
        )

        batch = call_api(prompt)
        if batch:
            all_convos.extend(batch)
            log.info(
                f"  Batch {i+1}/{total//batch_size} "
                f"— {len(batch)} convos "
                f"[{region}, {outcome}, DPD:{dpd}]"
            )
        else:
            log.warning(f"  Batch {i+1} returned empty")

        time.sleep(1.2)

    return all_convos


def validate_dataframe(df):
    required = ["reminder", "reply", "intent",
                "dpd", "amount", "tone", "region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"Missing columns: {missing}")

    nulls = df[required].isnull().sum()
    log.info(f"\nNull counts:\n{nulls}")

    dupes = df.duplicated(subset=["reminder", "reply"]).sum()
    log.info(f"Duplicate rows: {dupes}")

    if dupes > 0:
        df = df.drop_duplicates(subset=["reminder", "reply"])
        log.info(f"After dedup: {len(df)} rows")

    return df


if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    log.info("=" * 50)
    log.info("VAADA — Data Generation Pipeline")
    log.info("Vernacular Agentic AI for Debt & Arrears")
    log.info("=" * 50)

    # ── Single-turn ──
    log.info("\n[Phase 1] Single-turn conversations")
    df = generate_single_turn(per_intent=400)

    df = validate_dataframe(df)

    train = df.sample(frac=0.7, random_state=42)
    remaining = df.drop(train.index)
    val = remaining.sample(frac=0.5, random_state=42)
    test = remaining.drop(val.index)

    train.to_csv("data/raw/train.csv", index=False)
    val.to_csv("data/raw/val.csv", index=False)
    test.to_csv("data/raw/test.csv", index=False)

    log.info(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    log.info(f"\nIntent distribution:\n{df['intent'].value_counts()}")
    log.info(f"\nRegion distribution:\n{df['region'].value_counts()}")
    log.info(f"\nTone distribution:\n{df['tone'].value_counts()}")
    log.info(f"\nDPD distribution:\n{df['dpd'].describe()}")

    # ── Multi-turn ──
    log.info("\n[Phase 2] Multi-turn conversations")
    multi = generate_multi_turn(total=500)

    with open("data/raw/multi_turn.json", "w",
              encoding="utf-8") as f:
        json.dump(multi, f, indent=2, ensure_ascii=False)

    log.info(f"\nMulti-turn saved: {len(multi)} conversations")

    log.info("\n" + "=" * 50)
    log.info("Generation complete. Check data/raw/")
    log.info("Logs saved to data/generation.log")
    log.info("=" * 50)
