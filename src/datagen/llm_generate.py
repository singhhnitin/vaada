from openai import OpenAI
import json, os, time, logging, random
from pathlib import Path

# ── Logging ──────────────────────────────────
Path("data").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/generation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Client ───────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)
MODEL  = "meta/llama-3.1-70b-instruct"
BATCH  = 5       # safe size — JSON never truncates
TOKENS = 1500    # enough for 5 samples

# ── Domain Knowledge ─────────────────────────
INTENTS = [
    "promise_to_pay",
    "needs_more_time",
    "partial_payment",
    "dispute",
    "refusal"
]

REGIONS = {
    "delhi":     "yaar, bhai, pakka, sun, turant, bol",
    "mumbai":    "arre, nako, re, kay, thoda, lagech",
    "hyderabad": "boss, anna, okay ra, definitely, enti",
    "bangalore": "sir, illa, UPI maadtini, bekku, transaction",
}

DPD = {
    "soft":   (1,  15,  "polite, friendly, payment link"),
    "mid":    (15, 30,  "firm, mentions CIBIL score impact"),
    "hard":   (30, 60,  "serious, legal notice warning"),
    "severe": (60, 90,  "settlement offer, legal action"),
}

LOANS = [
    "personal loan EMI",
    "BNPL payment",
    "credit line EMI",
    "bike loan EMI",
    "mobile loan EMI",
    "business loan EMI",
]

AMOUNTS = [
    1500, 2200, 3500, 4500, 5500,
    6000, 7500, 8000, 9000, 10000,
    12000, 15000, 18000, 22000, 25000
]

# ── Real Few-Shot Examples ────────────────────
# Grounded in actual ICICI/HDFC/NBFC collection
# scripts and RBI-compliant communication patterns
SHOTS = {
    "promise_to_pay": [
        ("Hi Rahul ji, ₹4500 EMI kal due thi. Aaj tak payment nahi aayi. "
         "UPI se kar dein: rzp.io/pay 🙏",
         "haan bhai kal pakka kar dunga, aaj office mein busy tha"),
        ("Amit ji 2nd reminder — ₹8000 EMI 12 din overdue. CIBIL affect "
         "hoga. Abhi karein: rzp.io/pay",
         "bhai friday salary aayegi pakka usse pehle transfer 🙏🙏"),
    ],
    "needs_more_time": [
        ("Suresh ji ₹6500 overdue 8 din. Late fees lag rahi hai aaj settle "
         "karein.",
         "bhai sach mein paisa nahi h, mummy hospital mein hain 10 din "
         "aur do please 🙏"),
        ("Vikram ji ₹12000 EMI bahut overdue. Turant contact karein.",
         "yaar business mein problem chal raha next month double kar dunga"),
    ],
    "partial_payment": [
        ("Neha ji ₹9000 EMI overdue. Full payment karein.",
         "poora ek baar mein nahi hoga, 4500 abhi baaki 15 tarikh ko"),
        ("Ravi ji ₹15000 pending. Legal notice aa sakta hai.",
         "bhai 5000 abhi bhejta hun baaki 2 hafte mein pakka"),
    ],
    "dispute": [
        ("Anjali ji ₹5500 EMI miss ho gayi. Payment karein.",
         "maine 3 din pehle UPI kar diya tha aapke system mein gadbad hai"),
        ("Amit ji ₹7000 due hai.",
         "yeh amount galat hai mera loan 6000 ka tha extra charges kyun"),
    ],
    "refusal": [
        ("Rajesh ji ₹18000 overdue. Legal action lena padega.",
         "band karo yeh messages main court mein milta hun"),
        ("Sunita ji ₹11000 clear karein please.",
         "nahi kar sakta abhi jo karna ho karo"),
    ],
}

# ── Prompt ───────────────────────────────────
PROMPT = """\
Generate {n} WhatsApp payment collection conversations (India, Hinglish).

Setup:
- Intent: {intent}
- Loan: {loan}
- DPD: {dpd} days overdue
- Agent tone: {tone}
- Region dialect words: {dialect}

Reference examples (vary style, do NOT copy):
{shots}

Rules:
1. Natural Hinglish — real code-switching, not translation
2. Agent: RBI-compliant, no threats, polite-firm
3. Customer: psychologically real — stressed=emoji+apology,
   angry=blunt+short, evasive=vague timelines
4. Use: CIBIL, UPI, EMI, late fees, settlement where natural
5. Different vocab each sample — no repetition

Return ONLY a JSON array, no markdown, no explanation:
[{{"reminder":"...","reply":"...","intent":"{intent}",
   "dpd":{dpd},"amount":<int>,"loan":"{loan}",
   "tone":"<polite|desperate|angry|evasive|cooperative|worried|resigned>",
   "region":"{region}","ptp_date":"<day or null>",
   "ptp_amount":<int or null>,"dispute_type":"<type or null>",
   "excuse_type":"<type or null>","cibil_mentioned":<true|false>,
   "legal_mentioned":<true|false>}}]
"""

# ── JSON Extractor ────────────────────────────
def extract_json(raw: str):
    """4-layer extraction — handles all model output formats."""
    raw = raw.strip()

    # Layer 1: direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Layer 2: strip markdown fences
    for fence in ["```json", "```"]:
        if fence in raw:
            parts = raw.split(fence)
            for p in parts:
                p = p.strip().rstrip("`").strip()
                try:
                    return json.loads(p)
                except Exception:
                    pass

    # Layer 3: find first [ ... last ]
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end+1])
        except Exception:
            pass

    # Layer 4: salvage complete objects from truncated array
    start = raw.find("[")
    if start != -1:
        partial = raw[start:]
        depth, i, objs, buf = 0, 0, [], ""
        in_str, escape = False, False
        for ch in partial:
            buf += ch
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_str = not in_str
            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(buf.lstrip(",").strip())
                            objs.append(obj)
                            buf = ""
                        except Exception:
                            buf = ""
        if objs:
            log.warning(f"Salvaged {len(objs)} objects from truncated JSON")
            return objs

    return None

# ── API Call ─────────────────────────────────
def call_api(prompt: str, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=TOKENS,
            )
            raw = resp.choices[0].message.content
            result = extract_json(raw)
            if result:
                return result
            log.warning(f"Extraction failed attempt {attempt+1}")

        except Exception as e:
            code = getattr(e, "status_code", None)
            if code == 429:
                wait = 30 * (attempt + 1)
                log.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif code == 401:
                log.error("Auth failed. Check NVIDIA_API_KEY.")
                raise
            else:
                log.error(f"API error attempt {attempt+1}: {e}")
                time.sleep(3 * (attempt + 1))

    return []

# ── Save JSONL ───────────────────────────────
def append_jsonl(path: str, records: list):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ── Single-Turn Generation ───────────────────
def generate_single_turn(per_intent=200):
    out = "data/raw/single_turn.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    total = 0
    for intent in INTENTS:
        log.info(f"\n=== Intent: {intent} ===")
        intent_total = 0
        batches = per_intent // BATCH
        shots_str = json.dumps(
            SHOTS.get(intent, []), ensure_ascii=False
        )

        for i in range(batches):
            stage      = random.choice(list(DPD.keys()))
            lo, hi, tn = DPD[stage]
            dpd        = random.randint(lo, hi)
            region     = random.choice(list(REGIONS.keys()))
            loan       = random.choice(LOANS)

            prompt = PROMPT.format(
                n       = BATCH,
                intent  = intent,
                loan    = loan,
                dpd     = dpd,
                tone    = tn,
                dialect = REGIONS[region],
                region  = region,
                shots   = shots_str,
            )

            batch = call_api(prompt)
            if batch:
                # Ensure required fields exist
                clean = []
                for rec in batch:
                    if "reminder" in rec and "reply" in rec:
                        rec.setdefault("intent", intent)
                        rec.setdefault("region", region)
                        rec.setdefault("dpd", dpd)
                        rec.setdefault("loan", loan)
                        clean.append(rec)

                append_jsonl(out, clean)
                intent_total += len(clean)
                log.info(
                    f"  [{i+1}/{batches}] +{len(clean)} "
                    f"[{region}, DPD:{dpd}, {stage}] "
                    f"total={intent_total}"
                )
            else:
                log.warning(f"  [{i+1}/{batches}] empty batch")

            time.sleep(1.0)

        total += intent_total
        log.info(f"  Subtotal {intent}: {intent_total}")

    log.info(f"\nSingle-turn total: {total}")
    return out

# ── Multi-Turn Generation ────────────────────
MULTI_PROMPT = """\
Generate {n} multi-turn WhatsApp loan collection threads (India, Hinglish).
Each thread has 3-5 turns across multiple days showing a real collections arc.

Region: {region} | Loan: {loan} | Start DPD: {dpd} | Outcome: {outcome}

Arc patterns:
- paid_full: excuses → commits → pays
- paid_partial: negotiates → partial now → rest later
- defaulted: vague promises → goes silent → no payment
- disputed: claims already paid or wrong amount → escalates
- settled: negotiates discount → agrees to settlement

Return ONLY JSON array, no markdown:
[{{"id":"c{idx}_{i}","loan":"{loan}","region":"{region}",
   "amount":<int>,"outcome":"{outcome}",
   "recovery_likelihood":"<high|medium|low>",
   "ptp_kept":<true|false>,
   "turns":[
     {{"day":<int>,"sender":"agent","msg":"...","dpd":<int>}},
     {{"day":<int>,"sender":"customer","msg":"...","intent":"...","tone":"..."}}
   ]}}]
"""

def generate_multi_turn(total=200):
    out = "data/raw/multi_turn.jsonl"
    outcomes = [
        "paid_full", "paid_partial",
        "defaulted", "disputed", "settled"
    ]
    batches = total // BATCH
    count   = 0

    log.info("\n=== Multi-turn generation ===")

    for i in range(batches):
        region  = random.choice(list(REGIONS.keys()))
        loan    = random.choice(LOANS)
        outcome = random.choice(outcomes)
        stage   = random.choice(list(DPD.keys()))
        lo, hi, _ = DPD[stage]
        dpd     = random.randint(lo, hi)

        prompt = MULTI_PROMPT.format(
            n=BATCH, idx=i,
            region=region, loan=loan,
            dpd=dpd, outcome=outcome, i=i
        )

        batch = call_api(prompt)
        if batch:
            append_jsonl(out, batch)
            count += len(batch)
            log.info(
                f"  [{i+1}/{batches}] +{len(batch)} "
                f"[{region}, {outcome}, DPD:{dpd}] "
                f"total={count}"
            )
        else:
            log.warning(f"  [{i+1}/{batches}] empty batch")

        time.sleep(1.0)

    log.info(f"Multi-turn total: {count}")
    return out

# ── JSONL → CSV ──────────────────────────────
def jsonl_to_splits(path: str):
    import pandas as pd
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["reminder", "reply"])
    log.info(f"\nLoaded {len(df)} unique single-turn samples")
    log.info(f"Intent dist:\n{df['intent'].value_counts()}")
    log.info(f"Region dist:\n{df['region'].value_counts()}")

    train = df.sample(frac=0.7, random_state=42)
    rest  = df.drop(train.index)
    val   = rest.sample(frac=0.5, random_state=42)
    test  = rest.drop(val.index)

    train.to_csv("data/raw/train.csv", index=False)
    val.to_csv("data/raw/val.csv",     index=False)
    test.to_csv("data/raw/test.csv",   index=False)

    log.info(
        f"Splits — Train:{len(train)} "
        f"Val:{len(val)} Test:{len(test)}"
    )

# ── Main ─────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 50)
    log.info("VAADA — Data Generation Pipeline")
    log.info("Vernacular Agentic AI for Debt & Arrears")
    log.info("=" * 50)

    # Phase 1
    single_path = generate_single_turn(per_intent=200)
    jsonl_to_splits(single_path)

    # Phase 2
    generate_multi_turn(total=200)

    log.info("\n" + "=" * 50)
    log.info("Done. Files in data/raw/")
    log.info("=" * 50)
single_path = generate_single_turn(per_intent=600)
generate_multi_turn(total=500)
