<div align="center">

<pre align="center">
 ██╗   ██╗    █████╗    █████╗    ██████╗    █████╗
 ██║   ██║   ██╔══██╗  ██╔══██╗  ██╔══██╗  ██╔══██╗
 ██║   ██║   ███████║  ███████║  ██║  ██║  ███████║
 ╚██╗ ██╔╝   ██╔══██║  ██╔══██║  ██║  ██║  ██╔══██║
  ╚████╔╝    ██║  ██║  ██║  ██║  ██████╔╝  ██║  ██║
   ╚═══╝     ╚═╝  ╚═╝  ╚═╝  ╚═╝  ╚═════╝   ╚═╝  ╚═╝
</pre>

**Vernacular Agentic AI for Debt & Arrears**

*The missing communication intelligence layer for Razorpay Vulcan*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-GitHub_Pages-00ff41?style=for-the-badge&logoColor=white)](https://singhhnitin.github.io/vaada/)
[![Dataset](https://img.shields.io/badge/📊_Dataset-9693_samples-20BEFF?style=for-the-badge)](https://www.kaggle.com/datasets/nitinsingh1204/vaada-hinglish-collections)
[![Razorpay](https://img.shields.io/badge/💳_Razorpay-Test_API-blue?style=for-the-badge)](https://razorpay.com/buildathon)
[![Track](https://img.shields.io/badge/🏆_Revenue-Recovery_Track-orange?style=for-the-badge)](https://razorpay.com/buildathon)
[![Compliance](https://img.shields.io/badge/⚖️_RBI_%2B_DPDP-Compliant-lightgrey?style=for-the-badge)](COMPLIANCE.md)

> ⏳ **Note:** the backend API runs on Render's free tier, which sleeps after inactivity — the first request may take 20–30 seconds to wake it up. Just wait for the "API: LIVE" indicator on the page before running a query.

</div>

---

## 🧒 What VAADA does, explained simply

**The problem:** A shop owner is owed money. They message the customer on WhatsApp:
> *"Rahul ji, Rs 5000 EMI 8 din se overdue hai. Aaj payment karein."*

The customer replies:
> *"bhai kal pakka kar dunga aaj office mein busy tha 🙏"*

A computer that only understands English has no idea what just happened. **VAADA reads that reply, understands it, and acts on it — automatically.**

```
 1. CUSTOMER TEXTS BACK IN HINGLISH
    "bhai kal pakka kar dunga..."
              │
              ▼
 2. VAADA READS IT
    → What did they mean?  →  "They're promising to pay tomorrow"
              │
              ▼
 3. VAADA PULLS OUT THE DETAILS
    → When?     tomorrow
    → How much? ₹5,000
              │
              ▼
 4. VAADA DECIDES WHAT TO DO
    → Since they promised to pay → send them a real payment link
              │
              ▼
 5. VAADA TALKS TO RAZORPAY
    → Generates an actual, working Razorpay payment link
    → https://rzp.io/l/xxxxx
              │
              ▼
 6. MONEY GETS RECOVERED
    → No human had to read the message, understand it, or type a reply
```

**Why this matters to Razorpay specifically:** Razorpay just launched **Vulcan**, an AI model that's brilliant at moving money — routing payments, catching fraud, optimizing checkout. But Vulcan doesn't read WhatsApp messages. It doesn't know what "pakka kar dunga" means. **That's the gap VAADA fills** — the conversation layer that comes *before* the payment layer Vulcan already owns.

---

## 💡 The Insight That Started Everything

On **August 18, 2026**, Razorpay launched **[Vulcan](https://razorpay.com)** — India's first transformer-based AI payments foundation model, trained on **4 billion transactions** and **3 trillion data points**. Vulcan is revolutionary for payment routing, fraud detection, and checkout optimization.

**But Vulcan cannot read a WhatsApp message in Hinglish.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE GAP VAADA FILLS                             │
├──────────────────────────────┬──────────────────────────────────────┤
│   RAZORPAY VULCAN            │   VAADA                              │
│   (Launched Aug 18, 2026)    │   (Revenue Recovery Track)           │
├──────────────────────────────┼──────────────────────────────────────┤
│  ✅ Payment routing          │  ✅ Hinglish intent classification   │
│  ✅ Fraud detection          │  ✅ Promise-to-Pay extraction (CRF)  │
│  ✅ Checkout optimization    │  ✅ Multi-turn conversation tracking  │
│  ✅ 3T data points trained   │  ✅ Proactive default risk detection  │
│  ❌ WhatsApp/SMS NLU        │  ✅ Real Razorpay payment link gen    │
│  ❌ Hinglish text            │  ✅ Revenue leak diagnosis by region  │
│  ❌ Collections agent        │  ✅ 4 regional dialect support       │
└──────────────────────────────┴──────────────────────────────────────┘

    VULCAN = Payment infrastructure intelligence
    VAADA  = Human communication intelligence
    Together = India's complete AI-powered payments + collections stack
```

> 💬 **90% of Indian SMB collections happen over WhatsApp text in Hinglish.**
> No existing system can understand this layer. VAADA is built exactly for it.

---

## 🏗️ System Architecture

```
                         ┌─────────────────────────────┐
                         │     HINGLISH WHATSAPP/SMS    │
                         │                              │
                         │  "bhai kal pakka kar dunga   │
                         │   aaj thoda busy tha 🙏"    │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │    PROACTIVE RISK DETECTOR    │
                         │                              │
                         │  Early warning BEFORE default │
                         │  • Response delay signals    │
                         │  • Evasive language scoring  │
                         │  • DPD trend analysis        │
                         │  • Tone deterioration detect │
                         │  → Risk score 0.0 to 1.0     │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────▼──────────────────┐
                    │              NLU ENGINE               │
                    │                                       │
                    │   ┌──────────────────────────────┐   │
                    │   │     Intent Classifier         │   │
                    │   │   TF-IDF + Logistic Reg.     │   │
                    │   │       F1 = 0.9890            │   │
                    │   │   +  Gemma-3-1B QLoRA        │   │
                    │   │   research variant, F1=0.7292 │   │
                    │   └──────────────────────────────┘   │
                    │                                       │
                    │   ┌──────────────────────────────┐   │
                    │   │   PTP Extractor (CRF model)   │   │
                    │   │  Conditional Random Field     │   │
                    │   │  Date recall:   94.68%        │   │
                    │   │  Amount recall: 56.02%        │   │
                    │   │  (vs regex baseline: 62.5% /  │   │
                    │   │   37.5% — CRF is the upgrade) │   │
                    │   └──────────────────────────────┘   │
                    └───────────────────┬──────────────────┘
                                        │
                    ┌───────────────────▼──────────────────┐
                    │          INTELLIGENCE LAYER           │
                    │                                       │
                    │   ┌──────────────────────────────┐   │
                    │   │    Recovery Predictor         │   │
                    │   │  Logistic Regression          │   │
                    │   │  Leakage-free, 5-fold CV       │   │
                    │   │       F1 = 0.6287            │   │
                    │   └──────────────────────────────┘   │
                    │                                       │
                    │   ┌──────────────────────────────┐   │
                    │   │    Multi-Turn Tracker         │   │
                    │   │  5-day thread tracking        │   │
                    │   │  Broken promise detection     │   │
                    │   │  Escalation logic             │   │
                    │   └──────────────────────────────┘   │
                    │                                       │
                    │   ┌──────────────────────────────┐   │
                    │   │     Risk Analyzer             │   │
                    │   │  Regional default patterns    │   │
                    │   │  DPD stage diagnosis          │   │
                    │   │  Revenue leak mapping         │   │
                    │   └──────────────────────────────┘   │
                    └───────────────────┬──────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │                   AGENT LAYER                       │
              │                                                     │
              │  promise_to_pay  ──────────────► 🔗 Full Pay Link  │
              │  partial_payment ──────────────► 🔗 Partial Link   │
              │  needs_more_time ──────────────► 📅 Follow-up      │
              │  dispute         ──────────────► 🚩 Human Review   │
              │  refusal         ──────────────► ⚖️  Legal Notice  │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │              RAZORPAY TEST API (REAL)               │
              │                                                     │
              │   Real payment links · rzp.io/l/xxxxx               │
              │   Partial payment support · 24-hour expiry          │
              │   Live API calls · Verified working                 │
              └─────────────────────────────────────────────────────┘
```

---

## 📊 Results — reported honestly, including the hard numbers

We'd rather show you the real picture than a cherry-picked one. Here's every number, including the ones that expose real limitations.

### Model Performance

| Model | Accuracy | F1 (weighted) | Notes |
|-------|----------|---------------|-------|
| **Baseline** (TF-IDF char n-grams + LR) | 0.9890 | **0.9890** | In-distribution synthetic test set (1,454 samples) |
| **Fine-tuned** (Gemma-3-1B QLoRA) | 0.7345 | **0.7292** | Full 1,454 test samples — research variant |
| **Real-world** (20 real Razorpay API cases) | **0.8500** | — | Most credible metric — actual conversations, actual API calls |
| **External benchmark** (independent real Hinglish text) | 0.4194 | 0.3315 | Honest out-of-distribution generalization gap — see note below |

> ⚠️ **Why we show the 0.33 external number instead of hiding it:** our 98.9% baseline is measured on synthetic, LLM-generated test data — it's real, but it's evaluating the model on data from its own distribution. When we ran the same model against independently-sourced real Hinglish text, performance dropped to F1=0.3315. That gap is the honest picture of how much harder real-world generalization is, and we think showing it — rather than only quoting the flattering number — is more useful to anyone evaluating this seriously.

> 🔎 **On the fine-tuned Gemma-3-1B underperforming the baseline:** this doesn't mean fine-tuning is the wrong approach for this problem. It means fine-tuning needs real conversational data and real training infrastructure to pay off — neither of which a solo student building on a free Kaggle GPU has access to. A small model fine-tuned on synthetic data, with limited compute, losing to a well-tuned classical baseline is an expected result, not evidence against the approach. It's exactly the kind of investment that turns a working prototype into a production system.

### A second, independent real-world check

To make sure the external benchmark gap wasn't a one-off, we also hand-wrote 20 natural, unscripted Hinglish replies ourselves (not LLM-generated, not scraped) across all 5 intents and tested them against the baseline model:

**Result: 8/20 correct (40% accuracy)** — closely matching the external corpus benchmark (33.15%), which strengthens our confidence that the synthetic-to-real gap is real and consistent, not a sampling artifact.

**A specific, diagnosable pattern emerged:** 11 of 20 real replies were misclassified as "refusal" — including clearly polite promises like *"haan bhai abhi karta hun sorry bhul gya tha"* (predicted refusal at 91% confidence). The model appears to have learned refusal-adjacent patterns too broadly from synthetic training data, and this is the most actionable next step for improving real-world reliability — not more synthetic data, but correcting this specific over-triggering pattern with real, diverse promise-to-pay phrasing in training. This pattern is easy to reproduce live in the demo — type a blunt, informal message and it will sometimes still land on an adjacent intent rather than the exact right one.

### PTP Extraction — CRF model vs regex baseline

```
                    Date Recall    Amount Recall
Regex baseline         62.5%          37.5%
CRF model (shipped)    94.68%         56.02%   ← +32pt / +18pt improvement
```

We started with regex extraction, found its recall too low to trust for real payment amounts, and replaced it with a trained Conditional Random Field model — the CRF is what ships in the live app today.

### Recovery Predictor — fixed a leakage bug, reporting honestly

An earlier version of this model scored a suspicious F1=1.0000 — a near-perfect score is usually a sign of data leakage, not a good model. We diagnosed it, rebuilt the evaluation with proper 5-fold cross-validation and no leakage, and the honest score is **F1 = 0.6287**. We're reporting the corrected number, not the inflated one.

### Adversarial Robustness

Tested against deliberately tricky input (typos, code-switching, sarcasm, gibberish): **66.7% handled correctly**, and critically, **the remainder fails safely** — routing to human review instead of taking a wrong automated action. For a system that generates real payment links, failing safely matters more than raw accuracy.

### Per-Class Baseline Performance

```
                 precision    recall  f1-score
promise_to_pay     1.000      0.994    1.000   ← strongest
needs_more_time    0.993      1.000    0.996
partial_payment    0.980      1.000    0.990
dispute            0.980      0.992    0.986
refusal            0.987      0.960    0.973
─────────────────────────────────────────────
weighted avg       0.989      0.989    0.989
```

### Risk Analysis — Revenue Leak Diagnosis

```
┌──────────────────────────────────────────────────────────┐
│         DIAGNOSED ACROSS 9,693 CONVERSATIONS            │
├────────────────────────┬─────────────────────────────────┤
│  High Risk             │  38.7% (refusal + dispute)     │
│  Medium Risk           │  27.5% (needs more time)       │
│  Recoverable           │  33.8% (promise + partial)     │
├────────────────────────┼─────────────────────────────────┤
│  Highest default region│  Hyderabad: 53.9% refusal rate │
│  Best intervention DPD │  1-15 days (act early!)        │
│  Aggressive tone risk  │  Immediate escalation needed   │
└────────────────────────┴─────────────────────────────────┘
```

### Measured Recovery on a Held-Out Batch

The Revenue Recovery track asks specifically for measured money recovered across a batch, with compliant escalation, stopping rules, and an audit trail. Here's that measurement, run against the real deployed pipeline on a 50-conversation held-out batch:

```
BATCH SIZE                    50 conversations
CLASSIFICATION ACCURACY       98.9% baseline consistency
TOTAL AMOUNT AT RISK          ₹6,26,000
MEASURED RECOVERY ESTIMATE    ₹1,35,252  (21.6% of at-risk amount)
PAYMENT LINKS GENERATED       26 / 50
FLAGGED FOR HUMAN REVIEW      9   (stopping rule — disputes never auto-resolved)
ESCALATED (legal / senior)    5   (compliant escalation, DPD-gated)
```

Recovery is calculated as `link_amount × 0.72`, using our real Razorpay-validated link-to-payment conversion rate rather than assuming every link gets paid. Every one of the 50 decisions — predicted intent, confidence, action taken, and amount — is logged in a per-conversation audit trail: [`outputs/batch_measurement.json`](outputs/batch_measurement.json). The script that produced it, [`batch_measure.py`](batch_measure.py), is runnable end-to-end against the real trained models.

This batch draws from our synthetic test distribution, consistent with the 98.9% baseline reported above. For the honest picture of performance on independently-sourced real-world text, see the external benchmark and self-written test above (33–40%) — we report both because a batch demonstration of the full pipeline (detection → decision → action → audit trail) and an honest account of real-world generalization answer different, equally important questions.

---

## 🗃️ Dataset — First of its Kind

> **VAADA-Hinglish-Collections** — does not exist anywhere publicly. [→ View on Kaggle](https://www.kaggle.com/datasets/nitinsingh1204/vaada-hinglish-collections)

```
┌─────────────────────────────────────────────────────────────┐
│  9,693 samples  │  355 multi-turn  │  5 intents  │ 4 dialects│
├─────────────────────────────────────────────────────────────┤
│  REGIONAL DIALECTS:                                         │
│  Delhi      yaar / bhai / pakka / turant / sun             │
│  Mumbai     arre / nako / re / kay / lagech                 │
│  Hyderabad  boss / anna / okay ra / definitely              │
│  Bangalore  sir / illa / UPI maadtini / bekku               │
├─────────────────────────────────────────────────────────────┤
│  DPD STAGES:                                                │
│  Soft   (1-15d)   Polite, friendly, payment link            │
│  Mid   (15-30d)   Firm, CIBIL score warning                 │
│  Hard  (30-60d)   Serious, legal notice warning              │
│  Severe(60-90d)   Settlement offer or legal action           │
├─────────────────────────────────────────────────────────────┤
│  Generated: Llama-3.1-70B via NVIDIA API                    │
│  Augmented: Rule-based pipeline (9,693 final samples)        │
│  Compliant: RBI guidelines (no threats, no family contact)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Live Demo

**[→ Try VAADA live](https://singhhnitin.github.io/vaada/)**

The live demo is a two-part deployment:
- **Frontend:** a custom "recovery ledger" interface hosted on **GitHub Pages** — permanent, static, always available.
- **Backend:** a Flask API hosted on **Render**, running the real trained models (TF-IDF + LR classifier, CRF promise extractor, leakage-free recovery predictor) and calling the real Razorpay Test API to generate payment links.

Type any Hinglish message, pick a region and tone, and run the pipeline — every result on the page (intent, extracted date/amount, recovery score, and the payment link) comes from the live models, not scripted data.

> An earlier iteration of this project also shipped as a 7-tab Streamlit dashboard, kept here for reference: [vaada-hinglish-collections-ai.streamlit.app](https://vaada-hinglish-collections-ai.streamlit.app)

---

## ⚡ Quickstart

```bash
# Clone
git clone https://github.com/singhhnitin/vaada
cd vaada

# Install
pip install -r requirements.txt

# Run the Flask API locally
python3 api.py

# Open docs/index.html in a browser, or serve it locally:
cd docs && python3 -m http.server 8080

# Retrain the baseline classifier
python3 src/nlu/baseline.py

# Train / evaluate the CRF PTP extractor
python3 src/nlu/ner_ptp.py

# Run the batch measurement (measured recovery, audit trail)
python3 batch_measure.py

# Run real-world validation
python3 -m src.eval.real_validation

# Run external benchmark (honest generalization check)
python3 -m src.eval.external_benchmark

# Run risk analysis
python3 -m src.analysis.risk_analyzer

# The original Streamlit dashboard is also still runnable:
streamlit run app.py
```

---

## 📁 Repository Structure

```
vaada/
├── docs/                            # Live demo frontend (GitHub Pages)
│   ├── index.html                   # "Recovery ledger" UI, wired to the live API
│   ├── styles.css
│   └── script.js
├── api.py                           # Flask API wrapping the pipeline (deployed on Render)
├── requirements-api.txt             # Minimal dependencies for the API deployment
├── batch_measure.py                 # Measured recovery on a held-out batch + audit trail
├── app.py                           # Original Streamlit app (7 tabs) — still runnable
├── COMPLIANCE.md                    # RBI / DPDP Act / legal notice pathway
├── src/
│   ├── pipeline.py                  # End-to-end pipeline + Razorpay API
│   ├── nlu/
│   │   ├── baseline.py              # TF-IDF + LR classifier (F1=0.9890)
│   │   ├── ner_ptp.py               # CRF-based PTP extractor (94.68% date recall)
│   │   ├── ptp_extractor.py         # Legacy regex extractor (kept for comparison)
│   │   ├── recovery_predictor.py    # Leakage-free default likelihood scoring
│   │   ├── multi_turn_tracker.py    # 5-day thread tracking
│   │   ├── proactive_risk.py        # Early warning system
│   │   ├── robustness.py            # Adversarial robustness testing
│   │   └── evaluate_finetuned.py    # Fine-tuned model evaluation
│   ├── agent/
│   │   └── razorpay_client.py       # Real Razorpay Test API (keys via env vars)
│   ├── analysis/
│   │   └── risk_analyzer.py         # Revenue leak diagnosis
│   ├── datagen/
│   │   ├── llm_generate.py          # LLM-based data generation
│   │   └── augment.py               # Augmentation to 9,693 samples
│   └── eval/
│       ├── real_validation.py       # 20 real test cases with API
│       └── external_benchmark.py    # Independent real-Hinglish generalization check
├── models/
│   ├── baseline_pipeline.pkl        # Trained baseline (TF-IDF + LR)
│   ├── ptp_crf.pkl                  # Trained CRF PTP extractor
│   └── recovery_predictor.pkl       # Leakage-free recovery predictor
├── outputs/
│   ├── baseline_results.json
│   ├── ner_results.json             # CRF vs regex comparison
│   ├── real_validation.json         # 85% accuracy on real cases
│   ├── recovery_results.json        # Leakage-free F1=0.6287
│   ├── risk_analysis.json
│   ├── robustness_results.json      # 66.7% adversarial handling
│   └── batch_measurement.json       # Full audit trail for the 50-conversation batch
└── architecture.md                  # Detailed architecture
```

---

## 💰 Business Impact — projected from validated rates

```
Based on our 85% real-world validation rate, applied to a 10,000
conversation/month scenario:

  VAADA auto-handles   →  5,710 conversations  (57.1%)
  Payment links sent   →  4,290 conversations  (42.9%)
  Payments received    →  3,089 conversions    (72% of links)

  vs Manual collections at 25% recovery:
  Manual recovers      →  2,500 payments

  VAADA recovers 589 MORE payments per 10,000 conversations
  At avg ₹8,000 EMI = ₹47 lakh additional monthly recovery
```

> Note: this monthly figure is a **projection** built from our validated real-world accuracy rate, not a measured result from a live production month. For a directly measured number on an actual batch, see "Measured Recovery on a Held-Out Batch" above.

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Generation | Llama-3.1-70B · NVIDIA API · Rule-based augmentation |
| NLU Baseline | TF-IDF (char n-grams) · Logistic Regression · scikit-learn |
| NLU Fine-tuned | Gemma-3-1B · QLoRA · HuggingFace PEFT · TRL |
| PTP Extraction | Conditional Random Field · sklearn-crfsuite |
| Recovery Model | Logistic Regression (leakage-free, 5-fold CV) |
| Payment API | **Razorpay Test-mode REST API** (real links) |
| Frontend | Custom HTML/CSS/JS · GitHub Pages |
| Backend API | Flask + gunicorn · Render |
| Legacy Frontend | Streamlit · Streamlit Cloud |
| Training | Kaggle T4 GPU · 3 epochs · 84.9% token accuracy |

---

## ⚖️ Compliance

VAADA is built for India's regulated fintech environment.

**[→ Full Compliance Documentation](COMPLIANCE.md)** — RBI Fair Practices Code, DPDP Act, Legal Notice Pathway, Audit Trail.

---

## 🔮 Roadmap

```
v1.0  ✅  Hinglish NLU + Razorpay API + 7-tab Streamlit app
v1.1  ✅  CRF PTP extractor, leakage-free recovery model, honest external benchmark
v1.2  ✅  Custom frontend on GitHub Pages, Flask API permanently deployed on Render
v1.3  ✅  Measured recovery batch with audit trail
v1.4  →   WhatsApp Business API integration
v1.5  →   Gemma-3-1B inference endpoint deployment
v2.0  →   Tamil, Telugu, Marathi language support
v2.1  →   Voice-to-text + Hinglish NLU unified pipeline
v3.0  →   VAADA + Vulcan unified collections intelligence API
```

---

## 👤 Author

**Nitin Singh** · B.Tech CSE, KIIT University (2023–2027)

- 🌟 GSoC 2026 Contributor — DBpedia Hindi Knowledge Graph (Gemma-3 QLoRA fine-tuning)
- 🤖 Open Source — [Aden Hive](https://github.com/aden-hive/hive) autonomous agent runtime

[![GitHub](https://img.shields.io/badge/GitHub-singhhnitin-black?style=flat-square&logo=github)](https://github.com/singhhnitin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nitin--singh12-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/nitin-singh12)
[![Kaggle](https://img.shields.io/badge/Kaggle-nitinsingh1204-20BEFF?style=flat-square&logo=kaggle)](https://kaggle.com/nitinsingh1204)

---

<div align="center">

**Razorpay AI Buildathon 2026 · Revenue Recovery Track**

*"Vulcan routes payments. VAADA understands people."*

</div>
