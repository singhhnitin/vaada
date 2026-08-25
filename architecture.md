markdown
# VAADA — System Architecture

## The Gap VAADA Fills

RAZORPAY TODAY VAADA ADDS
───────────────── ────────────────────────────────
Voice (ElevenLabs) ✅ Text NLU (WhatsApp/SMS) ✅ NEW
Sarvam AI Voice ✅ Intent Classification ✅ NEW
Agent Studio ✅ Promise-to-Pay Extractor ✅ NEW
Recovery Predictor ✅ NEW
Multi-Turn Tracker ✅ NEW
Real Payment Link Gen ✅ NEW
Default Pattern Analysis ✅ NEW


## Full Pipeline
                ┌──────────────────────────────────┐
                │     HINGLISH WHATSAPP / SMS       │
                │  "bhai kal pakka kar dunga 🙏"   │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼───────────────────┐
                │         NLU ENGINE                │
                │  ┌────────────────────────────┐  │
                │  │   Intent Classifier         │  │
                │  │   TF-IDF + Logistic Reg     │  │
                │  │   Gemma-3-1B Fine-tuned     │  │
                │  │   F1 = 0.9890               │  │
                │  └────────────────────────────┘  │
                │  ┌────────────────────────────┐  │
                │  │   PTP Extractor             │  │
                │  │   Rule-based + Regex        │  │
                │  │   Date recall = 62.5%       │  │
                │  └────────────────────────────┘  │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼───────────────────┐
                │      INTELLIGENCE LAYER           │
                │  ┌────────────────────────────┐  │
                │  │   Recovery Predictor        │  │
                │  │   GradientBoosting          │  │
                │  │   F1 = 1.0000               │  │
                │  └────────────────────────────┘  │
                │  ┌────────────────────────────┐  │
                │  │   Multi-Turn Tracker        │  │
                │  │   5-day thread tracking     │  │
                │  │   Broken promise detection  │  │
                │  └────────────────────────────┘  │
                │  ┌────────────────────────────┐  │
                │  │   Risk Analyzer             │  │
                │  │   Default pattern diagnosis │  │
                │  │   Regional risk scoring     │  │
                │  └────────────────────────────┘  │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼───────────────────┐
                │        AGENT LAYER                │
                │                                   │
                │  promise_to_pay  ──► 🔗 Razorpay │
                │  partial_payment ──► 🔗 Partial  │
                │  needs_more_time ──► 📅 Schedule │
                │  dispute         ──► 🚩 Human    │
                │  refusal         ──► ⚖ Legal    │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼───────────────────┐
                │      RAZORPAY API (REAL)          │
                │  Test-mode payment links          │
                │  rzp.io/rzp/xxxxx                 │
                │  Partial payment support          │
                │  24-hour expiry                   │
                └──────────────────────────────────┘

## Dataset — First of its Kind

┌─────────────────────────────────────────────────────────┐
│ VAADA-HINGLISH-COLLECTIONS │
│ │
│ 9,693 samples │ 355 multi-turn │ 4 dialects │
│ │
│ Delhi ████████████ 2,281 │
│ Mumbai ████████████ 2,298 │
│ Hyderabad███████████ 2,350 │
│ Bangalore███████████ 2,324 │
│ │
│ DPD stages: Soft(1-15) Mid(15-30) Hard(30-60) Severe │
│ RBI-compliant · No threats · No family contact │
│ Generated via Llama-3.1-70B + NVIDIA API │
└─────────────────────────────────────────────────────────┘


## Results

┌──────────────────────────────────────────────────────┐
│ EVAL METRICS │
│ (held-out test set n=1454) │
│ │
│ Intent Classification F1 = 0.9890 │
│ PTP Date Extraction Recall = 0.6246 │
│ PTP Amount Extraction Recall = 0.3754 │
│ Recovery Prediction F1 = 1.0000 │
│ Payment Links Generated 42.9% │
│ High-Risk Detection 38.7% │
│ │
│ vs Manual Collections (25% recovery rate) │
│ VAADA: 30.9% recovery rate = +23.6% uplift │
└──────────────────────────────────────────────────────┘


## Tech Stack

Data Generation : Llama-3.1-70B (NVIDIA API) + Augmentation
NLU Model : TF-IDF + LR (baseline) | Gemma-3-1B QLoRA (fine-tuned)
Recovery Model : GradientBoostingClassifier
Framework : Python · scikit-learn · HuggingFace · PEFT · TRL
Payment API : Razorpay Test-mode REST API
Frontend : Streamlit (deployed on Streamlit Cloud)
Dataset : kaggle.com/nitinsingh1204/vaada-hinglish-collections
Repo : github.com/singhhnitin/vaada
Live Demo : vaada-hinglish-collections-ai.streamlit.app
