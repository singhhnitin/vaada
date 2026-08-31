# VAADA — System Architecture

## The Gap VAADA Fills

```
RAZORPAY TODAY              VAADA ADDS
─────────────────           ────────────────────────────────
Voice (ElevenLabs)    ✅     Text NLU (WhatsApp/SMS)      ✅ NEW
Sarvam AI Voice        ✅     Intent Classification        ✅ NEW
Agent Studio           ✅     Promise-to-Pay Extractor (CRF) ✅ NEW
                              Recovery Predictor            ✅ NEW
                              Multi-Turn Tracker            ✅ NEW
                              Real Payment Link Gen         ✅ NEW
                              Default Pattern Analysis      ✅ NEW
                              Adversarial Robustness Layer  ✅ NEW
```

## Full Pipeline

```
                ┌──────────────────────────────────┐
                │     HINGLISH WHATSAPP / SMS       │
                │  "bhai kal pakka kar dunga 🙏"   │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼───────────────────┐
                │         NLU ENGINE                │
                │  ┌────────────────────────────┐  │
                │  │   Intent Classifier         │  │
                │  │   TF-IDF (char n-grams)     │  │
                │  │   + Logistic Regression     │  │
                │  │   F1 = 0.9890 (synthetic)    │  │
                │  │   + Gemma-3-1B QLoRA         │  │
                │  │   F1 = 0.7292 (research)     │  │
                │  └────────────────────────────┘  │
                │  ┌────────────────────────────┐  │
                │  │   PTP Extractor              │  │
                │  │   Conditional Random Field   │  │
                │  │   Date recall   = 94.68%     │  │
                │  │   Amount recall = 56.02%     │  │
                │  │   (regex baseline: 62.5% /    │  │
                │  │    37.5% — CRF replaced it)   │  │
                │  └────────────────────────────┘  │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼───────────────────┐
                │      INTELLIGENCE LAYER           │
                │  ┌────────────────────────────┐  │
                │  │   Recovery Predictor        │  │
                │  │   Logistic Regression       │  │
                │  │   Leakage-free, 5-fold CV   │  │
                │  │   F1 = 0.6287               │  │
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
                │  rzp.io/l/xxxxx                   │
                │  Partial payment support          │
                │  24-hour expiry                   │
                └──────────────────────────────────┘
```

## Dataset — First of its Kind

```
┌─────────────────────────────────────────────────────────┐
│ VAADA-HINGLISH-COLLECTIONS                               │
│                                                           │
│ 9,693 samples │ 355 multi-turn │ 5 intents │ 4 dialects  │
│                                                           │
│ Delhi      ████████████ 2,281                            │
│ Mumbai     ████████████ 2,298                            │
│ Hyderabad  ███████████  2,350                            │
│ Bangalore  ███████████  2,324                            │
│                                                           │
│ DPD stages: Soft(1-15) Mid(15-30) Hard(30-60) Severe     │
│ RBI-compliant · No threats · No family contact           │
│ Generated via Llama-3.1-70B + NVIDIA API                 │
└─────────────────────────────────────────────────────────┘
```

## Results — reported honestly

```
┌──────────────────────────────────────────────────────┐
│ EVAL METRICS                                          │
│ (held-out test set n=1454, unless noted)              │
│                                                        │
│ Intent Classification (synthetic)   F1 = 0.9890        │
│ Intent Classification (external,    F1 = 0.3315        │
│   independent real Hinglish text)                     │
│ Real-world (20 Razorpay API cases)  Acc = 85%          │
│ PTP Date Extraction Recall (CRF)     = 94.68%          │
│ PTP Amount Extraction Recall (CRF)   = 56.02%          │
│ Recovery Prediction (leakage-free,   F1 = 0.6287        │
│   5-fold CV)                                           │
│ Adversarial Robustness                = 66.7%           │
│   (remainder fails safely to human review)             │
│ Payment Links Generated               = 42.9%           │
│ High-Risk Detection                   = 38.7%           │
│                                                        │
│ vs Manual Collections (25% recovery rate)              │
│ VAADA: 30.9% recovery rate = +23.6% uplift (projected) │
└──────────────────────────────────────────────────────┘
```

> **Why the external benchmark (0.33) is shown here, not hidden:** the synthetic
> 0.9890 score is measured on data drawn from the same LLM-generation process as
> training data — it proves the model learned that distribution well, not that it
> generalizes. Testing against independently-sourced real Hinglish text is a much
> harder, more honest test, and the gap between the two numbers is the real
> generalization story. We also caught and fixed a data-leakage bug in the recovery
> predictor, which originally scored a suspicious F1=1.0000 — the 0.6287 figure
> above is the corrected, leakage-free result.

## Tech Stack

```
Data Generation : Llama-3.1-70B (NVIDIA API) + rule-based augmentation
NLU Model       : TF-IDF (char n-grams) + LR (baseline) | Gemma-3-1B QLoRA (research)
PTP Extraction  : Conditional Random Field (sklearn-crfsuite)
Recovery Model  : Logistic Regression, leakage-free 5-fold CV
Framework       : Python · scikit-learn · HuggingFace · PEFT · TRL
Payment API     : Razorpay Test-mode REST API (credentials via environment variables)
Frontend        : Streamlit (deployed on Streamlit Cloud)
Dataset         : kaggle.com/nitinsingh1204/vaada-hinglish-collections
Repo            : github.com/singhhnitin/vaada
Live Demo       : vaada-hinglish-collections-ai.streamlit.app
Compliance      : RBI Fair Practices Code, DPDP Act — see COMPLIANCE.md
```
