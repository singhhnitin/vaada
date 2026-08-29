<div align="center">

# VAADA — Vernacular Agentic AI for Debt & Arrears

### The missing communication intelligence layer for Razorpay Vulcan

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-00ff41?style=for-the-badge)](https://vaada-hinglish-collections-ai.streamlit.app)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge)](https://www.kaggle.com/datasets/nitinsingh1204/vaada-hinglish-collections)
[![Track](https://img.shields.io/badge/Track-Revenue_Recovery-orange?style=for-the-badge)](https://razorpay.com/buildathon)

</div>

---

## The Insight

On **August 18, 2026**, Razorpay launched **Vulcan** — India's first AI payments foundation model trained on 4 billion transactions. Vulcan handles payment routing, fraud detection, checkout optimization.

**But Vulcan cannot read a WhatsApp message in Hinglish.**

RAZORPAY VULCAN VAADA
Payment routing --> Hinglish intent classification
Fraud detection --> Promise-to-Pay extraction
Checkout optim. --> Multi-turn conversation tracking
[missing] --> Proactive default risk detection
[missing] --> Real Razorpay payment link gen
[missing] --> Revenue leak diagnosis

VULCAN = Payment infrastructure intelligence
VAADA = Human communication intelligence
Together = Complete AI-powered collections stack


> 90% of Indian SMB collections happen over WhatsApp text in Hinglish.
> No existing system understands this. VAADA does.

---

## Results

Model Accuracy F1 (weighted)
Baseline (TF-IDF + LR) 0.9890 0.9890
Fine-tuned (Gemma-3-1B) 0.6900 0.6318
Real-world (20 API cases) 0.8500 --
Training Token Accuracy Epoch 3: 84.9%


---

## Live Demo

Try VAADA: https://vaada-hinglish-collections-ai.streamlit.app

- LIVE DEMO: Paste Hinglish message, get real Razorpay link
- EVAL RESULTS: Full metrics and confusion matrix
- BUSINESS IMPACT: Rs recovered simulation
- WHATSAPP SIM: Simulate multi-day collections thread
- RISK ANALYSIS: Diagnose revenue leaks by region and DPD

---

## Dataset

First regionalized Hinglish fintech NLU dataset:
- 9,693 samples, 355 multi-turn conversations
- 5 intent classes, 4 regional dialects
- DPD stages 1-90, RBI-compliant patterns
- Generated via Llama-3.1-70B + NVIDIA API
- Public prior art: NONE

---

## Architecture

Hinglish WhatsApp/SMS
|
Proactive Risk Detector (early warning signals)
|
NLU Engine

Intent Classifier (TF-IDF+LR F1=0.9890 / Gemma-3-1B 84.9%)
PTP Extractor (date + amount)
|
Intelligence Layer
Recovery Predictor (GradientBoosting)
Multi-Turn Tracker (5-day thread)
Risk Analyzer (regional patterns)
|
Agent Layer
promise_to_pay --> Real Razorpay Full Link
partial_payment --> Real Razorpay Partial Link
needs_more_time --> Schedule Followup
dispute --> Human Review
refusal --> Legal Notice
|
Razorpay Test API (real payment links)

---

## Quickstart

```bash
git clone https://github.com/singhhnitin/vaada
cd vaada
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

- Data: Llama-3.1-70B via NVIDIA API + augmentation
- NLU: TF-IDF + LR baseline / Gemma-3-1B QLoRA fine-tuned
- Recovery: GradientBoostingClassifier
- Payment: Razorpay Test-mode REST API
- Frontend: Streamlit (Streamlit Cloud)
- Training: Kaggle T4 GPU

---

## Author

Nitin Singh | GSoC 2026 Contributor | KIIT University
GitHub: singhhnitin | Kaggle: nitinsingh1204

**Razorpay AI Buildathon 2026 - Revenue Recovery Track**

*VAADA: Because Vulcan routes payments. VAADA understands people.*
