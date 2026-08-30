# VAADA — Compliance & Legal Framework

## Overview

VAADA is designed for deployment in India's regulated fintech collections environment. This document outlines compliance with RBI guidelines, DPDP Act requirements, and the legal escalation pathway.

---

## 1. RBI Fair Practices Code Compliance

The Reserve Bank of India's Fair Practices Code for NBFCs and Banks governs how lenders can communicate with borrowers. VAADA is built with these constraints hardcoded at the data generation level.

### What VAADA enforces:

```
✅ No threats of violence or physical harm
✅ No contact with family members, friends, or colleagues
✅ No calls or messages outside 8 AM – 7 PM window
✅ No use of intimidating language
✅ No misleading information about legal consequences
✅ Polite-firm tone enforced across all agent messages
✅ Human escalation path for all disputes
✅ Settlement offer before legal notice at severe DPD
```

### How it's implemented:

**At data generation:** The LLM prompt explicitly instructs: *"Agent messages must be RBI-compliant: no threats, no family contact, no harassment — polite but firm."* This bakes compliance into the training data itself.

**At inference:** The agent workflow enforces:
- `dispute` → FLAG_FOR_HUMAN_REVIEW (never auto-escalate a dispute)
- `refusal` at DPD ≤ 60 → ESCALATE_TO_SENIOR (human review before legal)
- `refusal` at DPD > 60 → TRIGGER_LEGAL_NOTICE (only after senior escalation)
- Settlement offer always sent before legal notice at severe DPD

**At robustness layer:** Low-confidence predictions and ambiguous inputs route to human agents, never auto-classified.

---

## 2. DPDP Act Compliance (Digital Personal Data Protection Act 2023)

India's DPDP Act governs how personal data of Indian residents is collected, processed, and stored.

### VAADA's data handling:

```
DATA MINIMIZATION
  VAADA processes: reminder text, customer reply text
  VAADA does NOT store: phone numbers, account numbers, Aadhaar, PAN
  Session-only: all conversation data cleared after pipeline run

CONSENT
  Lawful basis: customer initiated contact with the lending institution
  Implied consent: debtor responding to a legitimate payment reminder
  No third-party data sharing

DATA SUBJECT RIGHTS
  Right to erasure: no persistent storage of conversation data
  Right to access: conversation logs available to the lending institution
  Right to correction: dispute pathway routes to human for correction

CROSS-BORDER
  All processing on-premise or Indian cloud infrastructure
  No data leaves Indian jurisdiction
```

### What VAADA does NOT do:

```
❌ Does not store customer PII beyond the active session
❌ Does not share data with third parties
❌ Does not use customer data for model training without consent
❌ Does not process data of minors
❌ Does not infer sensitive personal characteristics
```

---

## 3. Legal Notice Pathway

The most scrutinized part of any collections system is the escalation to legal action. VAADA has a deliberate, multi-step pathway.

```
STEP 1: Standard collections (DPD 1-30)
  → Polite reminder + payment link
  → VAADA handles automatically

STEP 2: Firm follow-up (DPD 30-60)
  → CIBIL warning + settlement offer
  → VAADA + human oversight

STEP 3: Senior escalation (DPD 60+ with refusal)
  → ESCALATE_TO_SENIOR_TEAM
  → Human agent contacts customer
  → Settlement negotiation attempted

STEP 4: Legal notice (DPD 60+ after senior escalation fails)
  → TRIGGER_LEGAL_NOTICE
  → Routes to qualified legal team (NOT automated)
  → Legal team sends notice per SARFAESI / applicable law
  → Audit trail maintained throughout

VAADA NEVER sends an automated legal notice.
Legal action always requires human authorization.
```

---

## 4. Audit Trail

Every VAADA decision is logged with:

```json
{
  "timestamp":     "2026-08-29T10:30:00",
  "conversation_id": "VAADA-XXXX",
  "intent_detected": "refusal",
  "confidence":    0.94,
  "dpd":           67,
  "action_taken":  "ESCALATE_TO_SENIOR_TEAM",
  "human_review":  true,
  "escalation_reason": "refusal at severe DPD"
}
```

This audit trail enables:
- Regulatory inspection readiness
- Dispute resolution evidence
- Model performance monitoring
- Bias detection across customer segments

---

## 5. Bias and Fairness

VAADA's training data covers 4 regional dialects equally. However:

```
KNOWN LIMITATIONS:
- Hyderabad customers show 53.9% refusal rate in training data
  This reflects synthetic data distribution, not real default rates
- Regional dialect detection affects confidence scores
- Non-standard Hinglish may route to human more frequently

MITIGATION:
- Adversarial robustness layer routes low-confidence to human
- No protected characteristic (caste, religion, gender) used as feature
- Regular bias audits recommended on production deployment
```

---

## 6. Responsible Deployment Checklist

Before production deployment, the following must be completed:

```
□ Legal review of all agent message templates
□ NBFC/Bank compliance team sign-off
□ DPO (Data Protection Officer) registration under DPDP Act
□ Privacy Policy updated to include VAADA processing
□ Staff training on human escalation workflows
□ Regular model performance audits (monthly)
□ Real customer data collected and model retrained
□ RBI inspection readiness documentation
```

---

*VAADA is a research prototype. Production deployment requires full legal and compliance review by qualified professionals.*
