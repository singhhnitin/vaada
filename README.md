# Hindi Relational Triple Extraction — GSoC 2026

**DBpedia Hindi Chapter 2026: Fine-Tuning Indic Models for Hindi Relational Triple Extraction + Human-in-the-Loop Feedback**

> GSoC 2026 Warm-up Project | Nitin Singh | KIIT University  
> Mentor: [@tiwarisanju18](https://github.com/tiwarisanju18) | Organisation: DBpedia Association

---

## Overview

This repository contains warm-up experiments for the GSoC 2026 DBpedia Hindi Chapter project. The goal is to extract structured relational triples from free-text Hindi sentences and align them to the DBpedia ontology.

The core finding: **zero-shot LLMs perfectly identify Hindi entities but completely fail at predicate normalization.** This repo demonstrates the problem, measures it, and builds the first layer of the solution.

---

## Pipeline

```
Hindi Sentence
      ↓
Zero-shot Gemma-3 (Baseline)
      ↓
Error Taxonomy Analysis
      ↓
Ontology Alignment Layer  ──→  Low Confidence  ──→  HITL Review
      ↓                                                    ↓
High Confidence Triple                          Corrected Triple
      ↓                                                    ↓
      └──────────────────────────────────────────────────→↓
                                              JSONL Feedback Data
                                              (Ready for Fine-tuning)
```

---

## Results

### Baseline — Zero-shot Gemma-3-1b-it on Hindi BenchIE

| Metric | Score | Interpretation |
|---|---|---|
| Subject Accuracy | 5/5 = **100%** | Entity identification is strong |
| Predicate Accuracy | 0/5 = **0%** | Complete failure at ontology alignment |
| Object Accuracy | 5/5 = **100%** | Entity identification is strong |
| Full Triple Match | 0/5 = **0%** | No complete triple correct |

### Error Taxonomy — 3 Distinct Failure Modes

| Error Type | Count | Example |
|---|---|---|
| Predicate Normalization Failure | 2/5 (40%) | Extracted `का निर्माण` instead of `dbo:builder` |
| Language Mixing | 2/5 (40%) | Extracted `was born in` (English) instead of `dbo:birthPlace` |
| Implicit Relation Error | 1/5 (20%) | Extracted `है` (copula) instead of `dbo:capital` |

> **Language Mixing** is a previously undocumented failure mode — the model switches to English predicates for Hindi input, making DBpedia ontology alignment impossible downstream.

### Ontology Alignment Layer — 0% → 80% Predicate Accuracy

| Stage | Predicate Accuracy | Notes |
|---|---|---|
| Zero-shot Gemma-3 alone | 0/5 = **0%** | Raw surface verb phrases |
| + Ontology Alignment Layer | 4/5 = **80%** | MiniLM cosine similarity against DBpedia properties |
| Remaining failure (`है`) | 1/5 = **20%** | High confidence (0.691) but wrong — flagged for HITL |

The alignment layer uses `paraphrase-multilingual-MiniLM-L12-v2` to map extracted Hindi predicates to DBpedia ontology properties via cosine similarity. Triples below the confidence threshold are automatically routed to human review rather than silently entering the knowledge graph.

---

## Repository Structure

```
hindi-triple-extraction-gsoc2026/
│
├── notebooks/
│   ├── 01_baseline_gemma3.ipynb        # Zero-shot Gemma-3 baseline + evaluation
│   ├── 02_error_taxonomy.ipynb         # 3-type error classification
│   ├── 03_ontology_alignment.ipynb     # MiniLM alignment layer (0% → 80%)
│   └── 04_hitl_prototype.ipynb         # Streamlit HITL interface demo
│
├── src/
│   ├── extractor.py                    # Triple extraction with Gemma-3
│   ├── ontology_aligner.py             # DBpedia property alignment layer
│   ├── error_taxonomy.py               # Error classification utilities
│   └── hitl_app.py                     # Streamlit HITL feedback interface
│
├── data/
│   ├── test_sentences.json             # Hindi BenchIE test sentences
│   ├── gold_annotations.json           # Gold DBpedia triple annotations
│   └── feedback_output.jsonl           # Sample HITL corrected output
│
├── results/
│   ├── baseline_results.json           # Zero-shot evaluation results
│   └── alignment_results.json          # Ontology alignment evaluation
│
└── README.md
```

---

## Notebooks

### 01 — Zero-shot Baseline
Runs `google/gemma-3-1b-it` in zero-shot mode on Hindi BenchIE sentences. Evaluates subject, predicate, and object accuracy separately against gold DBpedia annotations.

### 02 — Error Taxonomy
Categorises every predicate failure into one of three types: Predicate Normalization Failure, Language Mixing, or Implicit Relation Error. Provides structured analysis of each failure mode.

### 03 — Ontology Alignment Layer
Builds a multilingual embedding-based alignment layer using `paraphrase-multilingual-MiniLM-L12-v2`. Maps extracted Hindi predicates to DBpedia ontology properties via cosine similarity. Implements confidence-based flagging for human review routing.

### 04 — HITL Prototype
Streamlit-based Human-in-the-Loop annotation interface. Reviewers can Accept, Reject, or Edit each extracted triple. Outputs structured JSONL corrections ready for fine-tuning retraining.

---

## HITL Interface

The feedback interface captures structured corrections from human reviewers:

| Field | Description |
|---|---|
| Sentence | Original Hindi source sentence |
| Extracted Triple | Model output (subject, predicate, object) |
| Decision | Accept / Reject / Edit |
| Error Type | From the 3-type taxonomy above |
| Corrected Predicate | Reviewer-supplied DBpedia property |
| Reviewer ID | For inter-annotator agreement tracking |
| Timestamp | For dataset versioning |

Output format: JSONL, one record per reviewed triple, ready for direct retraining.

**Sample output:**
```json
{"sentence": "ताजमहल का निर्माण शाहजहाँ ने करवाया था।", "original": {"subject": "ताजमहल", "predicate": "का निर्माण", "object": "शाहजहाँ"}, "corrected": {"subject": "ताजमहल", "predicate": "dbo:builder", "object": "शाहजहाँ"}, "decision": "Edit", "error_type": "Predicate Normalization Failure", "reviewer": "Nitin"}
```

---

## Key Findings

1. **Predicate normalization is the single largest gap** in zero-shot Hindi triple extraction. Subjects and objects are identified perfectly; predicates fail completely.

2. **Language Mixing is a previously undocumented failure mode.** The model switches to English predicates for Hindi input (`was born in` instead of `dbo:birthPlace`), making ontology alignment impossible without a dedicated alignment layer.

3. **The ontology alignment layer recovers 80% of predicate failures** using multilingual sentence embeddings. The remaining 20% (implicit copula constructions like `X की राजधानी Y है`) require special handling — neither fine-tuning nor embedding similarity alone resolves them reliably.

4. **Confidence-based routing is essential.** The one remaining failure scored high confidence (0.691) but mapped to the wrong property. Without flagging, it would silently pollute the knowledge graph. The HITL interface is designed specifically to catch these high-confidence-but-wrong cases.

---

## Approach for GSoC 2026 Coding Period

The warm-up experiments directly validate the project architecture:

- **Fine-tune Gemma-3** with LoRA on Hindi BenchIE gold annotations, using iterative slot prompting (predicate generated after subject/object context is established)
- **Extend the ontology alignment layer** to full DBpedia property coverage with Hindi-specific embeddings
- **Harden the HITL interface** to production quality with confidence-based priority queuing and JSONL retraining pipeline
- **Evaluate** using full Hindi-BenchIE protocol (essential + compensatory triple matching, not exact string match)

---

## Setup

```bash
git clone https://github.com/singhhnitin/hindi-triple-extraction-gsoc2026
cd hindi-triple-extraction-gsoc2026
pip install -r requirements.txt
```

**Requirements:**
```
transformers>=4.40.0
accelerate
bitsandbytes
sentence-transformers
streamlit
torch
scikit-learn
```

Run the HITL interface:
```bash
streamlit run src/hitl_app.py
```

---

## References

- Gashteovski et al. (2022). *BenchIE: A Framework for Multi-Faceted Evaluation of Binary Open Information Extraction.* ACL 2022.
- Kotnis et al. (2022). *MILIE: Modular & Iterative Multilingual Open Information Extraction.* ACL 2022.
- Pai et al. (2024). *A Survey on Open Information Extraction from Text.* EMNLP Findings 2024.
- Kothari et al. *IndIE: A Platform-Independent, Constrained-based, Closed Information Extraction System for the Hindi Language.*

---

## About

**Nitin Singh** | B.Tech CSE, KIIT University (2023–2027)  
Native Hindi speaker | NLP & ML  
[GitHub](https://github.com/singhhnitin) · [LinkedIn](https://linkedin.com/in/nitin-singh12) · nitinsingh3323@gmail.com

*GSoC 2026 application for DBpedia Association — Hindi Chapter project.*
