"""
evaluate_finetuned.py — Compare baseline vs fine-tuned Gemma-3-1B
Run after downloading model from Kaggle.
Usage: python3 -m src.nlu.evaluate_finetuned --model_path /path/to/model
"""

import os
import sys
import json
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
                    default="models/vaada-gemma3-1b",
                    help="Path to fine-tuned model")
parser.add_argument("--test_csv", type=str,
                    default="data/processed/test.csv")
parser.add_argument("--max_samples", type=int, default=200,
                    help="Samples to evaluate (keep low for speed)")
args = parser.parse_args()

# ── Load test data ────────────────────────────────────────────
print("Loading test data...")
test_df = pd.read_csv(args.test_csv)
test_df  = test_df.sample(
    min(args.max_samples, len(test_df)),
    random_state=42
)
print(f"Evaluating on {len(test_df)} samples")

INTENTS = [
    "promise_to_pay", "needs_more_time",
    "partial_payment", "dispute", "refusal"
]

# ── Baseline evaluation ───────────────────────────────────────
print("\n=== BASELINE: TF-IDF + Logistic Regression ===")
baseline_results = {}

try:
    with open("models/baseline_pipeline.pkl", "rb") as f:
        baseline = pickle.load(f)

    X_test = (
        test_df["reminder"].fillna("") +
        " [SEP] " +
        test_df["reply"].fillna("")
    )
    y_test  = test_df["intent"]

    y_pred_baseline = baseline.predict(X_test)
    baseline_f1     = f1_score(y_test, y_pred_baseline,
                               average="weighted")
    baseline_acc    = accuracy_score(y_test, y_pred_baseline)

    print(f"Accuracy : {baseline_acc:.4f}")
    print(f"F1 Score : {baseline_f1:.4f}")
    print("\nPer-class:")
    print(classification_report(y_test, y_pred_baseline,
                                target_names=INTENTS,
                                zero_division=0))

    baseline_results = {
        "accuracy": round(baseline_acc, 4),
        "f1":       round(baseline_f1, 4),
        "report":   classification_report(
            y_test, y_pred_baseline,
            target_names=INTENTS,
            output_dict=True,
            zero_division=0
        )
    }

except Exception as e:
    print(f"Baseline error: {e}")
    baseline_results = {"error": str(e)}

# ── Fine-tuned model evaluation ───────────────────────────────
print(f"\n=== FINE-TUNED: Gemma-3-1B QLoRA ===")
print(f"Loading from: {args.model_path}")
finetuned_results = {}

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. "
            f"Download from Kaggle first."
        )

    BASE_MODEL = "/kaggle/input/models/google/gemma-3/transformers/gemma-3-1b-it/1"
    if not os.path.exists(BASE_MODEL):
        BASE_MODEL = "google/gemma-3-1b-it"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    SYSTEM = """You are VAADA, an AI for Hinglish payment collection.
Classify customer intent as one of:
promise_to_pay, needs_more_time, partial_payment, dispute, refusal"""

    def predict_intent(reminder: str, reply: str) -> str:
        prompt = (
            f"<start_of_turn>user\n{SYSTEM}\n\n"
            f"Reminder: {reminder}\n"
            f"Reply: {reply}<end_of_turn>\n"
            f"<start_of_turn>model\nIntent:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()

        # Extract intent from response
        for intent in INTENTS:
            if intent in response:
                return intent

        # Fallback
        if "promise" in response:   return "promise_to_pay"
        if "time" in response:      return "needs_more_time"
        if "partial" in response:   return "partial_payment"
        if "dispute" in response:   return "dispute"
        if "refusal" in response:   return "refusal"
        return "promise_to_pay"

    print(f"\nRunning inference on {len(test_df)} samples...")
    y_pred_ft = []
    y_true    = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        if i % 20 == 0:
            print(f"  {i}/{len(test_df)}...")
        pred = predict_intent(
            str(row.get("reminder", "")),
            str(row.get("reply", ""))
        )
        y_pred_ft.append(pred)
        y_true.append(row["intent"])

    ft_f1  = f1_score(y_true, y_pred_ft, average="weighted",
                      zero_division=0)
    ft_acc = accuracy_score(y_true, y_pred_ft)

    print(f"\nAccuracy : {ft_acc:.4f}")
    print(f"F1 Score : {ft_f1:.4f}")
    print("\nPer-class:")
    print(classification_report(y_true, y_pred_ft,
                                target_names=INTENTS,
                                zero_division=0))

    finetuned_results = {
        "accuracy": round(ft_acc, 4),
        "f1":       round(ft_f1, 4),
        "report":   classification_report(
            y_true, y_pred_ft,
            target_names=INTENTS,
            output_dict=True,
            zero_division=0
        )
    }

    # Comparison
    print("\n=== COMPARISON ===")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Baseline (TF-IDF + LR)':<30} "
          f"{baseline_results.get('accuracy', 0):>10.4f} "
          f"{baseline_results.get('f1', 0):>10.4f}")
    print(f"{'Fine-tuned (Gemma-3-1B)':<30} "
          f"{ft_acc:>10.4f} "
          f"{ft_f1:>10.4f}")

    improvement = ft_f1 - baseline_results.get("f1", 0)
    print(f"\nImprovement: {improvement:+.4f} F1")

except Exception as e:
    print(f"Fine-tuned model error: {e}")
    finetuned_results = {"error": str(e)}

# ── Save results ──────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
results = {
    "baseline":   baseline_results,
    "finetuned":  finetuned_results,
    "test_size":  len(test_df),
}

with open("outputs/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to outputs/model_comparison.json")
print("Done.")
