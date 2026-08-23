import pandas as pd
import json
import random
from pathlib import Path

# Load existing data
records = []
with open("data/raw/single_turn.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except:
                pass

df = pd.DataFrame(records)
df = df.drop_duplicates(subset=["reminder", "reply"])
print(f"Base samples: {len(df)}")

# Augmentation components
AMOUNTS = [1500, 2200, 3500, 4500, 5500, 6000,
           7500, 8000, 9000, 10000, 12000, 15000,
           18000, 22000, 25000, 30000, 50000]

NAMES = ["Rahul", "Amit", "Priya", "Suresh", "Neha",
         "Vikram", "Deepak", "Anjali", "Ravi", "Pooja",
         "Sanjay", "Kavita", "Arjun", "Meena", "Rohit",
         "Sunita", "Manoj", "Geeta", "Rajesh", "Divya"]

PTP_DATES = ["kal", "parso", "friday tak", "monday ko",
             "15 tarikh ko", "next week", "2-3 din mein",
             "shaam tak", "aaj raat tak", "tuesday morning",
             "weekend tak", "mahine end tak"]

TYPOS = {
    "nahi": ["nai", "nhi"],
    "please": ["plz", "pls"],
    "bhai": ["bhi", "bhaii"],
    "pakka": ["paka", "pakk"],
    "dunga": ["duga", "dunnga"],
    "payment": ["pymnt", "paymnt"],
}

EMOJIS = ["🙏", "🙏🙏", "😔", "😢", "✅", "👍", "😓"]

def swap_amount(text, old_amount):
    new_amount = random.choice([a for a in AMOUNTS if a != old_amount])
    text = text.replace(f"₹{old_amount}", f"₹{new_amount}")
    text = text.replace(str(old_amount), str(new_amount))
    return text, new_amount

def swap_name(text):
    for name in NAMES:
        if name in text:
            new_name = random.choice([n for n in NAMES if n != name])
            return text.replace(name, new_name, 1)
    return text

def inject_typo(text):
    for word, variants in TYPOS.items():
        if word in text.lower():
            return text.replace(word, random.choice(variants), 1)
    return text

def toggle_emoji(text):
    has_emoji = any(e in text for e in EMOJIS)
    if has_emoji:
        for e in EMOJIS:
            text = text.replace(e, "").strip()
    else:
        text = text + " " + random.choice(EMOJIS)
    return text

def augment_record(rec):
    new_rec = rec.copy()
    amount = rec.get("amount", 5000)
    aug_type = random.choice(["amount", "name", "typo", "emoji"])

    if aug_type == "amount":
        new_reminder, new_amount = swap_amount(rec["reminder"], amount)
        new_rec["reminder"] = new_reminder
        new_rec["amount"] = new_amount
        if rec.get("ptp_amount"):
            new_rec["ptp_amount"] = new_amount

    elif aug_type == "name":
        new_rec["reminder"] = swap_name(rec["reminder"])

    elif aug_type == "typo":
        new_rec["reply"] = inject_typo(rec["reply"])

    elif aug_type == "emoji":
        new_rec["reply"] = toggle_emoji(rec["reply"])

    return new_rec

# Augment to 10k
TARGET = 10000
needed = TARGET - len(df)
print(f"Need {needed} more samples")

augmented = []
base_records = df.to_dict("records")

while len(augmented) < needed:
    rec = random.choice(base_records)
    new_rec = augment_record(rec)
    if new_rec["reply"] != rec["reply"] or new_rec["reminder"] != rec["reminder"]:
        augmented.append(new_rec)

aug_df = pd.DataFrame(augmented)
final_df = pd.concat([df, aug_df], ignore_index=True)
final_df = final_df.drop_duplicates(subset=["reminder", "reply"])

print(f"Final samples: {len(final_df)}")
print(final_df["intent"].value_counts())

# Save
final_df.to_json("data/raw/single_turn_10k.jsonl",
                 orient="records", lines=True, force_ascii=False)

# New splits
train = final_df.sample(frac=0.7, random_state=42)
rest = final_df.drop(train.index)
val = rest.sample(frac=0.5, random_state=42)
test = rest.drop(val.index)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print("Saved to data/processed/")
