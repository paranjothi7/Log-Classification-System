"""
training/train_bert.py
Fine-tune BERT on synthetic_logs.csv and save the model to models/
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import json

# Paths
CSV_PATH   = Path("../resources/synthetic_logs.csv")
MODEL_DIR  = Path("../models/bert_log_classifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load & Prepare Data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df = df[["log_message", "target_label"]].dropna()
df.columns = ["text", "label"]

# Encode labels
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

# Save label mapping
label_map = {i: l for i, l in enumerate(le.classes_)}
with open(MODEL_DIR / "label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)
print(f"Labels: {label_map}")

# Train/test split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(
    columns={"label_id": "labels"}))
test_ds  = Dataset.from_pandas(test_df[["text", "label_id"]].rename(
    columns={"label_id": "labels"}))

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

# Model
num_labels = len(le.classes_)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    id2label=label_map,
    label2id={l: i for i, l in label_map.items()},
)

# Training
args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",           # ← renamed from evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir=str(MODEL_DIR / "logs"),
    logging_steps=50,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=False,                      # ← force False on CPU (no CUDA)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = (preds == labels).mean()
    return {"accuracy": round(float(acc), 4)}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

# Save Model & Tokenizer
print(f"\nSaving model to {MODEL_DIR} ...")
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print("Model saved!")

# Evaluate & Save Report
print("\nEvaluating...")
preds_output = trainer.predict(test_ds)
preds  = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids

report = classification_report(
    labels, preds,
    target_names=le.classes_,
    output_dict=True,
)

# Print report
print("\n" + classification_report(labels, preds, target_names=le.classes_))

# Save report to JSON
with open(MODEL_DIR / "eval_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"Evaluation report saved to {MODEL_DIR}/eval_report.json")