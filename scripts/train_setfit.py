"""Train SetFit Few-Shot Intent Classifier on NVIDIA RTX 4050 GPU."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from setfit import SetFitModel, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_setfit")

TAXONOMY = [
    "ACCOUNT_ACCESS",
    "BATTERY_DRAIN",
    "BILLING_DISPUTE",
    "DEVICE_BOOT",
    "UNKNOWN",
    "UPDATE_FAILURE",
]
LABEL2ID = {label: i for i, label in enumerate(TAXONOMY)}
ID2LABEL = {i: label for i, label in enumerate(TAXONOMY)}


def load_dataset(dataset_path: str = "data/expanded_dataset.json"):
    p = Path(dataset_path)
    if not p.exists():
        p = Path("data/golden_set.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: List[str] = []
    labels: List[int] = []
    label_names: List[str] = []

    items = data if isinstance(data, list) else data.get("cases", [])
    for item in items:
        if isinstance(item, dict) and "case" in item:
            text = item.get("case", {}).get("clean_text", "").strip()
            intent = item.get("expected", {}).get("intent", "").strip().upper()
        else:
            text = item.get("text", "").strip()
            intent = item.get("intent", "").strip().upper()

        if not text or intent not in LABEL2ID:
            continue
        texts.append(text)
        labels.append(LABEL2ID[intent])
        label_names.append(intent)

    logger.info(f"Loaded {len(texts)} valid labeled cases from {p}")
    return texts, labels, label_names


def main():
    # 1. Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        logger.info(f"GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    # 2. Load and split data (Stratified 80% train / 20% test)
    texts, labels, label_names = load_dataset()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    logger.info(f"Train split: {len(train_texts)} cases | Test split: {len(test_texts)} cases")

    # Save test split for benchmark verification
    test_data = [
        {"text": text, "label": label, "intent": ID2LABEL[label]}
        for text, label in zip(test_texts, test_labels)
    ]
    test_split_path = Path("data/processed/test_split.json")
    test_split_path.parent.mkdir(parents=True, exist_ok=True)
    with test_split_path.open("w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    logger.info(f"Saved held-out test split to {test_split_path}")

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels,
    })
    eval_dataset = Dataset.from_dict({
        "text": test_texts,
        "label": test_labels,
    })

    # 3. Initialize SetFitModel with all-MiniLM-L6-v2
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Initializing SetFitModel with backbone: {model_id}")
    model = SetFitModel.from_pretrained(
        model_id,
        labels=TAXONOMY,
    )

    # 4. Configure TrainingArguments with use_amp (mixed precision) and batch_size=16 on GPU
    args = TrainingArguments(
        batch_size=16,
        num_iterations=20,               # 20 pairs per example = ~11,520 contrastive pairs for Phase 1
        num_epochs=1,
        body_learning_rate=2e-5,
        warmup_proportion=0.1,           # 10% linear warmup
        use_amp=(device == "cuda"),      # Utilize RTX 4050 tensor cores natively via AMP (FP16)
        seed=42,
        eval_strategy="epoch",
        report_to="none",                # Disable dvclive/wandb callbacks
        output_dir="models/checkpoints",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
    )

    # 5. Contrastive tuning (Phase 1) + Classification head (Phase 2)
    logger.info("Starting SetFit contrastive training on GPU...")
    trainer.train()
    logger.info("Training completed successfully!")

    # 6. Save final fine-tuned model
    output_model_dir = Path("models/intent_setfit")
    output_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_model_dir))
    logger.info(f"Fine-tuned model successfully saved to {output_model_dir}")


if __name__ == "__main__":
    main()
