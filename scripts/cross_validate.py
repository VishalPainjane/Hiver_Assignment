"""Stratified 5-Fold Cross-Validation for SetFit Intent Classifier on NVIDIA RTX 4050 GPU."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from setfit import SetFitModel, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cross_validate")

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

    logger.info(f"Loaded {len(texts)} valid labeled cases from {p}")
    return np.array(texts), np.array(labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    texts, labels = load_dataset()
    logger.info(f"Dataset total size: {len(texts)} cases across {len(TAXONOMY)} classes")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_macro_f1s = []
    per_class_f1s = {intent: [] for intent in TAXONOMY}

    logger.info("Starting Stratified 5-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        logger.info(f"\n{'='*25} Fold {fold} / 5 {'='*25}")
        train_texts, val_texts = texts[train_idx].tolist(), texts[val_idx].tolist()
        train_labels, val_labels = labels[train_idx].tolist(), labels[val_idx].tolist()
        logger.info(f"Fold {fold}: Train={len(train_texts)} | Val={len(val_texts)}")

        train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

        model = SetFitModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            labels=TAXONOMY,
        )

        args = TrainingArguments(
            batch_size=16,
            num_iterations=20,
            num_epochs=1,
            body_learning_rate=2e-5,
            warmup_proportion=0.1,
            use_amp=(device == "cuda"),
            seed=42 + fold,
            eval_strategy="epoch",
            report_to="none",
            output_dir=f"models/checkpoints_fold_{fold}",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            metric="accuracy",
        )

        trainer.train()

        val_preds = model.predict(val_texts)
        if hasattr(val_preds, "cpu"):
            val_preds = val_preds.cpu().numpy().tolist()
        elif isinstance(val_preds, np.ndarray):
            val_preds = val_preds.tolist()

        pred_labels = []
        for p in val_preds:
            if isinstance(p, int):
                pred_labels.append(p)
            elif isinstance(p, str):
                pred_labels.append(LABEL2ID.get(p.strip().upper(), 4))
            else:
                pred_labels.append(int(p))

        acc = accuracy_score(val_labels, pred_labels)
        macro_f1 = f1_score(val_labels, pred_labels, average="macro")
        cls_report = classification_report(val_labels, pred_labels, target_names=TAXONOMY, output_dict=True, zero_division=0)

        fold_accuracies.append(acc)
        fold_macro_f1s.append(macro_f1)
        for intent in TAXONOMY:
            per_class_f1s[intent].append(cls_report[intent]["f1-score"])

        logger.info(f"Fold {fold} Results -> Accuracy: {acc*100:.2f}%, Macro-F1: {macro_f1:.4f}")

    print("\n" + "=" * 65)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Overall Accuracy:  {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
    print(f"Overall Macro-F1:  {np.mean(fold_macro_f1s):.4f} ± {np.std(fold_macro_f1s):.4f}")
    print("\nPer-Class F1 Scores across 5 folds:")
    for intent in TAXONOMY:
        mean_f1 = np.mean(per_class_f1s[intent])
        std_f1 = np.std(per_class_f1s[intent])
        print(f"  - {intent:<18}: {mean_f1:.4f} ± {std_f1:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
