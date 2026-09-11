"""Evaluate fine-tuned SetFit intent classifier on held-out test split."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from setfit import SetFitModel
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_classifier")

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


def load_test_split(path: str = "data/processed/test_split.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    intents = [item["intent"] for item in data]
    return texts, labels, intents


def main():
    model_dir = Path("models/intent_setfit")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} not found. Run training first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SetFitModel from {model_dir} on {device}...")
    model = SetFitModel.from_pretrained(str(model_dir))
    if hasattr(model, "to"):
        model.to(device)

    # 1. Load held-out test data
    texts, true_labels, true_intents = load_test_split()
    logger.info(f"Loaded {len(texts)} held-out test cases.")

    # 2. Run model.predict on test set
    preds = model.predict(texts)
    if hasattr(preds, "cpu"):
        preds = preds.cpu().numpy().tolist()
    elif isinstance(preds, np.ndarray):
        preds = preds.tolist()

    # Normalize predictions to class indices
    pred_labels: List[int] = []
    pred_intents: List[str] = []
    for p in preds:
        if isinstance(p, int):
            pred_labels.append(p)
            pred_intents.append(ID2LABEL.get(p, "UNKNOWN"))
        elif isinstance(p, str):
            p_clean = p.strip().upper()
            idx = LABEL2ID.get(p_clean, 4)
            pred_labels.append(idx)
            pred_intents.append(p_clean)
        else:
            p_idx = int(p)
            pred_labels.append(p_idx)
            pred_intents.append(ID2LABEL.get(p_idx, "UNKNOWN"))

    # 3. Custom Evaluation Metrics via scikit-learn
    print("\n" + "=" * 65)
    print("SETFIT INTENT CLASSIFIER EVALUATION REPORT")
    print("=" * 65)

    report = classification_report(
        y_true=true_intents,
        y_pred=pred_intents,
        labels=TAXONOMY,
        zero_division=0,
    )
    print(report)

    print("\n" + "=" * 65)
    print("CONFUSION MATRIX (Rows: Actual, Columns: Predicted)")
    print("=" * 65)
    cm = confusion_matrix(true_intents, pred_intents, labels=TAXONOMY)
    
    # Pretty print confusion matrix
    header = f"{'Actual \\ Pred':<18}" + "".join(f"{t[:10]:>12}" for t in TAXONOMY)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{TAXONOMY[i]:<18}" + "".join(f"{val:>12}" for val in row)
        print(row_str)

    # 4. GPU Inference Latency Benchmark
    print("\n" + "=" * 65)
    print("GPU INFERENCE BENCHMARK (NVIDIA RTX 4050)")
    print("=" * 65)

    # Warmup
    _ = model.predict(texts[:5])
    if device == "cuda":
        torch.cuda.synchronize()

    # Single-item latency
    latencies = []
    for text in texts:
        t0 = time.perf_counter()
        _ = model.predict([text])
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Batch throughput
    t0 = time.perf_counter()
    _ = model.predict(texts)
    if device == "cuda":
        torch.cuda.synchronize()
    batch_time = time.perf_counter() - t0
    throughput = len(texts) / max(batch_time, 1e-6)

    print(f"Average Single-Tweet Latency: {avg_latency:.2f} ms")
    print(f"95th Percentile Latency:     {p95_latency:.2f} ms")
    print(f"Batch Throughput:            {throughput:.1f} tweets/second")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

