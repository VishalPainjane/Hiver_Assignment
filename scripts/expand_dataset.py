"""Expand dataset to 120 high-quality, verified cases per intent using GPU + LLM Active Learning."""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch
from setfit import SetFitModel

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import UnifiedLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_dataset")

TAXONOMY = [
    "ACCOUNT_ACCESS",
    "BATTERY_DRAIN",
    "BILLING_DISPUTE",
    "DEVICE_BOOT",
    "UNKNOWN",
    "UPDATE_FAILURE",
]
TARGET_PER_CLASS = 120

SEARCH_CUES = {
    "ACCOUNT_ACCESS": [
        "apple id", "password", "sign in", "login", "locked", "verification", "2fa",
        "two-factor", "icloud", "recover", "passcode", "security question", "disabled"
    ],
    "BATTERY_DRAIN": [
        "battery", "drain", "charge", "charger", "charging", "overheat", "dying",
        "battery health", "percentage", "drops from", "cable", "hot"
    ],
    "BILLING_DISPUTE": [
        "charged", "charge", "refund", "subscription", "itunes", "app store", "receipt",
        "billing", "payment", "bank", "invoice", "unauthorized", "cancel", "money"
    ],
    "DEVICE_BOOT": [
        "turn on", "won't turn", "black screen", "frozen", "freeze", "stuck on apple logo",
        "reboot", "restart", "boot loop", "crash", "white screen", "hard reset"
    ],
    "UPDATE_FAILURE": [
        "ios update", "update failed", "unable to verify update", "error occurred installing",
        "storage full update", "ios 11", "downloading update", "software update", "installing update"
    ],
    "UNKNOWN": [
        "store", "appointment", "hours", "macbook", "airpods", "trade in", "repair",
        "genius bar", "case", "color", "shipping", "delivery", "broken glass"
    ],
}


def load_existing_golden(path: str = "data/golden_set.json") -> List[Dict[str, str]]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for c in data.get("cases", []):
        text = c.get("case", {}).get("clean_text", "").strip()
        intent = c.get("expected", {}).get("intent", "").strip().upper()
        if text and intent in TAXONOMY:
            cases.append({"text": text, "intent": intent, "source": "golden"})
    logger.info(f"Loaded {len(cases)} initial golden cases: {dict(Counter(c['intent'] for c in cases))}")
    return cases


def mine_candidate_pools(
    corpus_path: str = "data/processed/apple_cases.jsonl",
    existing_texts: Set[str] = None,
    pool_size: int = 350,
) -> Dict[str, List[str]]:
    existing = existing_texts or set()
    pools = defaultdict(list)
    seen_texts = set(existing)

    logger.info("Scanning 131k Apple cases for diverse candidates...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                text = row.get("clean_text", "").strip()
                if len(text) < 25 or len(text) > 250 or text in seen_texts:
                    continue

                text_lower = text.lower()
                for intent, cues in SEARCH_CUES.items():
                    if len(pools[intent]) >= pool_size:
                        continue
                    if any(cue in text_lower for cue in cues):
                        pools[intent].append(text)
                        seen_texts.add(text)
                        break

                if all(len(pools[k]) >= pool_size for k in TAXONOMY):
                    break
            except Exception:
                continue

    for k, v in pools.items():
        logger.info(f"Mined candidate pool for {k}: {len(v)} tweets")
    return pools


BATCH_PROMPT = """You are an expert customer support annotator for @AppleSupport.
Classify each customer tweet into EXACTLY ONE of:
[ACCOUNT_ACCESS, BATTERY_DRAIN, BILLING_DISPUTE, DEVICE_BOOT, UNKNOWN, UPDATE_FAILURE]

CRITICAL RULES:
- If a tweet is ambiguous, contradictory, or borderline between two classes, classify as "DISCARD".
- Only return high-confidence, clean ground truth.
- Output strictly a JSON array of objects:
[
  {"id": 1, "intent": "BATTERY_DRAIN"},
  ...
]
"""


def verify_with_llm(client: UnifiedLLMClient, batch: List[str]) -> List[str]:
    numbered_tweets = "\n".join(f"{i+1}. \"{text}\"" for i, text in enumerate(batch))
    prompt = f"{BATCH_PROMPT}\nTweets to classify:\n{numbered_tweets}"

    try:
        raw_json = client.generate(prompt=prompt, json_mode=True, temperature=0.0, max_tokens=600)
        data = json.loads(raw_json)
        results = ["DISCARD"] * len(batch)
        if isinstance(data, list):
            for item in data:
                idx = int(item.get("id", 0)) - 1
                intent = str(item.get("intent", "DISCARD")).upper().strip()
                if 0 <= idx < len(batch) and intent in TAXONOMY:
                    results[idx] = intent
        return results
    except Exception as e:
        logger.warning(f"LLM verification error: {e}")
        return ["DISCARD"] * len(batch)


def main():
    existing_cases = load_existing_golden()
    existing_texts = {c["text"] for c in existing_cases}
    counts = Counter(c["intent"] for c in existing_cases)

    # 1. Load trained SetFit model for fast GPU scoring
    model_path = Path("models/intent_setfit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SetFit model from {model_path} on {device}...")
    model = SetFitModel.from_pretrained(str(model_path))
    if hasattr(model, "to"):
        model.to(device)

    # 2. Mine candidate pools
    candidate_pools = mine_candidate_pools(existing_texts=existing_texts, pool_size=350)
    client = UnifiedLLMClient(primary="ollama", fallback="groq")

    expanded_cases = list(existing_cases)

    for intent in TAXONOMY:
        needed = TARGET_PER_CLASS - counts[intent]
        if needed <= 0:
            logger.info(f"Class {intent} already has {counts[intent]} cases (>= {TARGET_PER_CLASS}). Skipping.")
            continue

        candidates = candidate_pools.get(intent, [])
        random.shuffle(candidates)
        logger.info(f"Scoring {len(candidates)} candidates for {intent} on GPU (need {needed})...")

        # GPU batch probability scoring
        batch_size = 32
        borderline_candidates = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            probs = model.predict_proba(batch)
            if hasattr(probs, "cpu"):
                probs = probs.cpu().numpy()

            for text, row in zip(batch, probs):
                pred_idx = int(row.argmax())
                pred_intent = TAXONOMY[pred_idx]
                conf = float(row[pred_idx])

                # High confidence agreement on GPU (conf >= 0.80) -> Accept directly
                if conf >= 0.80 and pred_intent in TAXONOMY and counts[pred_intent] < TARGET_PER_CLASS:
                    expanded_cases.append({
                        "text": text,
                        "intent": pred_intent,
                        "source": "gpu_high_conf",
                    })
                    counts[pred_intent] += 1
                # Borderline candidate (0.60 <= conf < 0.80) -> Queue for LLM verification
                elif 0.60 <= conf < 0.80 and counts[intent] < TARGET_PER_CLASS:
                    borderline_candidates.append(text)

            if counts[intent] >= TARGET_PER_CLASS:
                break

        # If more are needed, verify borderline candidates with LLM in small paced batches
        if counts[intent] < TARGET_PER_CLASS and borderline_candidates:
            logger.info(f"Running LLM verification for {len(borderline_candidates[:needed*2])} borderline candidates...")
            llm_batch = borderline_candidates[: min(len(borderline_candidates), needed * 2)]
            sub_batch_size = 10
            for j in range(0, len(llm_batch), sub_batch_size):
                if counts[intent] >= TARGET_PER_CLASS:
                    break
                sub = llm_batch[j : j + sub_batch_size]
                verified_labels = verify_with_llm(client, sub)

                for text, v_intent in zip(sub, verified_labels):
                    if v_intent in TAXONOMY and counts[v_intent] < TARGET_PER_CLASS:
                        expanded_cases.append({
                            "text": text,
                            "intent": v_intent,
                            "source": "llm_verified",
                        })
                        counts[v_intent] += 1

        logger.info(f"Progress: {intent} now at {counts[intent]} / {TARGET_PER_CLASS} cases")

    # 3. Save expanded dataset
    output_path = Path("data/expanded_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(expanded_cases, f, ensure_ascii=False, indent=2)

    final_counts = Counter(c["intent"] for c in expanded_cases)
    logger.info(f"Expanded dataset saved to {output_path}. Total: {len(expanded_cases)} cases")
    for k in TAXONOMY:
        logger.info(f"  - {k:<18}: {final_counts[k]} cases")


if __name__ == "__main__":
    main()

