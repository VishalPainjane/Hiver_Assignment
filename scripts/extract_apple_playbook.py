"""Extract, sanitize, and curate Apple's historical resolution playbook from TWCS data.

Phase 1 (Offline Brain):
1. Extract first-turn customer complaint ↔ Apple first reply.
2. Sanitize user handles to {customer_handle} and URLs to {dm_link} / {help_link}.
3. Categorize into 6 intents (BATTERY_DRAIN, DEVICE_BOOT, ACCOUNT_ACCESS, BILLING_DISPUTE, UPDATE_FAILURE, UNKNOWN).
4. Deduplicate to a high-diversity canonical vault (200-300 unique pairs per intent).
5. Output: data/processed/apple_playbook_vault.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_playbook")

HANDLE_PATTERN = re.compile(r"@[\w_]+", re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)

CUES = {
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


def sanitize_customer_text(text: str) -> str:
    """Clean handles and urls from customer complaint."""
    cleaned = HANDLE_PATTERN.sub("", text)
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def sanitize_apple_reply(text: str) -> str:
    """Sanitize Apple's reply with standardized placeholders."""
    # Replace the initial customer handle mention with {customer_handle}
    text = re.sub(r"^@[\w_]+\s*", "{customer_handle} ", text.strip())
    # Remove any stray @115858 (Apple's handle tag in TWCS)
    text = re.sub(r"@115858\b", "", text)
    text = re.sub(r"@[\w_]+", "{customer_handle}", text)

    # Standardize links: if text mentions DM or private, mark as {dm_link}
    def link_replacer(match):
        return "{dm_link}" if ("dm" in text.lower() or "direct message" in text.lower()) else "{help_link}"

    text = URL_PATTERN.sub(link_replacer, text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def classify_text_by_cues(text: str) -> str:
    """Fast cue-based intent matching for streaming curation."""
    lowered = text.lower()
    scores = {}
    for intent, terms in CUES.items():
        score = sum(1 for t in terms if t in lowered)
        if score > 0:
            scores[intent] = score
    if not scores:
        return "UNKNOWN"
    return max(scores.items(), key=lambda x: x[1])[0]


def normalize_for_dedup(text: str) -> str:
    """Normalize reply text to detect duplicate boilerplate."""
    t = text.lower()
    t = re.sub(r"\{customer_handle\}|\{dm_link\}|\{help_link\}", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = WHITESPACE_PATTERN.sub(" ", t).strip()
    return t


def extract_playbook(
    cases_path: Path,
    output_path: Path,
    max_per_intent: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract and deduplicate all distinct historical customer-agent pairs across the full dataset."""
    logger.info(f"Streaming ALL cases from {cases_path} (Full Dataset Pass)...")
    vault: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_queries: Set[str] = set()
    pair_counter = 0
    scanned_lines = 0

    if not cases_path.exists():
        raise FileNotFoundError(f"{cases_path} does not exist.")

    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            scanned_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns = case.get("raw_turns", [])
            if len(turns) < 2:
                continue

            # Look for the first customer -> AppleSupport pair
            customer_turn: Optional[Dict[str, str]] = None
            apple_turn: Optional[Dict[str, str]] = None

            for i in range(len(turns) - 1):
                t1 = turns[i]
                t2 = turns[i + 1]
                a1 = str(t1.get("author_id", "")).lower()
                a2 = str(t2.get("author_id", "")).lower()
                if a1 != "applesupport" and a2 == "applesupport":
                    customer_turn = t1
                    apple_turn = t2
                    break

            if not customer_turn or not apple_turn:
                continue

            cust_text = customer_turn.get("text", "")
            apple_text = apple_turn.get("text", "")

            cust_clean = sanitize_customer_text(cust_text)
            apple_clean = sanitize_apple_reply(apple_text)

            # Quality filters
            if len(cust_clean) < 15 or len(apple_clean) < 20:
                continue
            if not ("{" in apple_clean or "dm" in apple_clean.lower() or "apple" in apple_clean.lower() or "help" in apple_clean.lower()):
                continue

            # Deduplicate by distinct customer query to preserve all unique symptom formulations
            dedup_key = normalize_for_dedup(cust_clean)
            if dedup_key in seen_queries:
                continue
            seen_queries.add(dedup_key)

            intent = classify_text_by_cues(cust_clean)
            if max_per_intent is not None and len(vault[intent]) >= max_per_intent:
                continue

            pair_counter += 1
            vault[intent].append({
                "pair_id": f"pair_{pair_counter:05d}",
                "case_id": str(case.get("case_id", "")),
                "intent": intent,
                "customer_query": cust_clean,
                "apple_reply": apple_clean,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: len(v) for k, v in vault.items()}
    total = sum(summary.values())
    result = {
        "metadata": {
            "source": str(cases_path),
            "total_pairs": total,
            "max_per_intent": max_per_intent,
            "counts_by_intent": summary,
        },
        "vault": dict(vault),
    }

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)

    logger.info(f"Saved {total} high-quality curated pairs to {output_path}. Counts: {summary}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and curate Apple's Playbook Vault.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/apple_cases.jsonl"),
        help="Path to apple_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/apple_playbook_vault.json"),
        help="Path to save curated playbook vault",
    )
    parser.add_argument(
        "--max-per-intent",
        type=int,
        default=None,
        help="Maximum unique canonical pairs per intent (None = full dataset pass)",
    )
    args = parser.parse_args()
    extract_playbook(args.input, args.output, args.max_per_intent)

