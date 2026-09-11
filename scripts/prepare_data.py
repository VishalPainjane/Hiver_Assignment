"""Prepare @AppleSupport cases from twcs.csv export."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_HANDLE_PATTERN = re.compile(r"@[\w_]+", re.UNICODE)
_URL_PATTERN = re.compile(r"https?://\S+", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


def clean_tweet_text(text: str) -> str:
    cleaned = _URL_PATTERN.sub("", text)
    cleaned = _HANDLE_PATTERN.sub("", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def parse_timestamp(value: str) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    try:
        dt = datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %z %Y")
        return dt.isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def extract_apple_cases(
    csv_path: str | Path,
    brand: str = "AppleSupport",
    output_path: str | Path = "data/processed/apple_cases.jsonl",
) -> int:
    target = brand.lstrip("@").lower()
    path = Path(csv_path)

    parent_of: Dict[str, str] = {}
    children_of: Dict[str, List[str]] = defaultdict(list)
    brand_roots: Set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tid = str(row.get("tweet_id") or "").strip()
            pid = str(row.get("in_response_to_tweet_id") or "").strip()
            author = str(row.get("author_id") or "").lower()
            if pid:
                parent_of[tid] = pid
                children_of[pid].append(tid)
            if author == target:
                brand_roots.add(tid)

    selected: Set[str] = set(brand_roots)
    queue: deque[str] = deque(brand_roots)
    while queue:
        current = queue.popleft()
        parent = parent_of.get(current)
        if parent and parent not in selected:
            selected.add(parent)
            queue.append(parent)
        for child in children_of.get(current, []):
            if child not in selected:
                selected.add(child)
                queue.append(child)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("r", encoding="utf-8", newline="") as handle, out_file.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(handle)
        for row in reader:
            tid = str(row.get("tweet_id") or "").strip()
            author = str(row.get("author_id") or "").strip()
            if tid in selected and author.lower() != target:
                raw_text = str(row.get("text") or "").strip()
                clean = clean_tweet_text(raw_text)
                if not clean:
                    continue
                case = {
                    "case_id": tid,
                    "customer_id": author,
                    "channel": "twitter",
                    "raw_turns": [
                        {
                            "tweet_id": tid,
                            "author_id": author,
                            "text": raw_text,
                            "created_at": parse_timestamp(str(row.get("created_at") or "")),
                        }
                    ],
                    "clean_text": clean,
                    "locale": "en",
                    "account_tier": "STANDARD",
                    "timestamp": parse_timestamp(str(row.get("created_at") or "")),
                }
                out.write(json.dumps(case, ensure_ascii=False) + "\n")
                count += 1

    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to Kaggle twcs.csv export")
    parser.add_argument("--brand", default="AppleSupport")
    parser.add_argument("--output", default="data/processed/apple_cases.jsonl")
    args = parser.parse_args(argv)

    total = extract_apple_cases(args.input, args.brand, args.output)
    print(f"Extracted {total} Apple customer cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
