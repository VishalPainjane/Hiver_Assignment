"""Prepare an @AppleSupport Kaggle export into contextual cases and reference evidence."""
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
from typing import Any, Dict, Iterable, List, Sequence, Set
from typing import Dict, List, Sequence, Set

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_HANDLE_PATTERN = re.compile(r"@[\w_]+", re.UNICODE)
_URL_PATTERN = re.compile(r"https?://\S+", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)

from src.context.assembler import ContextAssembler

def clean_tweet_text(text: str) -> str:
    cleaned = _URL_PATTERN.sub("", text)
    cleaned = _HANDLE_PATTERN.sub("", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()

def select_brand_components(rows: Sequence[Dict[str, str]], brand: str) -> List[Dict[str, str]]:
    """Keep entire reply-graph components that contain a brand-authored tweet."""
def select_brand_components(csv_path: str | Path, brand: str) -> List[Dict[str, str]]:
    """Stream twcs.csv and keep entire reply-graph components containing brand-authored tweets."""

def parse_timestamp(value: str) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    try:
        dt = datetime.strptime(value.strip(), "%a %b %d %H:%M:%S %z %Y")
        return dt.isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def extract_apple_cases(csv_path: str | Path, brand: str = "AppleSupport", output_path: str | Path = "data/processed/apple_cases.jsonl") -> int:
    target = brand.lstrip("@").lower()
    by_id = {str(row.get("tweet_id") or ""): row for row in rows if row.get("tweet_id")}
    children: Dict[str, List[str]] = defaultdict(list)
    for tweet_id, row in by_id.items():
        parent = str(row.get("in_response_to_tweet_id") or "")
        if parent in by_id:
            children[parent].append(tweet_id)
    selected: Set[str] = {tweet_id for tweet_id, row in by_id.items() if str(row.get("author_id") or "").lower() == target}
    queue: deque[str] = deque(selected)
    path = Path(csv_path)

    parent_of: Dict[str, str] = {}
    children_of: Dict[str, List[str]] = defaultdict(list)
    brand_roots: Set[str] = set()

    path = Path(csv_path)
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
        parent = str(by_id[current].get("in_response_to_tweet_id") or "")
        neighbours = list(children.get(current, [])) + ([parent] if parent in by_id else [])
        for neighbour in neighbours:
            if neighbour not in selected:
                selected.add(neighbour)
                queue.append(neighbour)
    return [row for tweet_id, row in by_id.items() if tweet_id in selected]
        parent = parent_of.get(current)
        if parent and parent not in selected:
            selected.add(parent)
            queue.append(parent)
        for child in children_of.get(current, []):
            if child not in selected:
                selected.add(child)
                queue.append(child)

    brand_rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("r", encoding="utf-8", newline="") as handle, out_file.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(handle)
        for row in reader:
            tid = str(row.get("tweet_id") or "").strip()
            if tid in selected:
                brand_rows.append(dict(row))
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

    return brand_rows
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to Kaggle twcs.csv export")
    parser.add_argument("--brand", default="AppleSupport")
    parser.add_argument("--output", default="data/processed/apple_cases.jsonl")
    parser.add_argument("--evidence-output", default="data/processed/apple_reference_evidence.json")
    args = parser.parse_args(argv)

    assembler = ContextAssembler(args.brand)
    rows = assembler.read_csv(args.input)
    brand_rows = select_brand_components(rows, args.brand)
    brand_rows = select_brand_components(args.input, args.brand)
    if not brand_rows:
        raise ValueError(f"No reply-graph component containing @{args.brand.lstrip('@')} was found")
    cases = assembler.reconstruct(brand_rows)
    assembler.write_cases_jsonl(cases, args.output)
    evidence_target = Path(args.evidence_output)
    evidence_target.parent.mkdir(parents=True, exist_ok=True)
    with evidence_target.open("w", encoding="utf-8") as handle:
        json.dump(assembler.resolved_evidence_rows(brand_rows), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Prepared {len(cases)} inbound cases and reference evidence from {len(brand_rows)} tweets.")
    total = extract_apple_cases(args.input, args.brand, args.output)
    print(f"Extracted {total} Apple customer cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

