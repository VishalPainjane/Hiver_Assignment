"""Primary immutable data contracts for the support decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SupportCase:
    case_id: str
    clean_text: str
    customer_id: str = "anon"
    channel: str = "twitter"
    raw_turns: List[Dict[str, str]] = field(default_factory=list)
    locale: str = "en"
    account_tier: str = "standard"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class CustomerNeed:
    primary_intent: str
    confidence: float = 1.0


