"""Pydantic v2 schemas and data contracts for the inference API."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Payload for classifying a single support tweet."""
    text: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Tweet content to classify",
        examples=["My iPhone battery drops from 90% to 15% in two hours and gets scorching hot"],
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Optional external case identifier",
        examples=["case_12345"],
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Optional customer identifier",
    )


class BatchClassifyRequest(BaseModel):
    """Payload for streaming batch classification with deduplication."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="List of tweet contents to classify (max 256 per batch)",
    )
    case_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of corresponding case identifiers",
    )


class TopClassScore(BaseModel):
    """Class name and calibrated probability for top-N ranking."""
    intent: str
    probability: float


class ClassifyResponse(BaseModel):
    """Standardized classification response with confidence & ambiguity gating."""
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    top2: List[TopClassScore]
    entropy: float
    ambiguous: bool
    cached: bool
    latency_ms: float
    case_id: Optional[str] = None


class BatchClassifyResponse(BaseModel):
    """Aggregated batch response reporting latency, cache hits, and deduplication savings."""
    results: List[ClassifyResponse]
    total_latency_ms: float
    cache_hits: int
    deduplicated_count: int
    batch_size: int


class HealthResponse(BaseModel):
    """Readiness probe schema reporting model state, device, and Redis status."""
    status: str
    model_loaded: bool
    device: str
    redis_connected: bool
    taxonomy: List[str]


class PrecedentItem(BaseModel):
    """Historical customer-agent resolution pair retrieved from the playbook vault."""
    pair_id: str
    case_id: str
    intent: str
    customer_query: str
    apple_reply: str
    similarity_score: float


class RetrievalSummary(BaseModel):
    """Semantic precedent retrieval summary."""
    intent: str
    top_similarity: float
    confidence_zone: str
    matches: List[PrecedentItem]


class RoutingSummary(BaseModel):
    """Deterministic routing decision with machine rule code and stated reason."""
    decision: str
    priority: str
    rule_code: str
    stated_reason: str


class DraftReplyRequest(BaseModel):
    """Payload for generating grounded brand reply and routing decision."""
    text: str = Field(..., min_length=2, max_length=500, description="Customer tweet to process")
    handle: Optional[str] = Field(default="@user", description="Customer Twitter handle")
    case_id: Optional[str] = Field(default=None, description="Optional case identifier")


class DraftReplyResponse(BaseModel):
    """Complete Decision Engine output bundle."""
    case_id: str
    customer_text: str
    customer_handle: str
    intent: str
    confidence: float
    routing: RoutingSummary
    retrieval: RetrievalSummary
    draft_reply: Optional[str] = None
    failsafe_used: bool = False
    character_count: int = 0
    latency_ms: float

