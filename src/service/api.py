"""Enterprise-grade FastAPI serving layer for SetFit intent classification."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.classification.setfit_classifier import SetFitIntentClassifier
from src.generation.brand_reply_generator import BrandReplyEngine
from src.service.cache import IntentCache, normalize_cache_key
from src.service.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    DraftReplyRequest,
    DraftReplyResponse,
    HealthResponse,
    PrecedentItem,
    RetrievalSummary,
    RoutingSummary,
    TopClassScore,
)

logger = logging.getLogger("service.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_prediction_details(
    probs_row: np.ndarray, taxonomy: List[str]
) -> Dict[str, Any]:
    """Compute calibrated class probabilities, top-2 ranking, Shannon entropy, and ambiguity flag."""
    sorted_indices = probs_row.argsort()[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx

    pred_intent = taxonomy[top1_idx]
    confidence = float(probs_row[top1_idx])
    margin = float(probs_row[top1_idx] - probs_row[top2_idx])

    # Shannon Entropy: H(P) = -sum(p * log2(p))
    probs_clipped = np.clip(probs_row, 1e-12, 1.0)
    entropy = float(-np.sum(probs_clipped * np.log2(probs_clipped)))

    # Ambiguity criteria: top margin < 0.15 or low confidence < 0.60
    is_ambiguous = bool(margin < 0.15 or confidence < 0.60)

    top2_list = [
        {"intent": taxonomy[top1_idx], "probability": round(confidence, 4)},
        {"intent": taxonomy[top2_idx], "probability": round(float(probs_row[top2_idx]), 4)},
    ]

    probs_dict = {
        tax: round(float(probs_row[i]), 4) for i, tax in enumerate(taxonomy)
    }

    return {
        "intent": pred_intent,
        "confidence": round(confidence, 4),
        "probabilities": probs_dict,
        "top2": top2_list,
        "entropy": round(entropy, 4),
        "ambiguous": is_ambiguous,
    }


def _run_model_predict_proba(
    classifier: Optional[Any], texts: List[str]
) -> np.ndarray:
    """Execute model prediction inside torch.inference_mode() without blocking autograd."""
    if classifier is None or getattr(classifier, "_model", None) is None:
        n_classes = len(getattr(app.state, "taxonomy", [])) or 6
        return np.full((len(texts), n_classes), 1.0 / n_classes)

    with torch.inference_mode():
        probs = classifier._model.predict_proba(texts)
        if hasattr(probs, "cpu"):
            probs = probs.cpu().numpy()
        elif not isinstance(probs, np.ndarray):
            probs = np.array(probs)
        return probs


def get_classifier() -> Optional[Any]:
    """Get or lazily initialize the SetFit classifier."""
    if not hasattr(app.state, "classifier") or app.state.classifier is None:
        try:
            from src.classification.setfit_classifier import SetFitIntentClassifier
            device = "cuda" if torch.cuda.is_available() else "cpu"
            classifier = SetFitIntentClassifier(model_path="models/intent_setfit", device=device)
            app.state.classifier = classifier
            taxonomy = list(classifier.taxonomy)
            if hasattr(classifier._model, "labels") and classifier._model.labels:
                taxonomy = list(classifier._model.labels)
            app.state.taxonomy = taxonomy
        except Exception as e:
            logger.warning(f"Could not load SetFit model: {e}")
            app.state.classifier = None
            app.state.taxonomy = ["BATTERY_DRAIN", "DEVICE_BOOT", "ACCOUNT_ACCESS", "BILLING_DISPUTE", "UPDATE_FAILURE", "UNKNOWN"]
    return app.state.classifier


def get_cache() -> IntentCache:
    """Get or lazily initialize the Redis/LRU cache."""
    if not hasattr(app.state, "cache") or app.state.cache is None:
        app.state.cache = IntentCache()
    return app.state.cache


def get_brand_reply_engine() -> BrandReplyEngine:
    """Get or lazily initialize the BrandReplyEngine decision engine."""
    if not hasattr(app.state, "brand_engine") or app.state.brand_engine is None:
        classifier = getattr(app.state, "classifier", None)
        app.state.brand_engine = BrandReplyEngine(classifier=classifier)
    return app.state.brand_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: pre-load SetFit model onto GPU, warm up CUDA graph, and initialize cache."""
    logger.info("Initializing SetFit inference engine...")
    classifier = get_classifier()
    cache = get_cache()

    # Pre-warm CUDA context and PyTorch graph to eliminate first-request cold start
    if classifier is not None and getattr(classifier, "_model", None) is not None:
        try:
            logger.info("Warming up CUDA context and SetFit inference graph...")
            _ = _run_model_predict_proba(classifier, ["warmup query for cuda context initialization"])
            if getattr(classifier, "device", None) == "cuda":
                torch.cuda.synchronize()
            logger.info("Model warm-up completed successfully.")
        except Exception as e:
            logger.warning(f"Model warm-up warning: {e}")

    device_name = getattr(classifier, "device", "cpu") if classifier else "cpu"
    logger.info(
        f"SetFit engine active on device: {device_name} | Taxonomy: {app.state.taxonomy}"
    )
    yield
    logger.info("Shutting down inference engine.")


app = FastAPI(
    title="Customer Support Intent Classification Service",
    description="Asynchronous, low-latency Few-Shot SetFit inference engine with Redis in-memory caching and ambiguity gating.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", tags=["Probes"])
async def liveness_probe():
    """Liveness probe: returns 200 OK if service process is running."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/ready", response_model=HealthResponse, tags=["Probes"])
async def readiness_probe():
    """Readiness probe: validates that model weights are loaded and reports Redis connectivity."""
    classifier = get_classifier()
    cache = get_cache()

    model_ready = classifier is not None and classifier.is_loaded()
    redis_ready = cache.is_redis_connected() if cache else False

    if not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded into memory.",
        )

    return HealthResponse(
        status="ready",
        model_loaded=model_ready,
        device=classifier.device,
        redis_connected=redis_ready,
        taxonomy=app.state.taxonomy,
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify_case(request: ClassifyRequest):
    """Classify a single incoming customer support tweet with Redis caching and non-blocking inference."""
    t0 = time.perf_counter()
    classifier = get_classifier()
    cache = get_cache()
    taxonomy = app.state.taxonomy

    # 1. Check normalized cache
    cached_val = cache.get(request.text)
    if cached_val is not None:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return ClassifyResponse(
            intent=cached_val["intent"],
            confidence=cached_val["confidence"],
            probabilities=cached_val["probabilities"],
            top2=[TopClassScore(**s) for s in cached_val["top2"]],
            entropy=cached_val["entropy"],
            ambiguous=cached_val["ambiguous"],
            cached=True,
            latency_ms=latency_ms,
            case_id=request.case_id,
        )

    # 2. Non-blocking GPU inference dispatched to worker threadpool
    probs = await run_in_threadpool(_run_model_predict_proba, classifier, [request.text])
    details = compute_prediction_details(probs[0], taxonomy)

    # 3. Save to cache
    cache.set(request.text, details)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return ClassifyResponse(
        intent=details["intent"],
        confidence=details["confidence"],
        probabilities=details["probabilities"],
        top2=[TopClassScore(**s) for s in details["top2"]],
        entropy=details["entropy"],
        ambiguous=details["ambiguous"],
        cached=False,
        latency_ms=latency_ms,
        case_id=request.case_id,
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse, tags=["Inference"])
async def classify_batch(request: BatchClassifyRequest):
    """Classify a stream of tweets with in-batch deduplication, Redis caching, and non-blocking GPU batching."""
    t0 = time.perf_counter()
    classifier = get_classifier()
    cache = get_cache()
    taxonomy = app.state.taxonomy

    batch_size = len(request.texts)
    case_ids = request.case_ids or [None] * batch_size

    results: List[Optional[ClassifyResponse]] = [None] * batch_size
    cache_hits = 0

    # 1. First pass: resolve from cache and identify misses
    uncached_indices: List[int] = []
    for i, text in enumerate(request.texts):
        cached_val = cache.get(text)
        if cached_val is not None:
            cache_hits += 1
            results[i] = ClassifyResponse(
                intent=cached_val["intent"],
                confidence=cached_val["confidence"],
                probabilities=cached_val["probabilities"],
                top2=[TopClassScore(**s) for s in cached_val["top2"]],
                entropy=cached_val["entropy"],
                ambiguous=cached_val["ambiguous"],
                cached=True,
                latency_ms=0.0,
                case_id=case_ids[i] if i < len(case_ids) else None,
            )
        else:
            uncached_indices.append(i)

    deduplicated_count = 0

    # 2. Second pass: deduplicate uncached queries before running GPU inference
    if uncached_indices:
        # Map normalized_key -> list of original indices
        norm_to_indices: Dict[str, List[int]] = {}
        norm_to_raw_text: Dict[str, str] = {}

        for idx in uncached_indices:
            raw = request.texts[idx]
            norm_key = normalize_cache_key(raw)
            if norm_key not in norm_to_indices:
                norm_to_indices[norm_key] = []
                norm_to_raw_text[norm_key] = raw
            norm_to_indices[norm_key].append(idx)

        unique_norm_keys = list(norm_to_indices.keys())
        unique_texts_to_infer = [norm_to_raw_text[k] for k in unique_norm_keys]
        deduplicated_count = len(uncached_indices) - len(unique_texts_to_infer)

        # Run model inference only on unique uncached texts in a single GPU batch
        probs = await run_in_threadpool(
            _run_model_predict_proba, classifier, unique_texts_to_infer
        )

        # Map predictions back to all original occurrences and populate cache
        for norm_key, prob_row in zip(unique_norm_keys, probs):
            details = compute_prediction_details(prob_row, taxonomy)
            raw_representative = norm_to_raw_text[norm_key]
            cache.set(raw_representative, details)

            for original_idx in norm_to_indices[norm_key]:
                results[original_idx] = ClassifyResponse(
                    intent=details["intent"],
                    confidence=details["confidence"],
                    probabilities=details["probabilities"],
                    top2=[TopClassScore(**s) for s in details["top2"]],
                    entropy=details["entropy"],
                    ambiguous=details["ambiguous"],
                    cached=False,
                    latency_ms=0.0,
                    case_id=case_ids[original_idx] if original_idx < len(case_ids) else None,
                )

    total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return BatchClassifyResponse(
        results=[r for r in results if r is not None],
        total_latency_ms=total_latency_ms,
        cache_hits=cache_hits,
        deduplicated_count=deduplicated_count,
        batch_size=batch_size,
    )


@app.post(
    "/draft-reply",
    response_model=DraftReplyResponse,
    tags=["Decision Engine"],
    summary="Generate grounded AppleSupport reply draft and deterministic routing decision",
)
async def draft_reply_endpoint(payload: DraftReplyRequest):
    """Process customer tweet through the full 6-stage decision pipeline."""
    start = time.perf_counter()
    engine = get_brand_reply_engine()
    case_id = payload.case_id or f"case_{int(time.time() * 1000)}"

    bundle = await run_in_threadpool(
        engine.process_case,
        customer_text=payload.text,
        customer_handle=payload.handle or "@user",
        case_id=case_id,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return DraftReplyResponse(
        case_id=bundle.case_id,
        customer_text=bundle.customer_text,
        customer_handle=bundle.customer_handle,
        intent=bundle.intent.name,
        confidence=bundle.intent.confidence,
        routing=RoutingSummary(
            decision=bundle.routing.decision,
            priority=bundle.routing.priority,
            rule_code=bundle.routing.rule_code,
            stated_reason=bundle.routing.stated_reason,
        ),
        retrieval=RetrievalSummary(
            intent=bundle.retrieval.get("intent", bundle.intent.name),
            top_similarity=bundle.retrieval.get("top_similarity", 0.0),
            confidence_zone=bundle.retrieval.get("confidence_zone", "UNKNOWN"),
            matches=[
                PrecedentItem(**m) for m in bundle.retrieval.get("matches", [])
            ],
        ),
        draft_reply=bundle.draft_reply,
        failsafe_used=bundle.failsafe_used,
        character_count=bundle.character_count,
        latency_ms=latency_ms,
    )

