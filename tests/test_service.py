"""Automated tests for FastAPI inference microservice, Redis caching, and ambiguity gating."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.service.api import app, get_classifier


class MockSetFitModel:
    def __init__(self, labels):
        self.labels = list(labels)

    def predict_proba(self, texts):
        probs = []
        for text in texts:
            t = text.lower()
            row = [0.01] * len(self.labels)
            if "battery" in t or "dies" in t:
                idx = self.labels.index("BATTERY_DRAIN")
            elif "billing" in t or "charge" in t or "card" in t or "itunes" in t:
                idx = self.labels.index("BILLING_DISPUTE")
            elif "update" in t:
                idx = self.labels.index("UPDATE_FAILURE")
            else:
                idx = self.labels.index("UNKNOWN")
            row[idx] = 0.95
            s = sum(row)
            probs.append([v / s for v in row])
        return np.array(probs)


class MockClassifier:
    def __init__(self):
        self.taxonomy = [
            "ACCOUNT_ACCESS",
            "BATTERY_DRAIN",
            "BILLING_DISPUTE",
            "DEVICE_BOOT",
            "UNKNOWN",
            "UPDATE_FAILURE",
        ]
        self.device = "cpu"
        self._model = MockSetFitModel(self.taxonomy)

    def is_loaded(self) -> bool:
        return True


@pytest.fixture(scope="module", autouse=True)
def setup_classifier():
    """Ensure an active classifier is available during test runs (e.g. in headless CI)."""
    classifier = get_classifier()
    if classifier is None or not classifier.is_loaded() or getattr(classifier, "_model", None) is None:
        mock_clf = MockClassifier()
        app.state.classifier = mock_clf
        app.state.taxonomy = mock_clf.taxonomy
        app.state.brand_engine = None


@pytest.fixture(scope="module")
def client(setup_classifier):
    """Module-scoped FastAPI TestClient with initialized lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_healthz_liveness(client: TestClient):
    """Verify /healthz returns 200 OK and live process status."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_ready_readiness(client: TestClient):
    """Verify /ready returns 200 OK with model loaded and active taxonomy."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "BATTERY_DRAIN" in data["taxonomy"]
    assert "ACCOUNT_ACCESS" in data["taxonomy"]


def test_classify_inference_and_cache_hit(client: TestClient):
    """Verify single query inference, text normalization, and subsequent sub-ms cache hit."""
    raw_query = "My iPhone battery dies within 2 hours and gets super hot"
    payload = {"text": raw_query, "case_id": "test_001"}

    # 1. Initial request -> Cache Miss
    res1 = client.post("/classify", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["intent"] == "BATTERY_DRAIN"
    assert data1["confidence"] > 0.80
    assert data1["cached"] is False
    assert len(data1["top2"]) == 2
    assert "entropy" in data1
    assert "ambiguous" in data1
    assert data1["case_id"] == "test_001"

    # 2. Subsequent request with whitespace/casing difference -> Cache Hit
    variant_query = "  my iphone battery dies within 2 hours and gets super hot   "
    res2 = client.post("/classify", json={"text": variant_query})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["intent"] == "BATTERY_DRAIN"
    assert data2["cached"] is True
    assert data2["latency_ms"] < 5.0


def test_classify_batch_deduplication_and_caching(client: TestClient):
    """Verify batch streaming endpoint with in-batch deduplication and caching."""
    queries = [
        "I was charged $9.99 on my iTunes card for an unauthorized purchase",
        "I was charged $9.99 on my iTunes card for an unauthorized purchase",  # duplicate
        "Software update to iOS 17 failed with installation error",
    ]
    payload = {"texts": queries, "case_ids": ["c1", "c2", "c3"]}

    # First batch pass: has 1 duplicate to deduplicate during inference
    res = client.post("/classify/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["batch_size"] == 3
    assert len(data["results"]) == 3
    assert data["deduplicated_count"] >= 1
    assert data["results"][0]["intent"] == "BILLING_DISPUTE"
    assert data["results"][1]["intent"] == "BILLING_DISPUTE"
    assert data["results"][2]["intent"] == "UPDATE_FAILURE"

    # Second batch pass: all items should hit cache
    res_cached = client.post("/classify/batch", json=payload)
    assert res_cached.status_code == 200
    data_cached = res_cached.json()
    assert data_cached["cache_hits"] == 3
    assert all(r["cached"] is True for r in data_cached["results"])


def test_payload_validation_constraints(client: TestClient):
    """Verify Pydantic validation rejects malformed, empty, and oversized payloads."""
    # Empty string (violates min_length=2)
    res_empty = client.post("/classify", json={"text": " "})
    assert res_empty.status_code == 422

    # Single char (violates min_length=2)
    res_short = client.post("/classify", json={"text": "x"})
    assert res_short.status_code == 422

    # Oversized payload (> 500 characters)
    oversized = "a" * 501
    res_long = client.post("/classify", json={"text": oversized})
    assert res_long.status_code == 422

    # Empty batch list (violates min_length=1)
    res_empty_batch = client.post("/classify/batch", json={"texts": []})
    assert res_empty_batch.status_code == 422
