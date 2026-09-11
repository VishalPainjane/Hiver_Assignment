"""Comprehensive integration test suite for the FastAPI inference microservice."""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from src.service.api import app, lifespan


async def run_suite():
    print("\n" + "=" * 80)
    print("RUNNING FASTAPI MICROSERVICE INTEGRATION TESTS")
    print("=" * 80)

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Liveness Probe
            print("\n[TEST 1] GET /healthz (Liveness Probe)")
            r = await client.get("/healthz")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data["status"] == "ok", f"Expected ok, got {data['status']}"
            print(f"  Passed: status={data['status']}, timestamp={data['timestamp']}")

            # 2. Readiness Probe
            print("\n[TEST 2] GET /ready (Readiness Probe)")
            r = await client.get("/ready")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data["status"] == "ready"
            assert data["model_loaded"] is True
            assert len(data["taxonomy"]) == 6
            print(f"  Passed: status={data['status']}, device={data['device']}, redis_connected={data['redis_connected']}")
            print(f"            taxonomy={data['taxonomy']}")

            # 3. Single Item Cache Miss (GPU Inference - Pre-Warmed)
            print("\n[TEST 3] POST /classify (Cache Miss -> Pre-Warmed GPU Inference)")
            raw_text = "My iPhone battery dies within 2 hours of charging and feels scorching hot"
            payload = {"text": raw_text, "case_id": "case_101"}
            r1 = await client.post("/classify", json=payload)
            assert r1.status_code == 200, f"Expected 200, got {r1.status_code}"
            d1 = r1.json()
            assert d1["intent"] == "BATTERY_DRAIN", f"Expected BATTERY_DRAIN, got {d1['intent']}"
            assert d1["confidence"] > 0.80
            assert d1["cached"] is False
            assert len(d1["top2"]) == 2
            assert "entropy" in d1
            assert "ambiguous" in d1
            assert d1["case_id"] == "case_101"
            print(f"  Passed: intent={d1['intent']} (confidence={d1['confidence']*100:.1f}%), cached={d1['cached']}")
            print(f"            top2={d1['top2']}, entropy={d1['entropy']}, latency={d1['latency_ms']} ms")

            # 4. Normalized Cache Hit (Sub-millisecond Return)
            print("\n[TEST 4] POST /classify (Normalized Cache Hit -> Bypasses GPU)")
            variant_text = "   my iphone battery dies within 2 hours of charging and feels scorching hot   "
            r2 = await client.post("/classify", json={"text": variant_text})
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["intent"] == "BATTERY_DRAIN"
            assert d2["cached"] is True
            assert d2["latency_ms"] < 5.0
            print(f"  Passed: intent={d2['intent']}, cached={d2['cached']}, latency={d2['latency_ms']} ms (< 5 ms!)")

            # 5. Batch Streaming with Deduplication
            print("\n[TEST 5] POST /classify/batch (Batch Stream with In-Batch Deduplication)")
            batch_queries = [
                "I was charged $14.99 on my iTunes receipt for an unauthorized in-app purchase",
                "I was charged $14.99 on my iTunes receipt for an unauthorized in-app purchase",
                "Software update to iOS 17 failed with unable to verify update installation error",
            ]
            batch_payload = {"texts": batch_queries, "case_ids": ["c1", "c2", "c3"]}
            rb = await client.post("/classify/batch", json=batch_payload)
            assert rb.status_code == 200
            db = rb.json()
            assert db["batch_size"] == 3
            assert db["deduplicated_count"] >= 1, f"Expected deduplication >= 1, got {db['deduplicated_count']}"
            assert db["results"][0]["intent"] == "BILLING_DISPUTE"
            assert db["results"][1]["intent"] == "BILLING_DISPUTE"
            assert db["results"][2]["intent"] == "UPDATE_FAILURE"
            print(f"  Passed: batch_size={db['batch_size']}, deduplicated={db['deduplicated_count']}, total_latency={db['total_latency_ms']} ms")
            for i, res in enumerate(db["results"]):
                print(f"            Item {i}: intent={res['intent']}, conf={res['confidence']*100:.1f}%, cached={res['cached']}")

            # 6. Batch Cache Hit
            print("\n[TEST 6] POST /classify/batch (Full Batch Cache Hit)")
            rb2 = await client.post("/classify/batch", json=batch_payload)
            assert rb2.status_code == 200
            db2 = rb2.json()
            assert db2["cache_hits"] == 3
            assert all(r["cached"] is True for r in db2["results"])
            print(f"  Passed: cache_hits={db2['cache_hits']}/3, total_latency={db2['total_latency_ms']} ms")

            # 7. Payload Validation Constraints
            print("\n[TEST 7] Pydantic Payload Constraints (422 Error Handling)")
            r_empty = await client.post("/classify", json={"text": " "})
            assert r_empty.status_code == 422
            r_short = await client.post("/classify", json={"text": "x"})
            assert r_short.status_code == 422
            r_long = await client.post("/classify", json={"text": "x" * 501})
            assert r_long.status_code == 422
            r_batch_empty = await client.post("/classify/batch", json={"texts": []})
            assert r_batch_empty.status_code == 422
            print("  Passed: correctly rejected whitespace, single-char, oversized (>500), and empty batch payloads with 422.")

    print("\n" + "=" * 80)
    print("ALL 7 SERVICE INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_suite())
