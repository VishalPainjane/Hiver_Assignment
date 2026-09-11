"""Comprehensive tests for Grounded Brand Reply Engine, Safety Filter, and Playbook Retriever."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.generation.brand_reply_generator import BrandReplyEngine
from src.retrieval.playbook_retriever import PlaybookRetriever
from src.safety.hazard_filter import TwoStageSafetyFilter


@pytest.fixture
def safety_filter():
    return TwoStageSafetyFilter()


@pytest.fixture
def playbook_retriever():
    return PlaybookRetriever()


@pytest.fixture
def brand_engine(playbook_retriever, safety_filter):
    return BrandReplyEngine(retriever=playbook_retriever, safety_filter=safety_filter)


# -------------------------------------------------------------
# 1. Safety Guardrail & Contextual Disambiguation Tests
# -------------------------------------------------------------

def test_safety_filter_unambiguous_hazard(safety_filter):
    """Verify genuine thermal and hardware hazards trigger immediate escalation."""
    res1 = safety_filter.evaluate("My phone charger started smoking and sparking!")
    assert res1.is_hazard is True
    assert res1.rule_code == "SAFETY_HAZARD_DETECTED"
    assert "smoking" in res1.stated_reason or "sparking" in res1.stated_reason

    res2 = safety_filter.evaluate("The battery swelled up and the screen shattered!")
    assert res2.is_hazard is True
    assert res2.rule_code == "SAFETY_HAZARD_DETECTED"


def test_safety_filter_slang_cleared(safety_filter):
    """Verify colloquial slang ('fire', 'the bomb') is not falsely escalated."""
    res1 = safety_filter.evaluate("This new iOS update is straight fire!")
    assert res1.is_hazard is False
    assert res1.is_slang_or_negation is True
    assert res1.rule_code == "SAFE_SLANG_NEGATION_CLEARED"

    res2 = safety_filter.evaluate("The camera on this iPhone is the bomb")
    assert res2.is_hazard is False
    assert res2.is_slang_or_negation is True


def test_safety_filter_negation_cleared(safety_filter):
    """Verify contextual negation ('no smoke', 'afraid of fire but didn't') is cleared."""
    res1 = safety_filter.evaluate("I was afraid it would start smoking, but thankfully there's no smoke.")
    assert res1.is_hazard is False
    assert res1.is_slang_or_negation is True
    assert res1.rule_code == "SAFE_SLANG_NEGATION_CLEARED"

    res2 = safety_filter.evaluate("My phone died with no sparks or smoke, just shut off.")
    assert res2.is_hazard is False
    assert res2.is_slang_or_negation is True


def test_safety_filter_legal_and_security(safety_filter):
    """Verify legal threats and account takeover trigger deterministic handoff."""
    res_legal = safety_filter.evaluate("You charged me unauthorized and I am filing a lawsuit with my lawyer.")
    assert res_legal.is_hazard is True
    assert res_legal.rule_code == "LEGAL_PR_RISK"

    res_sec = safety_filter.evaluate("Someone hacked my apple id and changed my security questions.")
    assert res_sec.is_hazard is True
    assert res_sec.rule_code == "SECURITY_FRAUD_RISK"


# -------------------------------------------------------------
# 2. Semantic Playbook Retrieval Tests
# -------------------------------------------------------------

def test_playbook_retriever_intent_partition(playbook_retriever):
    """Verify retrieval returns relevant historical customer-agent pairs."""
    res = playbook_retriever.retrieve(
        "My iPhone 7 battery is draining fast after update",
        intent="BATTERY_DRAIN",
        top_k=3,
    )
    assert res.intent == "BATTERY_DRAIN"
    assert len(res.matches) == 3
    assert res.top_similarity > 0.15
    for m in res.matches:
        assert m.intent == "BATTERY_DRAIN"
        assert "{customer_handle}" in m.apple_reply or "DM" in m.apple_reply


# -------------------------------------------------------------
# 3. Decision Engine & 3-Tier Routing Tests
# -------------------------------------------------------------

def test_brand_engine_hazard_handoff(brand_engine):
    """Verify physical hazard produces HANDOFF decision and suppresses draft."""
    bundle = brand_engine.process_case(
        customer_text="My charger sparked and is smoking!",
        customer_handle="@danger_user",
        case_id="case_haz_01",
    )
    assert bundle.routing.decision == "HANDOFF"
    assert bundle.routing.rule_code == "SAFETY_HAZARD_DETECTED"
    assert bundle.routing.priority == "URGENT_P1"
    assert bundle.draft_reply is None


def test_brand_engine_routine_battery(brand_engine):
    """Verify routine battery inquiry generates grounded reply with Apple guidelines."""
    bundle = brand_engine.process_case(
        customer_text="My iPhone 7 battery is dying within 2 hours after updating to iOS 11",
        customer_handle="@battery_user",
        case_id="case_bat_01",
    )
    assert bundle.routing.decision in ("AUTO_REPLY", "ASSISTED_REPLY")
    assert bundle.draft_reply is not None
    assert bundle.character_count <= 280
    assert bundle.character_count > 20
    assert "@battery_user" in bundle.draft_reply
    # Must contain DM pivot
    assert "dm" in bundle.draft_reply.lower() or "twitter.com" in bundle.draft_reply


def test_brand_engine_zero_downtime_failsafe(brand_engine):
    """Verify system uses historical template failsafe if LLM times out or errors."""
    # Mock LLM client to simulate exception
    failing_llm = MagicMock()
    failing_llm.generate.side_effect = TimeoutError("Groq API Timeout")

    engine_with_failing_llm = BrandReplyEngine(
        retriever=brand_engine.retriever,
        safety_filter=brand_engine.safety_filter,
        llm_client=failing_llm,
    )

    bundle = engine_with_failing_llm.process_case(
        customer_text="My iPhone battery drains so fast after iOS 11",
        customer_handle="@failsafe_user",
        case_id="case_fail_01",
    )

    assert bundle.routing.decision in ("AUTO_REPLY", "ASSISTED_REPLY")
    assert bundle.draft_reply is not None
    assert bundle.failsafe_used is True
    assert "@failsafe_user" in bundle.draft_reply
    assert len(bundle.draft_reply) <= 280


def test_api_draft_reply_endpoint():
    """Verify FastAPI /draft-reply endpoint returns complete decision bundle."""
    from src.service.api import app
    client = TestClient(app)

    # 1. Normal battery inquiry
    payload = {
        "text": "My iPhone battery dies within 2 hours and gets warm",
        "handle": "@alex_apple",
        "case_id": "test_api_01",
    }
    res = client.post("/draft-reply", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "test_api_01"
    assert data["customer_handle"] == "@alex_apple"
    assert data["routing"]["decision"] in ("AUTO_REPLY", "ASSISTED_REPLY")
    assert "stated_reason" in data["routing"]
    assert data["draft_reply"] is not None
    assert data["character_count"] <= 280

    # 2. Safety hazard inquiry
    haz_payload = {
        "text": "Help! My charger sparked and started smoking violently!",
        "handle": "@urgent_user",
        "case_id": "test_api_haz",
    }
    haz_res = client.post("/draft-reply", json=haz_payload)
    assert haz_res.status_code == 200
    haz_data = haz_res.json()
    assert haz_data["routing"]["decision"] == "HANDOFF"
    assert haz_data["routing"]["rule_code"] == "SAFETY_HAZARD_DETECTED"
    assert haz_data["draft_reply"] is None

