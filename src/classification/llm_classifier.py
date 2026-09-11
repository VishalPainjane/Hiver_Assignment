"""Strict, Fault-Tolerant Local LLM Intent Classifier using Ollama (Llama 3.2 on GPU)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, Field, ValidationError

from src.domain.models import SupportCase

logger = logging.getLogger("classification.llm")

# Exact closed taxonomy
# Exact closed taxonomy (6 intents)
VALID_INTENTS = (
    "BATTERY_DRAIN",
    "DEVICE_BOOT",
    "ACCOUNT_ACCESS",
    "BILLING_DISPUTE",
    "UPDATE_FAILURE",
    "HARDWARE_DAMAGE",
    "CONNECTIVITY_ISSUE",
    "UNKNOWN",
)

IntentType = Literal[
    "BATTERY_DRAIN",
    "DEVICE_BOOT",
    "ACCOUNT_ACCESS",
    "BILLING_DISPUTE",
    "UPDATE_FAILURE",
    "HARDWARE_DAMAGE",
    "CONNECTIVITY_ISSUE",
    "UNKNOWN",
]


class StrictIntentPayload(BaseModel):
    """Pydantic model enforcing rigorous schema and bounds on LLM outputs."""

    primary_intent: IntentType
    requested_action: str = Field(default="INQUIRY")
    issue_state: str = Field(default="INFORMATIONAL")
    urgency: str = Field(default="LOW")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=500)


SYSTEM_PROMPT = """You are an expert customer-support classifier for @AppleSupport.
Classify the customer's incoming issue into EXACTLY ONE of these 8 intents:
Classify the customer's incoming issue into EXACTLY ONE of these 6 intents:

1. BATTERY_DRAIN:
   - Rapid battery drop, battery health percentage, won't hold charge, charger cord position/cable issues, heating while charging.
   - Trick case: "Charged 9 times today" = BATTERY_DRAIN (charge cycles, not billing!).

2. DEVICE_BOOT:
   - Frozen screen, won't turn on, black screen, restart loop, stuck on Apple logo, hard restart required.

3. ACCOUNT_ACCESS:
   - Apple ID password reset, account locked for security, 2FA prompt, iMessage/iCloud sign-in failure.
   - Trick case: "I did not forget my password, but account is locked" = ACCOUNT_ACCESS.

4. BILLING_DISPUTE:
   - Unexpected App Store or iTunes charges, subscription cancellations, refund requests, payment method issues.

5. UPDATE_FAILURE:
   - iOS/macOS installation failed, update download error, insufficient storage for update.

6. HARDWARE_DAMAGE:
   - Physical cracked screen, shattered glass, dropped in water, broken speaker/microphone, physical button broken.
6. UNKNOWN:
   - Physical hardware damage (cracked screen, water damage, broken button/speaker), connectivity drops (Wi-Fi, Bluetooth pairing), physical hazards (smoke, fire, explosion, injury), legal threats (sue, lawyer, lawsuit), insults, foreign languages, or vague chatter.

7. CONNECTIVITY_ISSUE:
   - Wi-Fi dropping, Bluetooth failing to pair with car or AirPods, cellular "No Service" or dropped calls.

8. UNKNOWN:
   - Physical hazard (smoke, fire, explosion, injury), legal threats (sue, lawyer, lawsuit), insults, foreign languages, or vague chatter.

RULES:
- Never hallucinate new intents. You MUST pick from the 8 above.
- Never hallucinate new intents. You MUST pick from the 6 above.
- If the customer mentions fire, smoke, explosion, injury, or lawsuit, choose UNKNOWN with urgency HIGH.
- If the issue is ambiguous or vague, choose UNKNOWN with confidence 0.35.
- If the issue is ambiguous, vague, or out of scope, choose UNKNOWN with confidence 0.35.
- Output ONLY valid JSON matching this schema:
{
  "primary_intent": "...",
  "requested_action": "TROUBLESHOOTING" | "REFUND" | "ACCOUNT_RECOVERY" | "REPLACEMENT" | "INQUIRY",
  "issue_state": "BLOCKED" | "INCONVENIENCED" | "INFORMATIONAL",
  "urgency": "LOW" | "MEDIUM" | "HIGH",
  "confidence": 0.0 to 1.0,
  "reasoning": "1 sentence explanation"
}"""


class LocalLLMClassifier:
    """Fault-tolerant classifier with Pydantic validation and persistent caching."""

    HIGH_RISK_TERMS = frozenset(
        {"lawyer", "attorney", "sue", "lawsuit", "data breach", "gdpr", "fire", "smoke", "explosion", "injury"}
    )

    def __init__(
        self,
        model: str = "llama3.2:latest",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 15.0,
        cache_path: str = "outputs/llm_intent_cache.json",
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_path.exists():
            try:
                with self.cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read intent cache: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write intent cache: {e}")

    def _cache_key(self, case: SupportCase) -> str:
        text_hash = hashlib.sha256(case.clean_text.encode("utf-8")).hexdigest()[:16]
        return f"{case.case_id}_{text_hash}"

    def classify(self, case: SupportCase, force_refresh: bool = False) -> Dict[str, Any]:
        """Classify a support case with caching, safety overrides, and strict Pydantic validation."""

        # 1. Deterministic safety override check
        normalized = case.clean_text.lower()
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized) for term in self.HIGH_RISK_TERMS):
            return {
                "primary_intent": "UNKNOWN",
                "requested_action": "SAFETY_ESCALATION",
                "issue_state": "BLOCKED",
                "urgency": "HIGH",
                "confidence": 0.99,
                "reasoning": "High-risk safety or legal term detected; mandatory escalation to human.",
            }

        # 2. Check cache
        """Classify a support case using model inference with persistent caching and strict validation."""
        # 1. Check cache
        key = self._cache_key(case)
        if not force_refresh and key in self._cache:
            return self._cache[key]

        # 3. Build multi-turn prompt
        conversation_context = ""
        if len(case.raw_turns) > 1:
            turns_text = "\n".join(
                f"- {turn.get('author_id', 'user')}: {turn.get('text', '')}"
                for turn in case.raw_turns[:-1]
            )
            conversation_context = f"Preceding Conversation:\n{turns_text}\n\n"

        prompt = (
            f"{conversation_context}Incoming Customer Message:\n"
            f'"{case.clean_text}"\n\n'
            "Classify according to the rules and return JSON:"
        )

        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 130},
        }

        # 4. Call Local LLM on GPU
        # 4. Call Local LLM on GPU (with automatic failover to Unified client)
        raw_output = "{}"
        try:
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_output = data.get("response", "{}")
        except Exception as exc:
            logger.warning(f"Local LLM call failed: {exc}")
            raise
            logger.warning(f"Local LLM call failed: {exc}; attempting failover to cloud provider...")
            try:
                from src.llm.client import UnifiedLLMClient
                client = UnifiedLLMClient(primary="groq")
                raw_output = client.generate(
                    prompt=prompt,
                    system_instruction=SYSTEM_PROMPT,
                    json_mode=True,
                    temperature=0.0,
                    max_tokens=130,
                )
            except Exception as e2:
                logger.error(f"Cloud failover also failed: {e2}")
                raise

        # 5. Fault-tolerant JSON extraction & Pydantic validation
        result = self._parse_and_validate(raw_output)

        # 6. Save to cache
        self._cache[key] = result
        self._save_cache()

        return result

    def _parse_and_validate(self, raw_text: str) -> Dict[str, Any]:
        """Extract JSON and validate against strict schema, recovering from hallucinations."""
        data: Dict[str, Any] = {}
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        # Validate intent name against closed taxonomy
        intent = str(data.get("primary_intent", "UNKNOWN")).upper().strip()
        if intent not in VALID_INTENTS:
            logger.warning(f"LLM hallucinated invalid intent '{intent}'; coerced to UNKNOWN")
            intent = "UNKNOWN"
            data["confidence"] = min(float(data.get("confidence", 0.35)), 0.35)

        data["primary_intent"] = intent

        # Validate via Pydantic model
        try:
            validated = StrictIntentPayload(**data)
            return validated.model_dump()
        except ValidationError as err:
            logger.warning(f"Schema validation error: {err}; applying safe fallback values")
            return {
                "primary_intent": intent if intent in VALID_INTENTS else "UNKNOWN",
                "requested_action": str(data.get("requested_action", "INQUIRY")).upper(),
                "issue_state": str(data.get("issue_state", "INFORMATIONAL")).upper(),
                "urgency": str(data.get("urgency", "LOW")).upper(),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.40)))),
                "reasoning": str(data.get("reasoning", "Recovered from schema validation fallback")),
            }
