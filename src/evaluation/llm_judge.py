"""LLM-as-a-Judge for Apple Customer Support Tweet Quality.

Evaluates support replies on Apple's core operational dimensions:
1. Diagnostic Probe (1-5)
2. Privacy Pivot (1-5)
3. Zero-Liability Shield (1-5)
4. Overall Quality Score (1-5)
5. Rationale (1-2 sentences)

Calibrated with grounded factual verification and neuro-symbolic guardrails.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from src.llm.client import UnifiedLLMClient

logger = logging.getLogger("evaluation.llm_judge")

PROMISSORY_REGEX = re.compile(
    r"\b(we (can|will|do|shall) guarantee|we will replace|we'll replace|free of charge|we will refund|we can refund|we promise to fix|100% guarantee|definitely a hardware fault|free repair|guaranteed fix|free replacement)\b",
    re.IGNORECASE,
)
DM_LINK_REGEX = re.compile(
    r"(https?://\S+|twitter\.com/messages|t\.co/|apple\.com|reportaproblem\.apple\.com|getsupport\.apple\.com)",
    re.IGNORECASE,
)


@dataclass
class JudgeEvaluation:
    diagnostic_probe: int
    privacy_pivot: int
    zero_liability: int
    overall_score: int
    rationale: str
    char_length: int
    length_compliant: bool
    has_dm_link: bool
    zero_liability_compliant: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMReplyJudge:
    """Evaluates customer support tweet replies using an anchored LLM rubric."""

    SYSTEM_PROMPT = (
        "You are an expert Quality Assurance Lead auditing customer support replies for @AppleSupport on Twitter/X.\n"
        "Your role is to strictly and objectively grade the generated reply using the provided factual checks.\n"
        "Return ONLY a valid JSON object matching the exact schema requested."
    )

    RUBRIC_TEMPLATE = """You are evaluating an automated reply to a customer tweet for @AppleSupport on Twitter.

FACTUAL CHECKS VERIFIED BY SCANNER:
- Official DM Link Present: {has_dm_link}
- Inquires About iOS Version: {asks_ios_version}
- Inquires About Device Model: {asks_device_model}
- Character Count: {char_len} (Twitter Limit: 280)
- Promissory Language Detected: {promissory_detected}

CALIBRATION ANCHORS & RULES:
- If Promissory Language is TRUE -> zero_liability is 1; overall_score MUST be 1.
- If Official DM Link is FALSE -> privacy_pivot is at most 3; overall_score cannot exceed 3.
- If Generic Canned Template asks for 'device details' on billing/2FA/compliment -> overall_score MUST be 1 or 2.
- If the reply has DM Link + Inquires About Version/Model + Zero Promissory Language -> overall_score is 5.

Scoring Scale (1 to 5):
1: Severe liability breach (promises free hardware/refund), severe tone-deafness, or safety hazard mishandled.
2: Poor / unhelpful canned response (e.g. asking device details for credit card charge).
3: Mediocre canned response lacking direct link or technical guidance.
4: Helpful and safe troubleshooting, but minor omission.
5: Exceptional Apple brand response: diagnostic probe, direct DM link, concise (<240 chars), zero liability.

CASE TO EVALUATE:
Customer Tweet: "{customer_tweet}"
Customer Intent: {intent}
Generated Reply: "{generated_reply}"

Return ONLY valid JSON:
{{
  "diagnostic_probe": <integer 1-5>,
  "privacy_pivot": <integer 1-5>,
  "zero_liability": <integer 1-5>,
  "overall_score": <integer 1-5>,
  "rationale": "<1-2 sentences explaining score>"
}}"""

    def __init__(self, client: Optional[UnifiedLLMClient] = None) -> None:
        self.client = client or UnifiedLLMClient(primary="ollama")

    def evaluate(
        self,
        customer_tweet: str,
        generated_reply: str,
        intent: str = "UNKNOWN",
    ) -> JudgeEvaluation:
        char_len = len(generated_reply)
        length_compliant = char_len <= 280
        has_dm_link = bool(DM_LINK_REGEX.search(generated_reply))
        zero_liability_compliant = not bool(PROMISSORY_REGEX.search(generated_reply))

        asks_ios_version = bool(re.search(r"\b(ios\s*\d+|version|update|os)\b", generated_reply, re.IGNORECASE))
        asks_device_model = bool(re.search(r"\b(model|iphone|ipad|watch|mac|which device)\b", generated_reply, re.IGNORECASE))

        prompt = self.RUBRIC_TEMPLATE.format(
            customer_tweet=customer_tweet,
            intent=intent,
            generated_reply=generated_reply,
            has_dm_link=has_dm_link,
            asks_ios_version=asks_ios_version,
            asks_device_model=asks_device_model,
            char_len=char_len,
            promissory_detected=not zero_liability_compliant,
        )

        try:
            raw_response = self.client.generate(
                prompt=prompt,
                system_instruction=self.SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.0,
                max_tokens=300,
            )
            parsed = self._extract_json(raw_response)
            overall = int(max(1, min(5, parsed.get("overall_score", 3))))
            diag = int(max(1, min(5, parsed.get("diagnostic_probe", 3))))
            priv = int(max(1, min(5, parsed.get("privacy_pivot", 3))))
            liab = int(max(1, min(5, parsed.get("zero_liability", 3))))
            rationale = str(parsed.get("rationale", "")).strip()

            # Neuro-Symbolic Calibration Safeguards:
            # 1. Zero Liability Alignment
            if not zero_liability_compliant:
                overall = 1
                liab = 1
                rationale = f"Severe liability violation: contains unauthorized promissory guarantee. {rationale}"
            else:
                liab = 5  # Verified zero promissory language

            # 2. Privacy Link Alignment
            if not has_dm_link:
                priv = min(priv, 3)
                if overall > 3:
                    overall = 3
            else:
                priv = max(priv, 4)

            # 3. Canned Template Handling
            is_canned = "sorry to hear you are having trouble" in generated_reply.lower() and "device details" in generated_reply.lower()
            if is_canned:
                if intent in ("BILLING_DISPUTE", "ACCOUNT_ACCESS") or "love" in customer_tweet.lower() or "smoke" in customer_tweet.lower():
                    overall = min(overall, 2)
                else:
                    overall = min(overall, 3)

            # 4. Production Grounded RAG Excellence
            if has_dm_link and zero_liability_compliant and (asks_ios_version or asks_device_model or "reportaproblem" in generated_reply.lower() or "iforgot" in generated_reply.lower()):
                overall = 5 if char_len <= 240 else 4
                diag = max(diag, 4)
                if not rationale:
                    rationale = "High-quality Apple support response with diagnostic probe, official link, and non-promissory tone."

            return JudgeEvaluation(
                diagnostic_probe=diag,
                privacy_pivot=priv,
                zero_liability=liab,
                overall_score=overall,
                rationale=rationale,
                char_length=char_len,
                length_compliant=length_compliant,
                has_dm_link=has_dm_link,
                zero_liability_compliant=zero_liability_compliant,
            )
        except Exception as e:
            logger.warning(f"LLM Judge call failed: {e}; falling back to calibrated heuristic.")
            return self._heuristic_fallback(customer_tweet, generated_reply, intent)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    def _heuristic_fallback(
        self,
        customer_tweet: str,
        generated_reply: str,
        intent: str,
    ) -> JudgeEvaluation:
        char_len = len(generated_reply)
        length_compliant = char_len <= 280
        has_dm_link = bool(DM_LINK_REGEX.search(generated_reply))
        zero_liability_compliant = not bool(PROMISSORY_REGEX.search(generated_reply))

        cust_lower = customer_tweet.lower()
        rep_lower = generated_reply.lower()

        has_device = any(w in cust_lower for w in ["iphone", "ipad", "macbook", "watch", "ios"])
        asks_diag = any(w in rep_lower for w in ["ios", "version", "model", "which device"])
        diag_score = 5 if (has_device or asks_diag) else 1

        has_dm = "dm" in rep_lower or "direct message" in rep_lower
        priv_score = 5 if (has_dm and has_dm_link) else (3 if has_dm else 1)
        liab_score = 5 if zero_liability_compliant else 1

        if not zero_liability_compliant:
            overall = 1
            rationale = "Severe brand compliance violation: contains unauthorized promissory guarantee."
        elif not length_compliant:
            overall = 2
            rationale = "Exceeds Twitter 280 character limit."
        elif rep_lower.count("sorry to hear") and "device details" in rep_lower and intent == "BILLING_DISPUTE":
            overall = 2
            rationale = "Generic canned template inappropriately requests device details for a billing dispute."
        elif diag_score == 5 and priv_score >= 3 and liab_score == 5:
            overall = 5 if priv_score == 5 and char_len <= 240 else 4
            rationale = "Well-structured response adhering to Apple brand and diagnostic guidelines."
        else:
            overall = 3
            rationale = "Mediocre response missing diagnostic probe or official DM link."

        return JudgeEvaluation(
            diagnostic_probe=diag_score,
            privacy_pivot=priv_score,
            zero_liability=liab_score,
            overall_score=overall,
            rationale=rationale,
            char_length=char_len,
            length_compliant=length_compliant,
            has_dm_link=has_dm_link,
            zero_liability_compliant=zero_liability_compliant,
        )
