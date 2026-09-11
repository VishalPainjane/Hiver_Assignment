"""Two-Stage Contextual Safety & Hazard Guardrail.

Stage 1: Rapid Regex Prescan (<0.1ms) across hazard, legal, fraud, and churn vectors.
Stage 2: Contextual Semantic Verification:
  - Unambiguous physical hazards (smoke, sparks, exploded, melted, shock) immediately trigger
    SAFETY_HAZARD_DETECTED unless explicitly negated ("no smoke", "didn't explode").
  - Ambiguous words with high colloquial usage (fire, bomb, lit, dying) undergo slang/metaphor
    disambiguation to prevent false escalations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("safety.hazard_filter")


@dataclass(frozen=True)
class SafetyAssessment:
    """Immutable result of the two-stage safety and escalation evaluation."""
    is_hazard: bool
    category: Optional[str]
    rule_code: str
    stated_reason: str
    is_slang_or_negation: bool = False


class TwoStageSafetyFilter:
    """High-precision safety filter with slang and negation disambiguation."""

    # Unambiguous physical/thermal hazard keywords (Immediate escalation if not negated)
    _UNAMBIGUOUS_HAZARDS = re.compile(
        r"\b(smoke|smoking|spark|sparks|sparking|exploded|explosion|melted|melting|burning|burned|burns|shock|shocked|shattered|cut|bleed|bleeding)\b",
        re.IGNORECASE,
    )

    # Ambiguous keywords that could be slang or physical danger
    _AMBIGUOUS_KEYWORDS = re.compile(
        r"\b(fire|bomb|lit|dying|killing)\b",
        re.IGNORECASE,
    )

    _LEGAL_KEYWORDS = re.compile(
        r"\b(lawyer|attorney|lawsuit|sue|suing|court|fcc|ftc|gdpr|police|crime|consumer protection)\b",
        re.IGNORECASE,
    )
    _SECURITY_KEYWORDS = re.compile(
        r"\b(hacked|unauthorized charge|unauthorized purchase|stolen apple id|stolen card|compromised account|sim swap)\b",
        re.IGNORECASE,
    )
    _REPEAT_CHURN_KEYWORDS = re.compile(
        r"\b(3rd time|4th time|5th time|multiple times|switching to android|useless support|never buying apple again)\b",
        re.IGNORECASE,
    )

    # Slang & Colloquial patterns that trigger false alarms on hazard words
    _SLANG_PATTERNS = [
        re.compile(r"\b(is|are|was|straight|pure|so|too)\s+(fire|the bomb|lit)\b", re.IGNORECASE),
        re.compile(r"\b(fire|bomb)\s+(update|camera|feature|phone|color|app)\b", re.IGNORECASE),
        re.compile(r"\b(dying|killing me)\s+(laughing|rn|right now)\b", re.IGNORECASE),
        re.compile(r"\b(battery|phone|device|iphone|ipad|it|charge)\s+(is\s+)?(dying|dies|dead)\b", re.IGNORECASE),
        re.compile(r"\b(dying\s+(fast|quick|rapidly|in\s+\d+|within\s+\d+|soon|again|overnight))\b", re.IGNORECASE),
        re.compile(r"\b(dumpster fire)\b", re.IGNORECASE),
    ]

    # Negation patterns preceding hazard terms
    _NEGATION_PATTERNS = [
        re.compile(r"\b(no|not|neither|without|never|didn't|did not|wasn't|was not|isn't|is not)\s+(smoke|spark|sparks|fire|explosion|melting|burn|shock)\b", re.IGNORECASE),
        re.compile(r"\b(worried|afraid|thought|feared)\b.*?\b(smoke|smoking|fire|melt|melting|burn)\b.*?\b(but|though|luckily|thankfully|just|didn't|did not)\b", re.IGNORECASE),
        re.compile(r"\b(thankfully|luckily)\s+(no|there is no|there was no)\s+(smoke|fire|sparks)\b", re.IGNORECASE),
    ]

    def __init__(self, llm_client: Optional[object] = None) -> None:
        self.llm_client = llm_client

    def evaluate(self, text: str) -> SafetyAssessment:
        """Run two-stage evaluation on incoming customer text."""
        clean = text.strip()
        if not clean:
            return SafetyAssessment(
                is_hazard=False,
                category=None,
                rule_code="SAFE_PASS",
                stated_reason="Clean empty message, no hazard detected.",
            )

        # 1. Check Legal / Regulatory Threats (Zero tolerance for public AI)
        legal_match = self._LEGAL_KEYWORDS.search(clean)
        if legal_match:
            return SafetyAssessment(
                is_hazard=True,
                category="LEGAL_PR",
                rule_code="LEGAL_PR_RISK",
                stated_reason=(
                    f"Customer cited legal or regulatory keywords ('{legal_match.group(0)}'). "
                    "Public AI reply blocked; escalated to Corporate & Legal Relations."
                ),
            )

        # 2. Check Security Compromise & Fraud
        sec_match = self._SECURITY_KEYWORDS.search(clean)
        if sec_match:
            return SafetyAssessment(
                is_hazard=True,
                category="SECURITY_FRAUD",
                rule_code="SECURITY_FRAUD_RISK",
                stated_reason=(
                    f"Potential account compromise or unauthorized charge ('{sec_match.group(0)}'). "
                    "Auto-reply suppressed; routed to Senior Fraud & Account Specialists."
                ),
            )

        # 3. Check Repeat Contact & Churn
        churn_match = self._REPEAT_CHURN_KEYWORDS.search(clean)
        if churn_match:
            return SafetyAssessment(
                is_hazard=True,
                category="REPEAT_CHURN",
                rule_code="REPEAT_CONTACT_HIGH_FRUSTRATION",
                stated_reason=(
                    f"Customer expressed severe churn or repeat failed attempts ('{churn_match.group(0)}'). "
                    "Routed to Tier-2 Retention Recovery Specialist."
                ),
            )

        # 4. Check for Contextual Negation
        is_negated = any(p.search(clean) for p in self._NEGATION_PATTERNS)
        if is_negated:
            logger.info(f"Hazard term negated in text: '{clean}'")
            return SafetyAssessment(
                is_hazard=False,
                category=None,
                rule_code="SAFE_SLANG_NEGATION_CLEARED",
                stated_reason="Contextual negation of hazard detected ('no smoke' / 'worried but did not'). Safe to continue normal triage.",
                is_slang_or_negation=True,
            )

        # 5. Check Unambiguous Physical Hardware Hazard (Sparks, Smoke, Explosion, Shock)
        hazard_match = self._UNAMBIGUOUS_HAZARDS.search(clean)
        if hazard_match:
            keyword = hazard_match.group(0)
            return SafetyAssessment(
                is_hazard=True,
                category="SAFETY_HAZARD",
                rule_code="SAFETY_HAZARD_DETECTED",
                stated_reason=(
                    f"Physical hardware safety hazard detected ('{keyword}'). "
                    "Automated reply blocked to prevent corporate liability; escalated to Hardware Safety Specialists."
                ),
            )

        # 6. Check Ambiguous Terms (fire, bomb, lit) -> Slang check
        ambiguous_match = self._AMBIGUOUS_KEYWORDS.search(clean)
        if ambiguous_match:
            keyword = ambiguous_match.group(0)
            is_slang = any(p.search(clean) for p in self._SLANG_PATTERNS)
            if is_slang:
                logger.info(f"Keyword '{keyword}' cleared as colloquial slang in: '{clean}'")
                return SafetyAssessment(
                    is_hazard=False,
                    category=None,
                    rule_code="SAFE_SLANG_NEGATION_CLEARED",
                    stated_reason=f"Keyword '{keyword}' identified as colloquial praise/slang. Safe to continue normal triage.",
                    is_slang_or_negation=True,
                )

            # If ambiguous and not caught by common slang regex, run semantic check if LLM available
            if self.llm_client and hasattr(self.llm_client, "generate"):
                try:
                    verified = self._verify_hazard_semantic(clean, keyword)
                    if not verified:
                        return SafetyAssessment(
                            is_hazard=False,
                            category=None,
                            rule_code="SAFE_SEMANTIC_VERIFIED",
                            stated_reason=f"Semantic verifier confirmed '{keyword}' is metaphorical/non-hazardous.",
                            is_slang_or_negation=True,
                        )
                except Exception as exc:
                    logger.warning(f"Semantic verifier error: {exc}; treating as hazard for safety.")

            return SafetyAssessment(
                is_hazard=True,
                category="SAFETY_HAZARD",
                rule_code="SAFETY_HAZARD_DETECTED",
                stated_reason=f"Thermal/hazard keyword '{keyword}' detected. Escalated to Hardware Safety Specialists.",
            )

        # Clean Pass
        return SafetyAssessment(
            is_hazard=False,
            category=None,
            rule_code="SAFE_PASS",
            stated_reason="No safety, legal, or severe churn risk detected.",
        )

    def _verify_hazard_semantic(self, text: str, keyword: str) -> bool:
        """Micro-prompt verifier using LLM to confirm genuine physical danger."""
        prompt = (
            f"Customer tweet: '{text}'\n"
            f"The word '{keyword}' was used. Is the customer reporting an ACTUAL physical fire, burning, explosion, or physical hazard?\n"
            "Answer with ONLY YES or NO."
        )
        resp = self.llm_client.generate(prompt=prompt, max_tokens=10, temperature=0.0)
        return "YES" in resp.upper()
