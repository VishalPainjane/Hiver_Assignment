"""End-to-End Brand Reply & Routing Decision Engine for @AppleSupport.

Industrial Architecture:
1. Two-Stage Safety & Contextual Hazard Verification (Slang & Negation disambiguation).
2. Intent Classification via SetFit / Bounded NLU Analyzer.
3. Intent-Partitioned Customer-to-Customer Precedent Retrieval with 3-Zone Confidence Gating.
4. Deterministic 3-Tier Routing Decision (AUTO_REPLY, ASSISTED_REPLY, HANDOFF) with auditable stated reasons.
5. Guarded LLM Generation constrained by Apple's 4 Golden Rules:
   - Diagnostic Probe (ask for iOS / device if missing)
   - Privacy Pivot (DM redirect)
   - Zero-Liability Shield (non-promissory)
   - Brevity (< 280 chars)
6. Zero-Downtime Failsafe fallback using #1 historical template upon LLM timeout or policy violation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.client import UnifiedLLMClient
from src.retrieval.playbook_retriever import PlaybookRetriever, PrecedentMatch, RetrievalResult
from src.safety.hazard_filter import SafetyAssessment, TwoStageSafetyFilter

logger = logging.getLogger("generation.brand_reply")

# Forbidden promissory phrases
PROMISSORY_PATTERNS = re.compile(
    r"\b(we will replace|we guarantee|free of charge|we will refund|we promise to fix|100% guarantee|definitely a hardware fault)\b",
    re.IGNORECASE,
)

INTENT_CUES: Dict[str, List[str]] = {
    "ACCOUNT_ACCESS": [
        "apple id", "password", "sign in", "login", "locked", "verification", "2fa",
        "two-factor", "icloud", "recover", "passcode", "security question", "disabled"
    ],
    "BATTERY_DRAIN": [
        "battery", "drain", "charge", "charger", "charging", "overheat", "dying",
        "battery health", "percentage", "drops from", "cable", "hot", "dies"
    ],
    "BILLING_DISPUTE": [
        "charged", "charge", "refund", "subscription", "itunes", "app store", "receipt",
        "billing", "payment", "bank", "invoice", "unauthorized", "cancel", "money"
    ],
    "DEVICE_BOOT": [
        "turn on", "won't turn", "black screen", "frozen", "freeze", "stuck on apple logo",
        "reboot", "restart", "boot loop", "crash", "white screen", "hard reset"
    ],
    "UPDATE_FAILURE": [
        "ios update", "update failed", "unable to verify update", "error occurred installing",
        "storage full update", "ios 11", "downloading update", "software update", "installing update"
    ],
    "UNKNOWN": [
        "store", "appointment", "hours", "macbook", "airpods", "trade in", "repair",
        "genius bar", "case", "color", "shipping", "delivery", "broken glass"
    ],
}


@dataclass(frozen=True)
class IntentAssessment:
    name: str
    confidence: float


@dataclass(frozen=True)
class RoutingDecision:
    decision: str  # "AUTO_REPLY" | "ASSISTED_REPLY" | "HANDOFF"
    priority: str  # "NORMAL_P3" | "PRIORITY_P2" | "URGENT_P1"
    rule_code: str
    stated_reason: str


@dataclass(frozen=True)
class DecisionBundle:
    case_id: str
    customer_text: str
    customer_handle: str
    intent: IntentAssessment
    retrieval: Dict[str, Any]
    routing: RoutingDecision
    draft_reply: Optional[str]
    failsafe_used: bool
    character_count: int = 0


class BrandReplyEngine:
    """Enterprise-grade Decision Engine for grounded customer support reply generation."""

    DM_LINK_DEFAULT = "https://twitter.com/messages/compose"

    def __init__(
        self,
        retriever: Optional[PlaybookRetriever] = None,
        safety_filter: Optional[TwoStageSafetyFilter] = None,
        llm_client: Optional[UnifiedLLMClient] = None,
        classifier: Optional[Any] = None,
    ) -> None:
        self.retriever = retriever or PlaybookRetriever()
        self.llm_client = llm_client or UnifiedLLMClient()
        self.safety_filter = safety_filter or TwoStageSafetyFilter(llm_client=self.llm_client)
        self.classifier = classifier

    def process_case(
        self,
        customer_text: str,
        customer_handle: str = "@user",
        case_id: str = "case_001",
    ) -> DecisionBundle:
        """Process an incoming tweet through the complete 6-stage decision pipeline."""
        clean_text = customer_text.strip()
        handle = customer_handle if customer_handle.startswith("@") else f"@{customer_handle}"

        # -------------------------------------------------------------
        # STAGE 1: Two-Stage Safety & Contextual Hazard Verification
        # -------------------------------------------------------------
        safety = self.safety_filter.evaluate(clean_text)
        if safety.is_hazard:
            routing = RoutingDecision(
                decision="HANDOFF",
                priority="URGENT_P1" if safety.category == "SAFETY_HAZARD" else "PRIORITY_P2",
                rule_code=safety.rule_code,
                stated_reason=safety.stated_reason,
            )
            return DecisionBundle(
                case_id=case_id,
                customer_text=clean_text,
                customer_handle=handle,
                intent=IntentAssessment(name="UNKNOWN", confidence=0.0),
                retrieval={"matches": [], "top_similarity": 0.0, "confidence_zone": "SAFETY_BLOCKED"},
                routing=routing,
                draft_reply=None,
                failsafe_used=False,
                character_count=0,
            )

        # -------------------------------------------------------------
        # STAGE 2: Intent Classification
        # -------------------------------------------------------------
        intent_name, intent_conf = self._classify_intent(clean_text)
        intent_assessment = IntentAssessment(name=intent_name, confidence=intent_conf)

        # -------------------------------------------------------------
        # STAGE 3: Semantic Precedent Retrieval
        # -------------------------------------------------------------
        retrieval = self.retriever.retrieve(clean_text, intent=intent_name, top_k=3)

        # -------------------------------------------------------------
        # STAGE 4: Deterministic 3-Tier Routing Decision
        # -------------------------------------------------------------
        routing = self._evaluate_routing(intent_assessment, retrieval)

        # If pure HANDOFF from low confidence
        if routing.decision == "HANDOFF":
            return DecisionBundle(
                case_id=case_id,
                customer_text=clean_text,
                customer_handle=handle,
                intent=intent_assessment,
                retrieval=self._format_retrieval_dict(retrieval),
                routing=routing,
                draft_reply=None,
                failsafe_used=False,
                character_count=0,
            )

        # -------------------------------------------------------------
        # STAGE 5: Guarded Generation with Apple's 4 Rules
        # -------------------------------------------------------------
        draft, failsafe_used = self._generate_guarded_reply(
            clean_text=clean_text,
            handle=handle,
            intent=intent_name,
            retrieval=retrieval,
        )

        char_count = len(draft) if draft else 0

        return DecisionBundle(
            case_id=case_id,
            customer_text=clean_text,
            customer_handle=handle,
            intent=intent_assessment,
            retrieval=self._format_retrieval_dict(retrieval),
            routing=routing,
            draft_reply=draft,
            failsafe_used=failsafe_used,
            character_count=char_count,
        )

    def _classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify intent using available classifier or heuristic fallback."""
        if self.classifier is not None and hasattr(self.classifier, "classify"):
            try:
                from src.domain.models import SupportCase
                res = self.classifier.classify(SupportCase(case_id="temp", clean_text=text))
                return str(res.get("primary_intent", "UNKNOWN")), float(res.get("confidence", 0.90))
            except Exception as exc:
                logger.warning(f"Classifier error: {exc}; using fast heuristic fallback.")

        # High-precision cue fallback
        from scripts.extract_apple_playbook import CUES
        lowered = text.lower()
        scores = {}
        for intent, terms in CUES.items():
            matched = sum(1 for t in terms if t in lowered)
            if matched > 0:
                scores[intent] = matched

        if not scores:
            return "UNKNOWN", 0.40

        best_intent, match_count = max(scores.items(), key=lambda x: x[1])
        confidence = min(0.95, 0.70 + (match_count * 0.10))
        return best_intent, round(confidence, 2)

    def _evaluate_routing(
        self,
        intent: IntentAssessment,
        retrieval: RetrievalResult,
    ) -> RoutingDecision:
        """Issue deterministic 3-tier routing decision with auditable rule reason."""
        top_sim = retrieval.top_similarity
        zone = retrieval.confidence_zone

        # 1. Out of Scope / Edge Case HANDOFF
        if intent.name == "UNKNOWN" and intent.confidence < 0.50 and top_sim < 0.12:
            return RoutingDecision(
                decision="HANDOFF",
                priority="NORMAL_P3",
                rule_code="LOW_CONFIDENCE_UNRELIABLE_TWIN",
                stated_reason=(
                    f"Intent UNKNOWN (conf {intent.confidence:.2f}) and top similarity is low ({top_sim:.2f}). "
                    "Auto-reply suppressed to avoid hallucination; routed to Triage Queue."
                ),
            )

        # 2. AUTO_REPLY (High confidence intent + Strong precedent)
        if intent.confidence >= 0.80 and zone == "HIGH_PREPARATION":
            return RoutingDecision(
                decision="AUTO_REPLY",
                priority="NORMAL_P3",
                rule_code="RULE_AUTO_ROUTINE_TRIAGE",
                stated_reason=(
                    f"High confidence intent '{intent.name}' ({intent.confidence:.2f}) with verified "
                    f"historical twin (similarity {top_sim:.2f}). Standard triage auto-reply approved."
                ),
            )

        # 3. ASSISTED_REPLY (Moderate confidence or Partial precedent match)
        return RoutingDecision(
            decision="ASSISTED_REPLY",
            priority="PRIORITY_P2",
            rule_code="RULE_ASSISTED_MODERATE_AMBIGUITY",
            stated_reason=(
                f"Moderate confidence intent '{intent.name}' ({intent.confidence:.2f}) or stylistic twin "
                f"({zone}, sim {top_sim:.2f}). Draft generated for 1-click human agent review."
            ),
        )

    def _generate_guarded_reply(
        self,
        clean_text: str,
        handle: str,
        intent: str,
        retrieval: RetrievalResult,
    ) -> Tuple[str, bool]:
        """Generate draft reply with Apple guardrails, falling back to #1 template on failure."""
        # Check if user already provided specs in their tweet
        has_device = any(w in clean_text.lower() for w in ["iphone", "ipad", "macbook", "watch", "ios", "macos"])
        has_version = bool(re.search(r"\b(ios\s*\d+|1[0-8]\.\d+)\b", clean_text.lower()))

        precedent_text = "\n".join(
            f"Precedent {i+1}: Customer: \"{m.customer_query}\" -> Apple: \"{m.apple_reply}\""
            for i, m in enumerate(retrieval.matches[:3])
        )

        system_instruction = (
            "You are @AppleSupport on Twitter. Draft a response to the customer's tweet.\n"
            "MANDATORY BRAND RULES:\n"
            "1. Tone: Empathetic, concise, direct, professional. No conversational filler or apologetic essays.\n"
            f"2. The Diagnostic Ask: {'Inquire which iOS version and device model they are using.' if not (has_device and has_version) else 'Acknowledge the device/symptoms.'}\n"
            "3. The Privacy Pivot: Always instruct the user to DM us to continue safely: 'DM us here: {dm_link}'\n"
            "4. The Liability Shield: NEVER promise repairs, free replacements, or guarantee fixes. Say 'We'd like to help'.\n"
            "5. Length: Must be under 240 characters (Twitter limit is 280).\n"
            f"6. Start the tweet with the user handle: {handle}"
        )

        prompt = (
            f"Historical Apple Precedents for intent {intent}:\n{precedent_text}\n\n"
            f"New Customer Tweet: \"{clean_text}\"\n\n"
            "Draft the official @AppleSupport tweet:"
        )

        failsafe_used = False
        draft_reply = ""

        try:
            raw_generation = self.llm_client.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.2,
                max_tokens=100,
            ).strip()

            # Sanitize output formatting
            clean_draft = raw_generation.strip('"').strip("'").strip()
            clean_draft = clean_draft.replace("{customer_handle}", handle)
            clean_draft = clean_draft.replace("{dm_link}", self.DM_LINK_DEFAULT)
            clean_draft = clean_draft.replace("{help_link}", "https://apple.co/support")

            # Post-generation compliance checks:
            is_too_long = len(clean_draft) > 280
            has_promise = bool(PROMISSORY_PATTERNS.search(clean_draft))
            missing_handle = not clean_draft.startswith(handle)

            if is_too_long or has_promise or missing_handle:
                logger.warning(
                    f"LLM draft violated guardrail (length={len(clean_draft)}, promise={has_promise}). Triggering failsafe."
                )
                failsafe_used = True
                draft_reply = self._apply_failsafe(retrieval.best_template, handle)
            else:
                draft_reply = clean_draft

        except Exception as exc:
            logger.warning(f"LLM generation failed: {exc}. Using zero-downtime template failsafe.")
            failsafe_used = True
            draft_reply = self._apply_failsafe(retrieval.best_template, handle)

        return draft_reply, failsafe_used

    def _apply_failsafe(self, template: str, handle: str) -> str:
        """Deterministically populate the best retrieved historical template."""
        res = template.replace("{customer_handle}", handle)
        res = res.replace("{dm_link}", self.DM_LINK_DEFAULT)
        res = res.replace("{help_link}", "https://apple.co/support")
        if not res.startswith(handle):
            res = f"{handle} {res}"
        # Ensure under 280 chars
        if len(res) > 280:
            res = res[:277] + "..."
        return res

    def _format_retrieval_dict(self, res: RetrievalResult) -> Dict[str, Any]:
        return {
            "intent": res.intent,
            "top_similarity": res.top_similarity,
            "confidence_zone": res.confidence_zone,
            "matches": [asdict(m) for m in res.matches],
        }

