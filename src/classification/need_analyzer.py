"""Bounded, replaceable customer-need analyzer.

The default adapter is deliberately transparent and offline. A structured LLM
adapter can be injected at this boundary, but its output is normalised into the
same immutable contract before the policy layer sees it.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from src.domain.models import CustomerNeed, SupportCase


class NeedAnalyzer:
    TAXONOMY = (
        "BATTERY_DRAIN",
        "DEVICE_BOOT",
        "ACCOUNT_ACCESS",
        "BILLING_DISPUTE",
        "UPDATE_FAILURE",
        "UNKNOWN",
    )

    def __init__(
        self,
        classifier: Callable[[SupportCase], Mapping[str, Any]] | None = None,
    ) -> None:
        self.classifier = classifier

    def analyze(self, case: SupportCase) -> CustomerNeed:
        if self.classifier is None:
            raise ValueError("No classifier model configured.")
        proposal = self.classifier(case)
        return self._validate_proposal(proposal)

    def _validate_proposal(self, proposal: Mapping[str, Any]) -> CustomerNeed:
        intent = str(proposal.get("primary_intent", "UNKNOWN")).upper()
        if intent not in self.TAXONOMY:
            intent = "UNKNOWN"
        try:
            confidence = float(proposal.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        return CustomerNeed(
            primary_intent=intent,
            confidence=confidence,
        )

