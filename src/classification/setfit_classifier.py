"""SetFit Intent Classifier for @AppleSupport customer tweets on GPU."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from setfit import SetFitModel

from src.domain.models import SupportCase

logger = logging.getLogger("classification.setfit")


class SetFitIntentClassifier:
    """Production SetFit intent classifier running on GPU (NVIDIA RTX 4050)."""

    DEFAULT_TAXONOMY: Tuple[str, ...] = (
        "BATTERY_DRAIN",
        "DEVICE_BOOT",
        "ACCOUNT_ACCESS",
        "BILLING_DISPUTE",
        "UPDATE_FAILURE",
        "UNKNOWN",
    )

    def __init__(
        self,
        model_path: str | Path = "models/intent_setfit",
        device: Optional[str] = None,
        taxonomy: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.taxonomy = list(taxonomy or self.DEFAULT_TAXONOMY)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SetFitModel] = None

        if self.model_path.exists():
            self._load_model()
        else:
            logger.warning(
                f"Model path {self.model_path} does not exist yet. Call load or train first."
            )

    def _load_model(self) -> None:
        logger.info(f"Loading SetFit model from {self.model_path} to {self.device}...")
        self._model = SetFitModel.from_pretrained(str(self.model_path))
        if hasattr(self._model, "to"):
            self._model.to(self.device)

    def is_loaded(self) -> bool:
        return self._model is not None

    def classify(self, case: SupportCase) -> Dict[str, Any]:
        """Classify a single support case and return intent and confidence score."""
        if self._model is None:
            if self.model_path.exists():
                self._load_model()
            else:
                raise RuntimeError(
                    f"SetFit model not found at {self.model_path}. Train the model first."
                )

        text = case.clean_text.strip()
        if not text:
            return {"primary_intent": "UNKNOWN", "confidence": 0.0}

        probs = self._model.predict_proba([text])
        if hasattr(probs, "cpu"):
            probs = probs.cpu().numpy()

        class_probs = probs[0]
        pred_idx = int(class_probs.argmax())
        confidence = float(class_probs[pred_idx])

        # Map to taxonomy
        if hasattr(self._model, "labels") and self._model.labels:
            label = self._model.labels[pred_idx]
            intent = str(label)
        elif pred_idx < len(self.taxonomy):
            intent = self.taxonomy[pred_idx]
        else:
            intent = "UNKNOWN"

        return {
            "primary_intent": intent,
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Run batch inference on GPU returning (intent, confidence) for each text."""
        if self._model is None:
            self._load_model()

        clean_texts = [t.strip() if t.strip() else "empty" for t in texts]
        probs = self._model.predict_proba(clean_texts)
        if hasattr(probs, "cpu"):
            probs = probs.cpu().numpy()

        results = []
        for row in probs:
            pred_idx = int(row.argmax())
            conf = float(row[pred_idx])
            if hasattr(self._model, "labels") and self._model.labels:
                intent = str(self._model.labels[pred_idx])
            elif pred_idx < len(self.taxonomy):
                intent = self.taxonomy[pred_idx]
            else:
                intent = "UNKNOWN"
            results.append((intent, round(conf, 4)))

        return results

