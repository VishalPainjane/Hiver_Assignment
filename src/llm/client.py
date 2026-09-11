"""Unified Multi-Tier LLM Provider with Seamless Failover.

Tiers:
1. Google Gemini 2.5 Flash (Cloud, Fast)
2. Groq Cloud Llama / Qwen (Cloud, Ultra-fast)
3. Ollama Llama 3.2 (Local GPU RTX 4050, Offline)
4. Heuristic / Deterministic Fallback (Offline, Zero-dependency)
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("llm.client")


class UnifiedLLMClient:
    """Resilient LLM client with automatic failover between Cloud APIs and Local Ollama."""

    def __init__(
        self,
        primary: str = "gemini",
        fallback: str = "groq",
        ollama_model: str = "llama3.2:latest",
        groq_model: str = "qwen/qwen3.8-27b",
        gemini_model: str = "gemini-2.5-flash",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.ollama_model = ollama_model
        self.groq_model = groq_model
        self.gemini_model = gemini_model

        self.gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.groq_key = os.environ.get("GROQ_API_KEY", "").strip()

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        """Execute prompt across available tiers with automatic fallback."""
        errors = []

        # Tier 1: Primary provider
        if self.primary == "gemini" and self.gemini_key:
            try:
                return self._call_gemini(prompt, system_instruction, json_mode, temperature, max_tokens)
            except Exception as exc:
                errors.append(f"Gemini primary error: {exc}")
                logger.warning(f"Gemini primary failed: {exc}, attempting fallback...")
        elif self.primary == "ollama":
            try:
                return self._call_ollama(prompt, system_instruction, json_mode, temperature, max_tokens)
            except Exception as exc:
                errors.append(f"Ollama primary error: {exc}")
                logger.warning(f"Ollama primary failed: {exc}, attempting fallback...")
        elif self.primary == "groq" and self.groq_key:
            try:
                return self._call_groq(prompt, system_instruction, json_mode, temperature, max_tokens)
            except Exception as exc:
                errors.append(f"Groq primary error: {exc}")
                logger.warning(f"Groq primary failed: {exc}, attempting fallback...")

        # Tier 2: Secondary provider
        if self.fallback == "groq" and self.groq_key:
            try:
                return self._call_groq(prompt, system_instruction, json_mode, temperature, max_tokens)
            except Exception as exc:
                errors.append(f"Groq fallback error: {exc}")
                logger.warning(f"Groq fallback failed: {exc}, attempting Ollama...")
        elif self.fallback == "gemini" and self.gemini_key:
            try:
                return self._call_gemini(prompt, system_instruction, json_mode, temperature, max_tokens)
            except Exception as exc:
                errors.append(f"Gemini fallback error: {exc}")
                logger.warning(f"Gemini fallback failed: {exc}, attempting Ollama...")

        # Tier 3: Local Ollama (GPU-accelerated)
        try:
            return self._call_ollama(prompt, system_instruction, json_mode, temperature, max_tokens)
        except Exception as exc:
            errors.append(f"Ollama error: {exc}")
            logger.warning(f"Ollama failed: {exc}")

        raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")

    def _call_gemini(
        self,
        prompt: str,
        system_instruction: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"SYSTEM: {system_instruction}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Empty candidate response from Gemini")
            return candidates[0]["content"]["parts"][0]["text"].strip()

    def _call_groq(
        self,
        prompt: str,
        system_instruction: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content.strip()

    def _call_ollama(
        self,
        prompt: str,
        system_instruction: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        payload: Dict[str, Any] = {
            "model": self.ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
