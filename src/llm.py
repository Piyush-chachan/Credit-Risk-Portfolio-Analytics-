"""Small OpenAI-compatible client for Groq. Never logs credentials."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def _chat(self, system: str, user: str, max_tokens: int = 500, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_text(self, system: str, user: str, max_tokens: int = 220) -> str:
        return self._chat(system, user, max_tokens=max_tokens).strip()

    def chat_json(self, system: str, user: str, max_tokens: int = 500) -> dict:
        raw = self._chat(system, user, max_tokens=max_tokens, json_mode=True)
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be an object")
        return parsed


def get_llm() -> LLMClient | None:
    """Return None when no key is configured; caller then uses offline rules."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    # Streamlit-hosted deployments expose secrets through st.secrets, not env vars.
    if not api_key:
        try:
            import streamlit as st
            api_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    if not api_key:
        logger.info("GROQ_API_KEY is not configured; using offline rule-based mode")
        return None
    try:
        def setting(name: str, default: str) -> str:
            value = os.getenv(name)
            if value:
                return value
            try:
                import streamlit as st
                return str(st.secrets.get(name, default))
            except Exception:
                return default

        client = OpenAI(
            api_key=api_key,
            base_url=setting("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=float(setting("LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(setting("LLM_MAX_RETRIES", "2")),
        )
        model = setting("GROQ_MODEL", "llama-3.3-70b-versatile")
        return LLMClient(client, model)
    except Exception:
        logger.exception("LLM initialization failed; using offline rule-based mode")
        return None
