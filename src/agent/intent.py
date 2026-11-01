"""Cheap deterministic intent/safety gate before any model call."""
from __future__ import annotations
import re
from dataclasses import dataclass

INJECTION_RE = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above)|system prompt|"
    r"reveal.*(?:key|prompt|secret)|api[ _-]?key|jailbreak|drop\s+table|delete\s+from|;\s*--|<script",
    re.I,
)
SMALLTALK_RE = re.compile(r"^(hi|hello|hey|thanks|thank you|who are you|what can you do|help)\b", re.I)

@dataclass(frozen=True)
class Intent:
    kind: str  # portfolio_question | out_of_scope | unsafe | empty | too_long
    reason: str = ""


def detect_intent(question: str, max_chars: int = 400) -> Intent:
    q = (question or "").strip()
    if not q:
        return Intent("empty", "Please enter a question about the portfolio.")
    if len(q) > max_chars:
        return Intent("too_long", f"Please keep the question under {max_chars} characters.")
    if INJECTION_RE.search(q):
        return Intent("unsafe", "I can only answer questions about portfolio data; instructions to reveal secrets or change system rules are not supported.")
    if SMALLTALK_RE.search(q):
        return Intent("out_of_scope", "Ask a question about portfolio performance, risk, exposure, repayment or collections.")
    return Intent("portfolio_question")
