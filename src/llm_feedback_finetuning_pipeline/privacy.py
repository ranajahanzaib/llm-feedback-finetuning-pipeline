from __future__ import annotations

import re

PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)"),
    "credential": re.compile(r"\b(?:password|api[_ -]?key|secret|token)\s*[:=]\s*\S+", re.I),
    "key_like": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16})\b"),
}


def flags(text: str) -> list[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]


def redact(text: str) -> str:
    for name, pattern in PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{name.upper()}]", text)
    return text
