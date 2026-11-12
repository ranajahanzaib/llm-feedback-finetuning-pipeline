from __future__ import annotations

import hashlib


def input_fingerprint(text: str) -> str:
    """Hash normalized whitespace and case to catch trivial leakage variants."""
    canonical = " ".join(text.split()).casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
