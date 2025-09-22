#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```") and s.endswith("```"):
        try:
            lines = s.splitlines()
            # remove first and last fence line
            return "\n".join(lines[1:-1])
        except Exception:
            return s
    return s


def extract_last_json_block(text: str) -> str | None:
    if not text:
        return None
    # Find last {...} JSON object at the end (greedy but bounded)
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        return m.group(0)
    # Try last [...] array at the end
    m = re.search(r"\[[\s\S]*\]\s*$", text)
    if m:
        return m.group(0)
    return None


def parse_json_best_effort(text: str) -> Any:
    """Parse JSON from LLM output robustly.
    Steps: strip code fences → json.loads → extract last JSON block and parse.
    Raises ValueError if cannot parse.
    """
    s = strip_code_fences(text)
    try:
        return json.loads(s)
    except Exception:
        last = extract_last_json_block(s)
        if last is None:
            raise ValueError("No JSON object found in text")
        return json.loads(last)


