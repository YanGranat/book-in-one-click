from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.research import SufficiencyDecision


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "post" / "module_02_review" / "sufficiency.md"
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_sufficiency_agent() -> Any:
    Agent = try_import_sdk()
    from utils.config import load_config
    cfg = load_config(__file__)
    fast_model = cfg.get("fast_model", "gpt-5-mini")
    return Agent(
        name="Sufficiency Evaluator",
        instructions=_load_prompt(),
        model=fast_model,
        output_type=SufficiencyDecision,
    )


