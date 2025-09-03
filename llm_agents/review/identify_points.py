from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.research import ResearchPlan


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "review" / "identify_risky_points.md"
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_identify_points_agent() -> Any:
    Agent = try_import_sdk()
    from utils.config import load_config
    cfg = load_config(__file__)
    fast_model = cfg.get("fast_model", "gpt-5-mini")
    return Agent(
        name="Identify Risky Points",
        instructions=_load_prompt(),
        model=fast_model,
        output_type=ResearchPlan,
    )


