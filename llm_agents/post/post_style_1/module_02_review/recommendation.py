from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.research import Recommendation


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[4] / "prompts" / "post" / "post_style_1" / "module_02_review" / "recommendation.md"
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_recommendation_agent(model: str | None = None) -> Any:
    Agent = try_import_sdk()
    return Agent(
        name="Recommendation (Style 1)",
        instructions=_load_prompt(),
        model=model or "gpt-5",
        output_type=Recommendation,
    )


