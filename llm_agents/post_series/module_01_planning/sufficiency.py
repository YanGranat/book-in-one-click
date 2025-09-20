from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.series import ListSufficiency


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "post_series" / "module_01_planning" / "sufficiency.md"
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_sufficiency_agent(model: str | None = None) -> Any:
    Agent = try_import_sdk()
    return Agent(
        name="Series Sufficiency Evaluator",
        instructions=_load_prompt(),
        model=model or "gpt-5",
        output_type=ListSufficiency,
    )



