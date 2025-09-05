from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_post_prompt() -> str:
	prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "writing" / "post.md"
	return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""


def _load_refine_prompt() -> str:
	prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "rewriting" / "refine_post.md"
	return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""


def try_import_sdk():
	from agents import Agent  # type: ignore
	return Agent


def build_refine_agent() -> Any:
	Agent = try_import_sdk()
	combined_instructions = (
		_load_refine_prompt() + "\n\n" +
		"<style_contract_original_writer>\n" + _load_post_prompt() + "\n</style_contract_original_writer>\n"
	)
	return Agent(
		name="Post Style Refiner",
		instructions=combined_instructions,
		model="gpt-5",
	)







