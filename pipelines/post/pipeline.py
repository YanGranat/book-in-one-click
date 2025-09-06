from pathlib import Path


def build_instructions(topic: str, lang: str) -> str:
    # Return only the canonical system prompt text from file (single source of truth)
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "post" / "module_01_writing" / "post.md"
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
# Second pass formatter removed — system prompt is the single source of truth


