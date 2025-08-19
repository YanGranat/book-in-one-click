from pathlib import Path


def build_instructions(topic: str, lang: str) -> str:
    # System prompt for the post writing agent
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "writing" / "post.md"
    base = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "Напиши популярный научный пост."
    lang_prefix = "" if lang == "auto" else f"Язык вывода: {lang}.\n"
    return lang_prefix + base


