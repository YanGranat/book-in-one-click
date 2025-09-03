from pathlib import Path


def build_instructions(topic: str, lang: str) -> str:
	# System prompt for the summary writing agent
	prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "writing" / "summary.md"
	base = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "Подготовь краткий структурированный конспект (summary) по теме."
	lang_prefix = "" if lang == "auto" else f"Язык вывода: {lang}.\n"
	return lang_prefix + base
