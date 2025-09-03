#!/usr/bin/env python3
"""
Simple smoke test — minimal check that the API works and Markdown saves.
Saves results to output/simple_test/.
"""
import os
from pathlib import Path


def try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        print("❌ Ошибка импорта SDK: 'from agents import Agent, Runner' не удалось")
        print("➡️ Убедитесь, что локальная папка не называется 'agents' (используйте 'llm_agents').")
        print(f"Details: {e}")
        raise


def load_env_from_root() -> None:
    """Load .env from project root (parent of this script)."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


def to_safe_filename_base(text: str) -> str:
    """Make a safe filename base while preserving non-Latin characters and case."""
    forbidden = '<>:"/\\|?*\n\r\t'
    base = ''.join('_' if ch in forbidden else ch for ch in text)
    base = base.replace(' ', '_').strip().strip('. ')
    while '__' in base:
        base = base.replace('__', '_')
    return base or 'untitled'


def ensure_output_dir(subdir: str) -> Path:
    out = Path("output") / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_available_filepath(directory: Path, filename_base: str, ext: str = ".md") -> Path:
    """Return `<base>.ext` or `<base>_2.ext` ... if already exists."""
    candidate = directory / f"{filename_base}{ext}"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = directory / f"{filename_base}_{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1


def detect_lang_from_text(text: str) -> str:
    """Very simple language heuristic: return 'ru' if Cyrillic letters present, else 'en'.

    This is intentionally lightweight (no external deps). It is sufficient for the
    smoke test where the topic is a short phrase like "Transposons" or "Транспозоны".
    """
    has_cyrillic = False
    for ch in text:
        lower = ch.lower()
        if 'а' <= lower <= 'я' or lower == 'ё':
            has_cyrillic = True
            break
    return 'ru' if has_cyrillic else 'en'

def main() -> None:
    load_env_from_root()

    Agent, Runner = try_import_sdk()

    print("📝 Простой тест — один агент, страница текста")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден в .env файле в корне проекта")
        return

    print("\n🤔 Введите тему:")
    topic = input("➤ ").strip()
    if not topic:
        print("❌ Тема не может быть пустой")
        return

    print(f"\n🔄 Генерирую страницу текста на тему: '{topic}'")
    print("⏳ Это займет 30–60 секунд...")

    try:
        lang = detect_lang_from_text(topic)

        instructions = (
            f"Язык вывода: {lang}.\n"
            "Напиши информативную страницу текста на заданную тему.\n\n"
            "Структура:\n"
            "1. Заголовок\n"
            "2. Краткое введение (1 абзац)\n"
            "3. Основная часть (2–3 абзаца с подробностями)\n"
            "4. Заключение или практические выводы (1 абзац)\n\n"
            "Стиль: образовательный, понятный, с примерами.\n"
            "Объем: примерно 300–500 слов (страница текста)."
        )

        agent = Agent(
            name="Content Writer",
            instructions=instructions,
            model="gpt-5",
        )

        # Pass only the topic as the user message to avoid biasing the language.
        result = Runner.run_sync(agent, topic)
        content = getattr(result, "final_output", "")

        if not content:
            print("❌ Пустой результат от агента")
            return

        output_dir = ensure_output_dir("simple_test")

        base = to_safe_filename_base(topic)
        filepath = next_available_filepath(output_dir, f"{base}_test", ".md")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"💾 Сохранено: {filepath}")
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n📖 Начало результата:\n{preview}")
        print("\n📊 Статистика:")
        print(f"   Слов: {len(content.split())}")
        print(f"   Символов: {len(content)}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")

    print("\n👋 Тест завершен")


if __name__ == "__main__":
    main()


