#!/usr/bin/env python3
"""
Stub for deep popular science book generator.
Currently creates a minimal scaffold (outline + 1 section) and saves to output/deep_popular_science_book/.
Will be expanded to multi-agent pipeline later.
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse

# Ensure project root is on sys.path before importing project modules
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from utils.env import ensure_project_root_on_syspath as ensure_root, load_env_from_root
from utils.slug import safe_slug
from utils.io import ensure_output_dir, save_markdown
from orchestrator import progress


def load_env() -> None:
    load_env_from_root(__file__)


def ensure_project_root_on_syspath() -> None:
    ensure_root(__file__)


def try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        print("❌ ImportError: cannot import Agent/Runner from 'agents'.")
        print("➡️ Likely cause: a local folder named 'agents' shadows the SDK module.")
        print("➡️ Fix: rename local folder to 'llm_agents' (already recommended).")
        print(f"Details: {e}")
        raise


def safe_slug(text: str) -> str:
    forbidden = '<>:"/\\|?*\n\r\t'
    cleaned = ''.join('_' if ch in forbidden else ch for ch in text)
    cleaned = cleaned.replace(' ', '_')
    return cleaned.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deep popular science book (stub)")
    parser.add_argument("--topic", type=str, default="", help="Topic to generate about")
    parser.add_argument("--lang", type=str, default="auto", help="Language: auto|ru|en")
    parser.add_argument("--out", type=str, default="deep_popular_science_book", help="Output subdirectory")
    args = parser.parse_args()
    ensure_project_root_on_syspath()
    load_env()

    try:
        Agent, Runner = try_import_sdk()
    except ImportError:
        return

    progress("start: book_stub")
    print("📝 Deep popular science book (stub)")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден в .env файле в корне проекта")
        return

    topic = args.topic.strip()
    if not topic:
        print("\n🤔 Введите тему:")
        topic = input("➤ ").strip()
    if not topic:
        print("❌ Тема не может быть пустой")
        return

    print(f"\n🔄 Генерирую заготовку книги на тему: '{topic}'")
    progress("agent:outline:init")
    print("⏳ Это может занять 1-2 минуты...")

    try:
        # Optional language control
        lang_prefix = "" if args.lang == "auto" else f"Язык вывода: {args.lang}.\n"

        outline_agent = Agent(
            name="Book Outline Planner",
            instructions=(
                lang_prefix +
                "Составь оглавление глубокой научно-популярной книги по заданной теме.\n"
                "Структура: 5-8 глав, в каждой 3-5 разделов.\n"
                "Язык: совпадает с языком входного запроса."
            ),
            model="gpt-4o",
        )

        section_agent = Agent(
            name="Book Section Writer",
            instructions=(
                lang_prefix +
                "Напиши содержательный раздел книги в научно-популярном стиле.\n"
                "Поясняй термины, переходи от простого к сложному, избегай фактических ошибок."
            ),
            model="gpt-4o",
        )

        progress("agent:outline:run")
        outline_result = Runner.run_sync(outline_agent, f"Тема: {topic}")
        outline_text = getattr(outline_result, "final_output", "")

        progress("agent:section:run")
        section_result = Runner.run_sync(section_agent, f"Напиши первый раздел книги по теме: {topic}")
        section_text = getattr(section_result, "final_output", "")

        if not outline_text or not section_text:
            print("❌ Пустой результат от одного из агентов")
            return

        progress("io:prepare_output")
        output_dir = ensure_output_dir(args.out)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"book_{safe_slug(topic)}_{timestamp}.md"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {topic}\n\n")
            f.write(f"*Создано: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}*\n")
            f.write("*Генератор: OpenAI Agents SDK*\n")
            f.write("*Пайплайн: DeepPopularScienceBook (stub)*\n\n")
            f.write("## Оглавление (черновик)\n\n")
            f.write(outline_text + "\n\n")
            f.write("## Пример раздела\n\n")
            f.write(section_text + "\n")

        progress("done")
        print(f"✅ Готово. Сохранено: {filepath}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()


