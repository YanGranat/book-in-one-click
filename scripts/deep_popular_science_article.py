#!/usr/bin/env python3
"""
Interactive generator for a deep popular science article.
Saves results to output/deep_popular_science_article/.
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
from pipelines.article.pipeline import build_instructions as build_article_instructions


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
    parser = argparse.ArgumentParser(description="Generate a deep popular science article")
    parser.add_argument("--topic", type=str, default="", help="Topic to generate about")
    parser.add_argument("--lang", type=str, default="auto", help="Language: auto|ru|en")
    parser.add_argument("--out", type=str, default="deep_popular_science_article", help="Output subdirectory")
    args = parser.parse_args()
    ensure_project_root_on_syspath()
    load_env()

    try:
        Agent, Runner = try_import_sdk()
    except ImportError:
        return

    progress("start: article")
    print("📝 Deep popular science article generator")
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

    print(f"\n🔄 Генерирую глубокую научно-популярную статью на тему: '{topic}'")
    progress("agent:init")
    print("⏳ Это может занять 1-2 минуты...")

    try:
        agent = Agent(
            name="Deep Popular Science Article Writer",
            instructions=build_article_instructions(topic, args.lang),
            model="gpt-4o",
        )

        progress("agent:run")
        result = Runner.run_sync(agent, f"Тема: {topic}")
        content = getattr(result, "final_output", "")

        if not content:
            print("❌ Пустой результат от агента")
            return

        progress("io:prepare_output")
        output_dir = ensure_output_dir(args.out)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"article_{safe_slug(topic)}_{timestamp}.md"
        filepath = output_dir / filename

        progress("io:save")
        save_markdown(filepath, title=topic, generator="OpenAI Agents SDK", pipeline="DeepPopularScienceArticle", content=content)

        progress("done")
        print(f"✅ Готово. Сохранено: {filepath}")
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n📖 Превью:\n{preview}")
        print("\n📊 Статистика:")
        print(f"   Слов: {len(content.split())}")
        print(f"   Символов: {len(content)}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()


