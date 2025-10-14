#!/usr/bin/env python3
"""
Interactive generator for a popular science post.
Saves results to output/post/.
"""
import os
import sys
from pathlib import Path
import argparse
import asyncio

# Ensure project root is on sys.path before importing project modules
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from utils.env import ensure_project_root_on_syspath as ensure_root, load_env_from_root
from utils.slug import safe_filename_base
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from orchestrator import progress
# Legacy agent imports removed in favor of unified services.post.generate
from pipelines.post.pipeline import build_instructions as build_post_instructions


def load_env() -> None:
    load_env_from_root(__file__)


def ensure_project_root_on_syspath() -> None:
    ensure_root(__file__)


def try_import_sdk():
    try:
        # Primary import path used by OpenAI Agents SDK
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        print("❌ ImportError: cannot import Agent/Runner from 'agents'.")
        print("➡️ Likely cause: a local folder named 'agents' shadows the SDK module.")
        print("➡️ Fix: rename local folder to 'llm_agents' (already recommended).")
        print(f"Details: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a popular science post")
    parser.add_argument("--topic", type=str, default="", help="Topic to generate about")
    parser.add_argument("--lang", type=str, default="auto", help="Language: auto|ru|en")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider: openai|gemini|claude")
    parser.add_argument("--style", type=str, default="post_style_1", help="Post style: post_style_1|post_style_2")
    parser.add_argument("--out", type=str, default="post", help="Output subdirectory")
    parser.add_argument("--no-factcheck", action="store_true", help="Disable fact-check step")
    parser.add_argument("--factcheck-max-items", type=int, default=0, help="Limit number of points to research (0=all)")
    parser.add_argument("--research-iterations", type=int, default=2, help="Max research iterations per point")
    parser.add_argument("--research-concurrency", type=int, default=3, help="Parallel research workers per post")
    parser.add_argument("--include-logs", action="store_true", help="Also save generation log file")
    args = parser.parse_args()
    ensure_project_root_on_syspath()
    load_env()

    try:
        Agent, Runner = try_import_sdk()
    except ImportError:
        return

    

    # Check API keys based on selected provider
    provider = (args.provider or "openai").strip().lower()
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден в .env файле в корне проекта")
        return
    elif provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY не найден в .env файле в корне проекта")
        return
    elif provider in {"gemini", "google"} and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("❌ GOOGLE_API_KEY (или GEMINI_API_KEY) не найден в .env файле в корне проекта")
        return

    topic = args.topic.strip()
    if not topic:
        print("\n🤔 Введите тему:")
        topic = input("➤ ").strip()
    if not topic:
        print("❌ Тема не может быть пустой")
        return

    print(f"🔄 Генерирую научно-популярный пост: '{topic}'")
    

    try:
        agent = Agent(
            name="Popular Science Post Writer",
            instructions=build_post_instructions(topic, args.lang),
            model="gpt-5",
        )

        lang = (args.lang or "auto").strip()
        user_message = (
            f"<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"</input>"
        )
        result = Runner.run_sync(agent, user_message)
        content = getattr(result, "final_output", "")

        if not content:
            print("❌ Пустой результат от агента")
            return

        # Single-pass output is used; formatting is enforced via the system prompt only

        output_dir = ensure_output_dir(args.out)

        base = f"{safe_filename_base(topic)}_post"

        # Use unified generate_post for all providers (simpler and consistent)
        from services.post.generate import generate_post
        if args.include_logs:
            # Generate and return log path
            log_path = generate_post(
                topic,
                lang=args.lang,
                provider=args.provider,
                style=(args.style or "post_style_1"),
                factcheck=not args.no_factcheck,
                research_iterations=args.research_iterations,
                research_concurrency=args.research_concurrency,
                output_subdir=args.out,
                job_meta={"source": "cli", "script": "Popular_science_post.py"},
                return_log_path=True,
            )
            print(f"📋 Лог сохранён: {log_path}")
            # Also generate main result
            path = generate_post(
                topic,
                lang=args.lang,
                provider=args.provider,
                style=(args.style or "post_style_1"),
                factcheck=not args.no_factcheck,
                research_iterations=args.research_iterations,
                research_concurrency=args.research_concurrency,
                output_subdir=args.out,
                job_meta={"source": "cli", "script": "Popular_science_post.py"},
            )
        else:
            path = generate_post(
                topic,
                lang=args.lang,
                provider=args.provider,
                style=(args.style or "post_style_1"),
                factcheck=not args.no_factcheck,
                research_iterations=args.research_iterations,
                research_concurrency=args.research_concurrency,
                output_subdir=args.out,
                job_meta={"source": "cli", "script": "Popular_science_post.py"},
            )
        print(f"💾 Сохранено: {path}")
        print("✅ Готово.")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()


