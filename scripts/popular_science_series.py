#!/usr/bin/env python3
"""
Interactive generator for a series of popular science posts.
Saves results to output/post_series/ by default.
"""
import os
import sys
from pathlib import Path
import argparse

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.env import ensure_project_root_on_syspath as ensure_root, load_env_from_root


def load_env() -> None:
    load_env_from_root(__file__)


def ensure_project_root_on_syspath() -> None:
    ensure_root(__file__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a series of popular science posts")
    parser.add_argument("--topic", type=str, default="", help="Topic to generate about")
    parser.add_argument("--lang", type=str, default="auto", help="Language: auto|ru|en")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider: openai|gemini|claude")
    parser.add_argument("--mode", type=str, default="auto", help="Mode: auto|fixed")
    parser.add_argument("--count", type=int, default=0, help="Number of posts in fixed mode")
    parser.add_argument("--max-iterations", type=int, default=1, help="Max sufficiency/extend iterations")
    parser.add_argument("--sufficiency-heavy-after", type=int, default=3, help="Switch to heavy model for sufficiency after this iteration index")
    parser.add_argument("--output", type=str, default="single", help="Output: single|folder")
    parser.add_argument("--out", type=str, default="post_series", help="Output subdirectory")
    parser.add_argument("--factcheck", action="store_true", help="Enable fact-check for each post")
    parser.add_argument("--research-iterations", type=int, default=2, help="Max research iterations per point (fact-check)")
    parser.add_argument("--refine", action="store_true", help="Enable final refine pass for each post")
    args = parser.parse_args()

    ensure_project_root_on_syspath()
    load_env()

    topic = (args.topic or "").strip()
    if not topic:
        print("\n🤔 Введите тему:")
        topic = input("➤ ").strip()
    if not topic:
        print("❌ Тема не может быть пустой")
        return

    from services.post_series.generate_series import generate_series
    try:
        path = generate_series(
            topic,
            lang=args.lang,
            provider=args.provider,
            mode=args.mode,
            count=args.count,
            max_iterations=args.max_iterations,
            sufficiency_heavy_after=args.sufficiency_heavy_after,
            output_mode=args.output,
            output_subdir=args.out,
            factcheck=bool(args.factcheck),
            research_iterations=args.research_iterations,
            refine=bool(args.refine),
            job_meta={"source": "cli", "script": "popular_science_series.py"},
        )
        print(f"💾 Сохранено: {path}")
        print("✅ Готово.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()


