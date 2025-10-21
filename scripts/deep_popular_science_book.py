#!/usr/bin/env python3
"""
CLI entrypoint for deep popular science book generation.
"""
from __future__ import annotations

import argparse
from services.book.generate_book import generate_book


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deep popular science book")
    parser.add_argument("topic", type=str, help="Topic of the book")
    parser.add_argument("--lang", default="auto", help="Output language (ru|en|auto)")
    parser.add_argument("--provider", default="openai", help="Provider (openai|gemini|claude)")
    args = parser.parse_args()
    path = generate_book(args.topic, lang=args.lang, provider=args.provider)
    print(str(path))


if __name__ == "__main__":
    main()


