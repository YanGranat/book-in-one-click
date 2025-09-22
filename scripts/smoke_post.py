#!/usr/bin/env python3
from __future__ import annotations

import argparse
from utils.env import load_env_from_root
from services.post.generate import generate_post


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--provider", default="openai")
    p.add_argument("--lang", default="auto")
    args = p.parse_args()

    load_env_from_root(__file__)
    path = generate_post(
        args.topic,
        provider=args.provider,
        lang=args.lang,
        factcheck=False,
        research_iterations=1,
        return_log_path=True,
    )
    print(f"OK: {path}")


if __name__ == "__main__":
    main()


