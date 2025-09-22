#!/usr/bin/env python3
from __future__ import annotations

import argparse
from utils.env import load_env_from_root
from services.chat.run import run_chat_message, build_system_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--message", required=True)
    p.add_argument("--provider", default="openai")
    p.add_argument("--lang", default="ru")
    args = p.parse_args()

    load_env_from_root(__file__)
    system = build_system_prompt(chat_lang=args.lang, kind="result", full_content="")
    # no session memory for smoke
    out = run_chat_message(args.provider, system, args.message, session_id=None)
    print((out or "").strip())


if __name__ == "__main__":
    main()


