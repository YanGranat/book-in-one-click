#!/usr/bin/env python3
"""
I/O helpers for Markdown saving and output directories.
"""
from datetime import datetime
from pathlib import Path


def ensure_output_dir(subdir: str) -> Path:
    output_dir = Path("output") / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_markdown(path: Path, title: str, generator: str, pipeline: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"*Создано: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}*\n")
        f.write(f"*Генератор: {generator}*\n")
        f.write(f"*Пайплайн: {pipeline}*\n\n")
        f.write(f"{content}\n")


