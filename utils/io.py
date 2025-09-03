#!/usr/bin/env python3
"""
I/O helpers for Markdown saving and output directories.
"""
from datetime import datetime
from pathlib import Path
from utils.config import load_config


def ensure_output_dir(subdir: str) -> Path:
    output_dir = Path("output") / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_markdown(path: Path, title: str, generator: str, pipeline: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        cfg = load_config(__file__)
        meta_cfg = cfg.get("metadata", {}) if isinstance(cfg, dict) else {}
        include_title = bool(meta_cfg.get("include_title", False))
        include_ts = bool(meta_cfg.get("include_timestamp", False))
        include_gen = bool(meta_cfg.get("include_generator", False))
        include_pipe = bool(meta_cfg.get("include_pipeline", False))

        if include_title:
            f.write(f"# {title}\n\n")

        any_meta = False
        if include_ts:
            f.write(f"*Создано: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}*\n")
            any_meta = True
        if include_gen:
            f.write(f"*Генератор: {generator}*\n")
            any_meta = True
        if include_pipe:
            f.write(f"*Пайплайн: {pipeline}*\n")
            any_meta = True

        if any_meta:
            f.write("\n")
        f.write(f"{content}\n")


def next_available_filepath(directory: Path, filename_base: str, ext: str = ".md") -> Path:
    """
    Return `<base>.ext` or `<base>_2.ext` ... if already exists.
    """
    candidate = directory / f"{filename_base}{ext}"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = directory / f"{filename_base}_{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1


