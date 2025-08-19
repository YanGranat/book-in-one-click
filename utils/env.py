#!/usr/bin/env python3
"""
Environment and path helpers.
"""
import os
import sys
from pathlib import Path


def ensure_project_root_on_syspath(current_file: str) -> None:
    project_root = Path(current_file).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_env_from_root(current_file: str) -> None:
    project_root = Path(current_file).resolve().parents[1]
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


