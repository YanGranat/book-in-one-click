from __future__ import annotations

# Deprecated in 2-module architecture. Kept for compatibility if imported accidentally.
from typing import Any


def build_subsections_content_agent(model: str | None = None) -> Any:  # pragma: no cover
    raise RuntimeError("content_of_subsections is removed in 2-module pipeline. Use subsection_writer in module_02_writing.")


