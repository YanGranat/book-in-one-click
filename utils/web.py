#!/usr/bin/env python3
from __future__ import annotations

import re
import html
from typing import List, Dict

import requests
from bs4 import BeautifulSoup  # type: ignore

try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover
    DDGS = None  # type: ignore


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search_web(query: str, max_results: int = 8, timelimit: str = "y") -> List[Dict[str, str]]:
    """Search the web and return a list of result dicts: title, href, body/snippet.
    Uses DuckDuckGo; falls back to empty if package missing.
    """
    results: List[Dict[str, str]] = []
    if DDGS is None:
        return results
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="wt-wt", safesearch="moderate", timelimit=timelimit, max_results=max_results):
                results.append({
                    "title": _clean_text(r.get("title", "")),
                    "href": r.get("href", ""),
                    "snippet": _clean_text(r.get("body", "")),
                })
                if len(results) >= max_results:
                    break
    except Exception:
        pass
    return results


def fetch_url_text(url: str, timeout: int = 15, max_chars: int = 8000) -> str:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # remove scripts/styles/nav
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.extract()
        text = soup.get_text(" ")
        text = _clean_text(text)
        return text[:max_chars]
    except Exception:
        return ""


def build_search_context(queries: List[str], per_query: int = 4, max_chars: int = 6000) -> str:
    context_parts: List[str] = []
    for q in queries[:8]:
        results = search_web(q, max_results=per_query)
        for i, r in enumerate(results):
            if not r.get("href"):
                continue
            content = fetch_url_text(r["href"], max_chars=max_chars)
            if not content:
                continue
            part = (
                f"<source>\n<title>{r['title']}</title>\n"
                f"<url>{r['href']}</url>\n"
                f"<snippet>{r['snippet']}</snippet>\n"
                f"<content>{content}</content>\n"
                f"</source>"
            )
            context_parts.append(part)
    return "\n".join(context_parts)


