#!/usr/bin/env python3
"""
Reusable generation function for popular science posts.
Extracted from scripts/Popular_science_post.py and adapted for server/bot usage.

Key guarantees:
- No interactive input; accepts all parameters explicitly
- Can be safely called from a background thread
- Returns a Path to the saved Markdown file
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Type
import os
import asyncio

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.slug import safe_filename_base
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from pipelines.post.pipeline import build_instructions as build_post_instructions


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed, "
            "and no local package named 'agents' shadows it."
        ) from e


def generate_post(
    topic: str,
    *,
    lang: str = "auto",
    provider: str = "openai",  # openai|gemini|claude
    factcheck: bool = True,
    factcheck_max_items: int = 0,
    research_iterations: int = 2,
    research_concurrency: int = 6,
    output_subdir: str = "post",
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Generate a popular science post and save it to output/<output_subdir>/.

    Returns the Path to the saved .md file.
    """
    # Ensure project modules are importable and env is loaded
    _ensure_root(__file__)
    load_env_from_root(__file__)

    if not topic or not topic.strip():
        raise ValueError("Topic must be a non-empty string")

    _prov = (provider or "openai").strip().lower()
    if _prov == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found in environment")
    elif _prov == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment")
    elif _prov in {"gemini", "google"}:
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Ensure this thread has an event loop for libs that expect it
    try:
        asyncio.get_event_loop()
    except Exception:
        asyncio.set_event_loop(asyncio.new_event_loop())

    Agent, Runner = _try_import_sdk()

    def _emit(stage: str) -> None:
        if on_progress:
            try:
                on_progress(stage)
            except Exception:
                pass

    _emit("start:post")

    instructions = build_post_instructions(topic, lang)

    def _run_openai_with(system: str, user_message_local: str) -> str:
        agent = Agent(
            name="Generic Agent",
            instructions=system,
            model="gpt-5",
        )
        res_local = Runner.run_sync(agent, user_message_local)
        return getattr(res_local, "final_output", "")

    def _run_openai() -> str:
        agent = Agent(
            name="Popular Science Post Writer",
            instructions=instructions,
            model="gpt-5",
        )
        user_message_local = (
            f"<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{(lang or 'auto').strip()}</lang>\n"
            f"</input>"
        )
        res_local = Runner.run_sync(agent, user_message_local)
        return getattr(res_local, "final_output", "")

    def _run_gemini_with(system: str, user_message_local: str) -> str:
        import google.generativeai as genai  # type: ignore
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system)
        resp = model.generate_content(user_message_local)
        return (getattr(resp, "text", None) or "").strip()

    def _run_gemini() -> str:
        user_message_local = (
            f"<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{(lang or 'auto').strip()}</lang>\n"
            f"</input>"
        )
        return _run_gemini_with(instructions, user_message_local)

    def _run_claude_with(system: str, user_message_local: str) -> str:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4")
        msg = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_message_local}],
        )
        parts = []
        for blk in getattr(msg, "content", []) or []:
            txt = getattr(blk, "text", None)
            if txt:
                parts.append(txt)
        return ("\n\n".join(parts)).strip()

    def _run_claude() -> str:
        user_message_local = (
            f"<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{(lang or 'auto').strip()}</lang>\n"
            f"</input>"
        )
        return _run_claude_with(instructions, user_message_local)

    def run_with_provider(system: str, user_inp: str) -> str:
        if _prov == "openai":
            return _run_openai_with(system, user_inp)
        if _prov in {"gemini", "google"}:
            return _run_gemini_with(system, user_inp)
        return _run_claude_with(system, user_inp)

    def run_json_with_provider(system: str, user_inp: str, cls: Type):
        import json
        txt = run_with_provider(system, user_inp)
        # strip code fences if any
        if txt.strip().startswith("```") and txt.strip().endswith("```"):
            txt = "\n".join([line for line in txt.strip().splitlines()[1:-1]])
        try:
            data = json.loads(txt)
            return cls.model_validate(data)
        except Exception:
            # try to find first json block
            import re
            m = re.search(r"\{[\s\S]*\}\s*$", txt)
            if m:
                return cls.model_validate(json.loads(m.group(0)))
            raise RuntimeError(f"Failed to parse JSON for {cls.__name__}")

    if _prov == "openai":
        content = _run_openai()
    elif _prov in {"gemini", "google"}:
        content = _run_gemini()
    else:
        content = _run_claude()
    if not content:
        raise RuntimeError("Empty result from writer agent")

    report = None
    if factcheck:
        from utils.config import load_config
        from pathlib import Path
        from schemas.research import ResearchPlan, QueryPack, ResearchIterationNote, SufficiencyDecision, Recommendation

        _emit("factcheck:init")
        base = Path(__file__).resolve().parents[2] / "prompts" / "post" / "module_02_review"
        p_ident = (base / "identify_risky_points.md").read_text(encoding="utf-8")
        plan = run_json_with_provider(p_ident, f"<post>\n{content}\n</post>", ResearchPlan)
        points = plan.points or []
        if factcheck_max_items and factcheck_max_items > 0:
            points = points[: factcheck_max_items]

        cfg = load_config(__file__)
        pref = (cfg.get("research", {}) or {}).get("preferred_domains", [])
        p_iter = (base / "iterative_research.md").read_text(encoding="utf-8")
        p_suff = (base / "sufficiency.md").read_text(encoding="utf-8")
        p_rec = (base / "recommendation.md").read_text(encoding="utf-8")
        p_qs = (base / "query_synthesizer.md").read_text(encoding="utf-8")

        async def _run_with_retries(agent, inp: str, attempts: int = 3, base_delay: float = 1.0):
            for i in range(attempts):
                try:
                    return await Runner.run(agent, inp)
                except Exception:
                    if i == attempts - 1:
                        raise
                    await asyncio.sleep(base_delay * (2 ** i))

        async def process_point_async(p):
            # Query synthesis
            cfg_pref = ",".join(pref)
            _ = run_json_with_provider(
                p_qs,
                f"<input>\n<point>{p.model_dump_json()}</point>\n<preferred_domains>{cfg_pref}</preferred_domains>\n</input>",
                QueryPack,
            )

            notes = []
            for step in range(1, max(1, int(research_iterations)) + 1):
                rr_input = (
                    "<input>\n"
                    f"<point>{p.model_dump_json()}</point>\n"
                    f"<step>{step}</step>\n"
                    "</input>"
                )
                note = run_json_with_provider(p_iter, rr_input, ResearchIterationNote)
                notes.append(note)

                suff_input = (
                    "<input>\n"
                    f"<point>{p.model_dump_json()}</point>\n"
                    f"<notes>[{','.join([n.model_dump_json() for n in notes])}]</notes>\n"
                    "</input>"
                )
                decision = run_json_with_provider(p_suff, suff_input, SufficiencyDecision)
                if decision.done:
                    break

            class _TmpReport:
                def __init__(self, point_id, notes):
                    self.point_id = point_id
                    self.notes = notes
                    self.synthesis = ""

                def model_dump_json(self):
                    import json
                    return json.dumps(
                        {
                            "point_id": self.point_id,
                            "notes": [n.model_dump() for n in self.notes],
                            "synthesis": self.synthesis,
                        },
                        ensure_ascii=False,
                    )

            rr = _TmpReport(p.id, notes)
            rec = run_json_with_provider(
                p_rec,
                f"<input>\n<point>{p.model_dump_json()}</point>\n<report>{rr.model_dump_json()}</report>\n</input>",
                Recommendation,
            )
            return p, rec, notes

        async def process_all(points_list):
            sem = asyncio.Semaphore(max(1, int(research_concurrency)))

            async def worker(p):
                async with sem:
                    return await process_point_async(p)

            tasks = [asyncio.create_task(worker(p)) for p in points_list]
            results = []
            for t in asyncio.as_completed(tasks):
                results.append(await t)
            return results

        # For non-openai providers concurrency brings little benefit; run sequentially
        if _prov == "openai":
            results = asyncio.run(process_all(points)) if points else []
        else:
            seq_results = []
            for p in points or []:
                seq_results.append(asyncio.run(process_point_async(p)))
            results = seq_results

        class _SimpleItem:
            def __init__(self, claim_text: str, verdict: str, reason: str, supporting_facts: str):
                self.claim_text = claim_text
                self.verdict = verdict
                self.reason = reason
                self.supporting_facts = supporting_facts

        simple_items = []
        for (p, r, notes) in results:
            if getattr(r, "action", "keep") == "keep":
                continue  # confirmed parts не передаём в переписывание
            if r.action == "clarify":
                verdict = "uncertain"
            elif r.action == "rewrite" or r.action == "remove":
                verdict = "fail"
            else:
                verdict = "fail"
            reason = getattr(r, "explanation", "") or ""
            supporting_facts = " \n".join(getattr(n, "findings", "") for n in (notes or []))
            simple_items.append(_SimpleItem(p.text, verdict, reason, supporting_facts))

        class _SimpleReport:
            def __init__(self, items):
                self.items = items

            def model_dump_json(self):
                import json
                return json.dumps(
                    {
                        "summary": "Issues only for rewrite (exclude confirmed)",
                        "items": [
                            {
                                "claim_text": i.claim_text,
                                "verdict": i.verdict,
                                "reason": i.reason,
                                "supporting_facts": i.supporting_facts,
                            }
                            for i in self.items
                        ],
                    },
                    ensure_ascii=False,
                )

        report = _SimpleReport(simple_items) if simple_items else None

    # Rewrite and refine
    final_content = content
    if report is not None:
        needs_rewrite = any(i.verdict != "pass" for i in report.items)
        if needs_rewrite:
            _emit("rewrite:init")
            from pathlib import Path
            p_rewrite = (Path(__file__).resolve().parents[2] / "prompts" / "post" / "module_03_rewriting" / "rewrite.md").read_text(encoding="utf-8")
            rw_input = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                f"<post>\n{content}\n</post>\n"
                f"<critique_json>\n{report.model_dump_json()}\n</critique_json>\n"
                "</input>"
            )
            final_content = run_with_provider(p_rewrite, rw_input) or content

    from pathlib import Path
    p_refine = (Path(__file__).resolve().parents[2] / "prompts" / "post" / "module_03_rewriting" / "refine.md").read_text(encoding="utf-8")
    refine_input = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<post>\n{final_content}\n</post>\n"
        "</input>"
    )
    final_content = run_with_provider(p_refine, refine_input) or final_content

    # Save final
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_post"
    filepath = next_available_filepath(output_dir, base, ".md")
    save_markdown(
        filepath,
        title=topic,
        generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
        pipeline="PopularSciencePost",
        content=final_content,
    )
    return filepath


