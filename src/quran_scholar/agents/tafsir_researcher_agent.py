"""Tafsir Researcher — fetch raw tafsir for selected verses (no LLM rewrite)."""

from __future__ import annotations

import re
from typing import Any

from quran_scholar.agents.helpers import (
    make_evidence,
    result,
    selected_verses,
    session_error,
    start_trace,
    task_id,
)
from quran_scholar.config import configured_tafsir_sources
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import ResearchPlan, TafsirEvidence, VerseRef
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace

_ATTRIB_RE = re.compile(
    r"^(?P<title>[^،,]+)[،,]?\s*(?P<author>.*?)\s*\(ت\.\s*(?P<death>[^)]+)\)\s*$"
)


def _parse_attribution(attr: str | None) -> tuple[str | None, str | None, str | None]:
    if not attr:
        return None, None, None
    match = _ATTRIB_RE.match(attr.strip())
    if not match:
        return attr, None, None
    return (
        match.group("title").strip() or None,
        match.group("author").strip() or None,
        match.group("death").strip() or None,
    )


def _sources(plan: ResearchPlan | None) -> list[str]:
    if plan and plan.target_tafsir_sources:
        return configured_tafsir_sources(plan.target_tafsir_sources)
    return configured_tafsir_sources(None)


def _tafsir_records(raw: Any) -> list[dict[str, Any]]:
    payload = mcp_payload(raw)
    if isinstance(payload, dict) and isinstance(payload.get("tafsirs"), list):
        return [x for x in payload["tafsirs"] if isinstance(x, dict)]
    return as_list(raw)


def run_tafsir_research(state: ResearchState) -> dict:
    """Retrieve tafsir. MCP failures are warnings — never 'no tafsir exists'."""
    tid = task_id(state)
    verses = selected_verses(state)
    sources = _sources(state.get("research_plan"))
    warnings: list[str] = []
    lines = start_trace("tafsir_researcher", "Retrieving tafsir...")

    if not verses:
        lines.append(trace("tafsir_researcher", "Skipped — no selected verses yet."))
        return result(
            tid,
            traces=lines,
            warnings=["tafsir_researcher: no selected_verses to fetch"],
        )

    items: list[TafsirEvidence] = []
    mcp_failures = 0
    empty_results = 0

    try:
        with ScopedTafsirMCPClient("tafsir") as client:
            for verse in verses:
                ref = verse.ref
                outcome = safe_call_tool(
                    client,
                    "fetch_tafsir",
                    {"surah": ref.surah, "ayah": ref.ayah, "sources": sources},
                    label=f"fetch_tafsir {ref.surah}:{ref.ayah}",
                )
                warnings.extend(outcome.warnings)
                if outcome.failed:
                    mcp_failures += 1
                    continue

                usable = [
                    rec
                    for rec in _tafsir_records(outcome.data)
                    if str(rec.get("text") or "").strip()
                ]
                if not usable:
                    empty_results += 1
                    warnings.extend(
                        mark_empty(
                            outcome,
                            f"NO_EVIDENCE: MCP succeeded but no tafsir text for "
                            f"{ref.surah}:{ref.ayah} (sources={sources})",
                        )
                    )
                    continue

                for rec in usable:
                    title, author, death = _parse_attribution(
                        rec.get("attribution")
                        if isinstance(rec.get("attribution"), str)
                        else None
                    )
                    items.append(
                        TafsirEvidence(
                            ref=VerseRef(surah=ref.surah, ayah=ref.ayah),
                            source_id=str(
                                rec.get("source") or rec.get("source_id") or "unknown"
                            ),
                            source_title=title,
                            author=author,
                            death_year_hijri=death,
                            text=str(rec.get("text") or ""),
                            source_tool="fetch_tafsir",
                            raw=rec,
                        )
                    )
    except MCPError as exc:
        return session_error("tafsir_researcher", "Tafsir retrieval", exc, tid, lines)

    n_sources = len({t.source_id for t in items})
    lines.append(
        trace(
            "tafsir_researcher",
            f"Retrieved {len(items)} excerpt(s) from {n_sources} source(s)"
            + (
                f" ({mcp_failures} MCP failure(s) — not treated as no tafsir)."
                if mcp_failures
                else "."
            ),
        )
    )
    evidence = [
        make_evidence(
            kind="tafsir",
            content=t.text,
            refs=[t.ref],
            id_prefix=f"tafsir-{t.source_id}-{t.ref.surah}-{t.ref.ayah}",
            metadata={
                "source_id": t.source_id,
                "source_title": t.source_title,
                "author": t.author,
                "death_year_hijri": t.death_year_hijri,
                "source_tool": t.source_tool,
                "raw": t.raw,
                "raw_tafsir": True,
            },
        )
        for t in items
    ]
    return result(
        tid,
        traces=lines,
        warnings=warnings
        + [
            f"tafsir_researcher: fetched={len(items)} "
            f"mcp_failures={mcp_failures} empty_results={empty_results} "
            f"sources={sources}"
        ],
        tafsir_evidence=items,
        evidence_items=evidence,
    )
