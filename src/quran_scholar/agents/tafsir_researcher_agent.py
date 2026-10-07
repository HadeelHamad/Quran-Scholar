"""Tafsir Researcher — fetch raw tafsir for selected verses (no LLM rewrite)."""

from __future__ import annotations

import re
from quran_scholar.agents.helpers import make_evidence, pack, selected_verses, session_fail
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


def run_tafsir_research(state: ResearchState) -> dict:
    """Retrieve tafsir. MCP failures are warnings — never 'no tafsir exists'."""
    tid = state.get("current_task_id") or ""
    verses = selected_verses(state)
    plan: ResearchPlan | None = state.get("research_plan")
    sources = configured_tafsir_sources(
        plan.target_tafsir_sources if plan and plan.target_tafsir_sources else None
    )
    warnings: list[str] = []
    lines = [trace("tafsir_researcher", "Retrieving tafsir...", blank_before=True)]

    if not verses:
        lines.append(trace("tafsir_researcher", "Skipped — no selected verses yet."))
        return pack(
            tid,
            lines=lines,
            warnings=["tafsir_researcher: no selected_verses to fetch"],
        )

    items: list[TafsirEvidence] = []
    mcp_failures = empty_results = 0

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

                payload = mcp_payload(outcome.data)
                records = (
                    [x for x in payload["tafsirs"] if isinstance(x, dict)]
                    if isinstance(payload, dict)
                    and isinstance(payload.get("tafsirs"), list)
                    else as_list(outcome.data)
                )
                usable = [r for r in records if str(r.get("text") or "").strip()]
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
                    attr = rec.get("attribution")
                    title = author = death = None
                    if isinstance(attr, str) and attr.strip():
                        match = _ATTRIB_RE.match(attr.strip())
                        if match:
                            title = match.group("title").strip() or None
                            author = match.group("author").strip() or None
                            death = match.group("death").strip() or None
                        else:
                            title = attr
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
        return session_fail("tafsir_researcher", "Tafsir retrieval", exc, tid, lines)

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
    return pack(
        tid,
        lines=lines,
        warnings=warnings
        + [
            f"tafsir_researcher: fetched={len(items)} "
            f"mcp_failures={mcp_failures} empty_results={empty_results} "
            f"sources={sources}"
        ],
        tafsir_evidence=items,
        evidence_items=[
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
        ],
    )
