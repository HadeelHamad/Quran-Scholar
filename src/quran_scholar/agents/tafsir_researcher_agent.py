"""Tafsir Researcher — fetch raw tafsir for selected verses (no LLM rewrite)."""

from __future__ import annotations

import re
import uuid
from typing import Any

from quran_scholar.config import configured_tafsir_sources
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import (
    Evidence,
    ResearchPlan,
    TafsirEvidence,
    VerseEvidence,
    VerseRef,
)
from quran_scholar.state import ResearchState

# Attribution often looks like: "تفسير ابن كثير، أبو الفداء … (ت. 774هـ)"
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


def _verses_for_tafsir(state: ResearchState) -> list[VerseEvidence]:
    selected = list(state.get("selected_verses") or [])
    if selected:
        return selected
    plan: ResearchPlan | None = state.get("research_plan")
    if plan and plan.primary_verse:
        return [
            VerseEvidence(
                ref=plan.primary_verse,
                text_uthmani="",
                source_tool="plan.primary_verse",
            )
        ]
    return []


def _prioritize_sources(plan: ResearchPlan | None) -> list[str]:
    """
    Configurable sources; prefer plan targets when set (e.g. only katheer).
    Do not blindly expand to all defaults when the plan already narrowed sources.
    """
    if plan and plan.target_tafsir_sources:
        return configured_tafsir_sources(plan.target_tafsir_sources)
    return configured_tafsir_sources(None)


def _tafsir_to_evidence(t: TafsirEvidence) -> Evidence:
    from quran_scholar.services.citation_manager import citation_manager

    ev = Evidence(
        id=f"tafsir-{t.source_id}-{t.ref.surah}-{t.ref.ayah}-{uuid.uuid4().hex[:8]}",
        kind="tafsir",
        content=t.text,  # raw retrieved text — never an LLM summary
        citation="",  # filled by citation_manager
        refs=[t.ref],
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
    return ev.model_copy(update={"citation": citation_manager.format(ev).label})


def _records_from_payload(raw: Any) -> list[dict[str, Any]]:
    payload = mcp_payload(raw)
    if isinstance(payload, dict) and isinstance(payload.get("tafsirs"), list):
        return [x for x in payload["tafsirs"] if isinstance(x, dict)]
    return as_list(raw)


def run_tafsir_research(state: ResearchState) -> dict:
    """
    Retrieve tafsir for selected verses.

    MCP failures become warnings — never interpreted as "no tafsir exists."
    """
    plan: ResearchPlan | None = state.get("research_plan")
    task_id = state.get("current_task_id") or ""
    verses = _verses_for_tafsir(state)
    sources = _prioritize_sources(plan)
    warnings: list[str] = []
    errors: list[str] = []
    mcp_failures = 0
    empty_results = 0

    if not verses:
        return {
            "warnings": ["tafsir_researcher: no selected_verses to fetch"],
            "completed_task_ids": [task_id] if task_id else [],
        }

    tafsir_items: list[TafsirEvidence] = []

    try:
        client_cm = ScopedTafsirMCPClient("tafsir")
        client_cm.__enter__()
    except MCPError as exc:
        # Session failure — do NOT claim tafsir is absent
        msg = f"Tafsir retrieval failed (MCP session — not 'no tafsir'): {exc}"
        return {
            "warnings": [msg],
            "errors": [msg],
            "completed_task_ids": [task_id] if task_id else [],
        }
    except Exception as exc:
        msg = f"Tafsir retrieval failed (session — not 'no tafsir'): {exc}"
        return {
            "warnings": [msg],
            "errors": [msg],
            "completed_task_ids": [task_id] if task_id else [],
        }

    try:
        for verse in verses:
            ref = verse.ref
            label = f"fetch_tafsir {ref.surah}:{ref.ayah}"
            outcome = safe_call_tool(
                client_cm,
                "fetch_tafsir",
                {
                    "surah": ref.surah,
                    "ayah": ref.ayah,
                    "sources": sources,
                },
                label=label,
            )
            warnings.extend(outcome.warnings)

            if outcome.failed:
                mcp_failures += 1
                # Explicit: failure ≠ absence of tafsir
                continue

            records = _records_from_payload(outcome.data)
            usable = [
                rec
                for rec in records
                if str(rec.get("text") or "").strip()
            ]
            if not usable:
                empty_results += 1
                outcome = mark_empty(
                    outcome,
                    message=(
                        f"NO_EVIDENCE: MCP succeeded but no tafsir text for "
                        f"{ref.surah}:{ref.ayah} (sources={sources})"
                    ),
                )
                warnings.extend(outcome.warnings)
                continue

            for rec in usable:
                source_id = str(rec.get("source") or rec.get("source_id") or "unknown")
                text = str(rec.get("text") or "")
                title, author, death = _parse_attribution(
                    rec.get("attribution")
                    if isinstance(rec.get("attribution"), str)
                    else None
                )
                tafsir_items.append(
                    TafsirEvidence(
                        ref=VerseRef(surah=ref.surah, ayah=ref.ayah),
                        source_id=source_id,
                        source_title=title,
                        author=author,
                        death_year_hijri=death,
                        text=text,
                        source_tool="fetch_tafsir",
                        raw=rec,
                    )
                )
    finally:
        client_cm.__exit__(None, None, None)

    evidence = [_tafsir_to_evidence(t) for t in tafsir_items]
    summary = (
        f"tafsir_researcher: fetched={len(tafsir_items)} "
        f"mcp_failures={mcp_failures} empty_results={empty_results} "
        f"sources={sources}"
    )
    updates: dict[str, Any] = {
        "tafsir_evidence": tafsir_items,
        "evidence_items": evidence,
        "warnings": warnings + [summary],
    }
    if task_id:
        updates["completed_task_ids"] = [task_id]
    if errors:
        updates["errors"] = errors
    return updates
