"""Tafsir Researcher — fetch raw tafsir for selected verses (no LLM rewrite)."""

from __future__ import annotations

import re
import uuid
from typing import Any

from quran_scholar.config import configured_tafsir_sources
from quran_scholar.mcp.client import TafsirMCPClient
from quran_scholar.mcp.parse import as_list, mcp_payload
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
    citation = t.source_title or t.source_id
    if t.author:
        citation = f"{citation} — {t.author}"
    if t.death_year_hijri:
        citation = f"{citation} (ت. {t.death_year_hijri})"
    return Evidence(
        id=f"tafsir-{t.source_id}-{t.ref.surah}-{t.ref.ayah}-{uuid.uuid4().hex[:8]}",
        kind="tafsir",
        content=t.text,  # raw retrieved text — never an LLM summary
        citation=f"{citation} on {t.ref.surah}:{t.ref.ayah}",
        refs=[t.ref],
        metadata={
            "source_id": t.source_id,
            "source_tool": t.source_tool,
            "raw": t.raw,
            "raw_tafsir": True,
        },
    )


def run_tafsir_research(state: ResearchState) -> dict:
    """
    Retrieve tafsir for selected verses.

    Stores raw MCP tafsir in tafsir_evidence + evidence_items.
    Does NOT write LLM findings here — analysis nodes own findings.
    """
    plan: ResearchPlan | None = state.get("research_plan")
    task_id = state.get("current_task_id") or ""
    verses = _verses_for_tafsir(state)
    sources = _prioritize_sources(plan)
    warnings: list[str] = []
    errors: list[str] = []

    if not verses:
        return {
            "warnings": ["tafsir_researcher: no selected_verses to fetch"],
            "completed_task_ids": [task_id] if task_id else [],
        }

    tafsir_items: list[TafsirEvidence] = []

    try:
        with TafsirMCPClient() as client:
            for verse in verses:
                ref = verse.ref
                raw = client.call_tool(
                    "fetch_tafsir",
                    {
                        "surah": ref.surah,
                        "ayah": ref.ayah,
                        "sources": sources,
                    },
                )
                payload = mcp_payload(raw)
                records: list[dict[str, Any]]
                if isinstance(payload, dict) and isinstance(payload.get("tafsirs"), list):
                    records = [x for x in payload["tafsirs"] if isinstance(x, dict)]
                else:
                    records = as_list(raw)

                if not records:
                    warnings.append(
                        f"tafsir_researcher: no tafsir for {ref.surah}:{ref.ayah}"
                    )
                    continue

                for rec in records:
                    source_id = str(rec.get("source") or rec.get("source_id") or "unknown")
                    text = str(rec.get("text") or "")
                    if not text.strip():
                        continue
                    title, author, death = _parse_attribution(
                        rec.get("attribution") if isinstance(rec.get("attribution"), str) else None
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
                            raw=rec,  # raw retrieved block preserved
                        )
                    )
    except Exception as exc:
        errors.append(f"tafsir_researcher: MCP error: {exc}")
        return {
            "errors": errors,
            "warnings": warnings,
            "completed_task_ids": [task_id] if task_id else [],
        }

    evidence = [_tafsir_to_evidence(t) for t in tafsir_items]
    updates: dict[str, Any] = {
        "tafsir_evidence": tafsir_items,
        "evidence_items": evidence,
        "warnings": warnings
        + [
            f"tafsir_researcher: fetched {len(tafsir_items)} raw tafsir excerpt(s) "
            f"from sources={sources} for {len(verses)} verse(s)"
        ],
    }
    if task_id:
        updates["completed_task_ids"] = [task_id]
    if errors:
        updates["errors"] = errors
    return updates
