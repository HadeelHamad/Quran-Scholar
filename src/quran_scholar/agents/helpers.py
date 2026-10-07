"""Shared pack/evidence helpers for researcher agents (tool-calling)."""

from __future__ import annotations

import uuid
from typing import Any

from quran_scholar.models import Evidence, ResearchPlan, VerseEvidence, VerseRef
from quran_scholar.state import ResearchState


def selected_verses(state: ResearchState) -> list[VerseEvidence]:
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


def _citation_label(
    kind: str, refs: list[VerseRef], metadata: dict[str, Any]
) -> str:
    source = metadata.get("source_id") or metadata.get("attribution") or kind
    if refs:
        verse = ", ".join(f"{r.surah}:{r.ayah}" for r in refs)
        return f"[{source} {verse}]"
    return f"[{source}]"


def make_evidence(
    *,
    kind: str,
    content: str,
    refs: list[VerseRef],
    metadata: dict[str, Any] | None = None,
    id_prefix: str | None = None,
) -> Evidence:
    meta = metadata or {}
    return Evidence(
        id=f"{id_prefix or kind}-{uuid.uuid4().hex[:10]}",
        kind=kind,
        content=content,
        citation=_citation_label(kind, refs, meta),
        refs=refs,
        metadata=meta,
    )


def pack(
    tid: str,
    *,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a researcher state update."""
    out: dict[str, Any] = dict(fields)
    if warnings is not None:
        out["warnings"] = warnings
    if errors is not None:
        out["errors"] = errors
    if tid:
        out["completed_task_ids"] = [tid]
    return out


def session_fail(
    agent: str,
    what: str,
    exc: BaseException,
    tid: str,
    **extra: Any,
) -> dict[str, Any]:
    msg = f"{what} failed (MCP session — not 'no evidence'): {exc}"
    return pack(tid, warnings=[msg], errors=[msg], **extra)
