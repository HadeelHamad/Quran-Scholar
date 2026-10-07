"""Shared pack/evidence helpers for researcher agents (tool-calling)."""

from __future__ import annotations

import uuid
from typing import Any

from quran_scholar.models import Evidence, ResearchPlan, VerseEvidence, VerseRef
from quran_scholar.services.citation_manager import citation_manager
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines


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


def make_evidence(
    *,
    kind: str,
    content: str,
    refs: list[VerseRef],
    metadata: dict[str, Any] | None = None,
    id_prefix: str | None = None,
) -> Evidence:
    ev = Evidence(
        id=f"{id_prefix or kind}-{uuid.uuid4().hex[:10]}",
        kind=kind,
        content=content,
        citation="",
        refs=refs,
        metadata=metadata or {},
    )
    return ev.model_copy(update={"citation": citation_manager.format(ev).label})


def pack(
    tid: str,
    *,
    agent: str | None = None,
    message: str | None = None,
    lines: list[str] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a researcher state update. Optionally starts a trace line."""
    traces = list(lines or [])
    if agent and message:
        traces.insert(0, trace(agent, message, blank_before=True))
    out: dict[str, Any] = dict(fields)
    if warnings is not None:
        out["warnings"] = warnings
    if errors is not None:
        out["errors"] = errors
    if traces:
        out.update(trace_lines(*traces))
    if tid:
        out["completed_task_ids"] = [tid]
    return out


def session_fail(
    agent: str,
    what: str,
    exc: BaseException,
    tid: str,
    lines: list[str],
    **extra: Any,
) -> dict[str, Any]:
    msg = f"{what} failed (MCP session — not 'no evidence'): {exc}"
    lines.append(trace(agent, f"MCP session failed: {exc}"))
    return pack(tid, lines=lines, warnings=[msg], errors=[msg], **extra)
