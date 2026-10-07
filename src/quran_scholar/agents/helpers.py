"""Shared helpers so researcher agents stay focused on domain logic."""

from __future__ import annotations

import uuid
from typing import Any

from quran_scholar.models import Evidence, ResearchPlan, VerseEvidence, VerseRef
from quran_scholar.services.citation_manager import citation_manager
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines


def task_id(state: ResearchState) -> str:
    return state.get("current_task_id") or ""


def selected_verses(state: ResearchState) -> list[VerseEvidence]:
    """Selected verses, or plan.primary_verse as a placeholder."""
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


def result(
    tid: str,
    *,
    traces: list[str] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Standard researcher state update."""
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


def start_trace(agent: str, message: str) -> list[str]:
    return [trace(agent, message, blank_before=True)]


def session_error(
    agent: str,
    what: str,
    exc: BaseException,
    tid: str,
    lines: list[str],
    **extra: Any,
) -> dict[str, Any]:
    """MCP session failure — never report as missing evidence."""
    msg = f"{what} failed (MCP session — not 'no evidence'): {exc}"
    lines.append(trace(agent, f"MCP session failed: {exc}"))
    return result(tid, traces=lines, warnings=[msg], errors=[msg], **extra)
