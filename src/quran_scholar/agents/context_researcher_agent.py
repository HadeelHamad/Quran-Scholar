"""Context Researcher — asbab al-nuzool with FOUND / NOT_AVAILABLE / ERROR."""

from __future__ import annotations

import uuid
from typing import Any

from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import Evidence, NuzoolEvidence, ResearchPlan, VerseEvidence
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines


def _verses(state: ResearchState) -> list[VerseEvidence]:
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


def _classify_nuzool_source(
    surah: int,
    ayah: int,
    entry: dict[str, Any],
) -> NuzoolEvidence:
    """Map one MCP nuzool source block to FOUND / NOT_AVAILABLE."""
    attribution = entry.get("attribution")
    source = (
        str(attribution)
        if attribution
        else str(entry.get("source") or "nuzool")
    )
    text = entry.get("text")
    available = entry.get("available")
    reason = entry.get("reason")

    # Explicit not available from MCP
    if available is False:
        return NuzoolEvidence(
            status="NOT_AVAILABLE",
            content=str(reason) if reason else "لم يثبت سبب نزول لهذه الآية",
            source=source,
            surah_number=surah,
            ayah_number=ayah,
            raw=entry,
        )

    # Found textual asbab
    if isinstance(text, str) and text.strip():
        return NuzoolEvidence(
            status="FOUND",
            content=text,
            source=source,
            surah_number=surah,
            ayah_number=ayah,
            isnad=str(entry.get("isnad")) if entry.get("isnad") else None,
            raw=entry,
        )

    # Ambiguous empty payload → treat as not available (not ERROR)
    return NuzoolEvidence(
        status="NOT_AVAILABLE",
        content=str(reason) if reason else "لا تتوفر بيانات سبب نزول لهذه الآية",
        source=source,
        surah_number=surah,
        ayah_number=ayah,
        raw=entry,
    )


def _nuzool_to_evidence(item: NuzoolEvidence) -> Evidence | None:
    """Only FOUND nuzool becomes positive evidence; others stay in nuzool_evidence."""
    if item.status != "FOUND" or not item.content:
        return None
    from quran_scholar.services.citation_manager import citation_manager

    ev = Evidence(
        id=f"nuzool-{item.surah_number}-{item.ayah_number}-{uuid.uuid4().hex[:8]}",
        kind="nuzool",
        content=item.content,
        citation="",
        refs=[item.ref],
        metadata={
            "status": item.status,
            "source_id": "nuzool",
            "attribution": item.source,
            "source_tool": item.source_tool,
            "raw": item.raw,
        },
    )
    return ev.model_copy(update={"citation": citation_manager.format(ev).label})


def run_context_research(state: ResearchState) -> dict:
    """
    Fetch أسباب النزول for selected verses.

    Each verse gets an explicit status:
    - FOUND — text retrieved
    - NOT_AVAILABLE — MCP request succeeded; no asbab for this ayah
    - ERROR — MCP request failed (never treat as NOT_AVAILABLE)
    """
    task_id = state.get("current_task_id") or ""
    verses = _verses(state)
    warnings: list[str] = []
    errors: list[str] = []
    lines: list[str] = [
        trace(
            "context_researcher",
            "Checking asbab al-nuzool...",
            blank_before=True,
        )
    ]

    if not verses:
        lines.append(
            trace("context_researcher", "Skipped — no selected verses.")
        )
        return {
            "warnings": ["context_researcher: no selected_verses"],
            "completed_task_ids": [task_id] if task_id else [],
            **trace_lines(*lines),
        }

    nuzool_items: list[NuzoolEvidence] = []

    try:
        with ScopedTafsirMCPClient("context") as client:
            for verse in verses:
                surah, ayah = verse.ref.surah, verse.ref.ayah
                outcome = safe_call_tool(
                    client,
                    "fetch_nuzool_reason",
                    {"surah": surah, "ayah": ayah},
                    label=f"fetch_nuzool_reason {surah}:{ayah}",
                )
                warnings.extend(outcome.warnings)

                if outcome.failed:
                    nuzool_items.append(
                        NuzoolEvidence(
                            status="ERROR",
                            content=outcome.error or "MCP request failed",
                            source=None,
                            surah_number=surah,
                            ayah_number=ayah,
                            raw={"error": outcome.error, "mcp_failed": True},
                        )
                    )
                    continue

                payload = mcp_payload(outcome.data)
                if not isinstance(payload, dict):
                    warnings.extend(
                        mark_empty(
                            outcome,
                            f"NO_EVIDENCE: MCP succeeded but unexpected payload "
                            f"for nuzool {surah}:{ayah}",
                        )
                    )
                    nuzool_items.append(
                        NuzoolEvidence(
                            status="NOT_AVAILABLE",
                            content="Unexpected MCP payload shape (request succeeded)",
                            source=None,
                            surah_number=surah,
                            ayah_number=ayah,
                            raw={"payload": payload},
                        )
                    )
                    continue

                sources = payload.get("sources")
                if isinstance(sources, list) and sources:
                    for entry in sources:
                        if isinstance(entry, dict):
                            nuzool_items.append(
                                _classify_nuzool_source(surah, ayah, entry)
                            )
                else:
                    warnings.extend(
                        mark_empty(
                            outcome,
                            f"NO_EVIDENCE: MCP succeeded but no nuzool sources "
                            f"for {surah}:{ayah}",
                        )
                    )
                    nuzool_items.append(
                        NuzoolEvidence(
                            status="NOT_AVAILABLE",
                            content="No nuzool sources returned for this ayah",
                            source=None,
                            surah_number=surah,
                            ayah_number=ayah,
                            raw=payload,
                        )
                    )
    except MCPError as exc:
        msg = f"Nuzool retrieval failed (MCP session — not 'no nuzool'): {exc}"
        lines.append(trace("context_researcher", f"MCP session failed: {exc}"))
        for v in verses:
            nuzool_items.append(
                NuzoolEvidence(
                    status="ERROR",
                    content=msg,
                    source=None,
                    surah_number=v.ref.surah,
                    ayah_number=v.ref.ayah,
                    raw={"error": str(exc)},
                )
            )
        return {
            "nuzool_evidence": nuzool_items,
            "warnings": [msg],
            "errors": [msg],
            "completed_task_ids": [task_id] if task_id else [],
            **trace_lines(*lines),
        }

    evidence = [
        e for e in (_nuzool_to_evidence(n) for n in nuzool_items) if e is not None
    ]
    counts = {
        "FOUND": sum(1 for n in nuzool_items if n.status == "FOUND"),
        "NOT_AVAILABLE": sum(1 for n in nuzool_items if n.status == "NOT_AVAILABLE"),
        "ERROR": sum(1 for n in nuzool_items if n.status == "ERROR"),
    }
    lines.append(
        trace(
            "context_researcher",
            f"Nuzool results — FOUND={counts['FOUND']}, "
            f"NOT_AVAILABLE={counts['NOT_AVAILABLE']}, "
            f"ERROR={counts['ERROR']}.",
        )
    )
    updates: dict[str, Any] = {
        "nuzool_evidence": nuzool_items,
        "evidence_items": evidence,
        "warnings": warnings
        + [f"context_researcher: nuzool status counts={counts}"],
        **trace_lines(*lines),
    }
    if task_id:
        updates["completed_task_ids"] = [task_id]
    if errors:
        updates["errors"] = errors
    return updates
