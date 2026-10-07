"""Context Researcher — asbab al-nuzool with FOUND / NOT_AVAILABLE / ERROR."""

from __future__ import annotations

import uuid
from typing import Any

from quran_scholar.mcp.client import TafsirMCPClient, TafsirMCPError
from quran_scholar.mcp.parse import mcp_payload
from quran_scholar.models import Evidence, NuzoolEvidence, ResearchPlan, VerseEvidence
from quran_scholar.state import ResearchState


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
    return Evidence(
        id=f"nuzool-{item.surah_number}-{item.ayah_number}-{uuid.uuid4().hex[:8]}",
        kind="nuzool",
        content=item.content,
        citation=item.source or "asbab al-nuzool",
        refs=[item.ref],
        metadata={
            "status": item.status,
            "source_tool": item.source_tool,
            "raw": item.raw,
        },
    )


def run_context_research(state: ResearchState) -> dict:
    """
    Fetch أسباب النزول for selected verses.

    Each verse gets an explicit status:
    - FOUND — text retrieved
    - NOT_AVAILABLE — MCP reports no asbab for this ayah
    - ERROR — tool/transport failure for this ayah
    Never collapse these into a single None.
    """
    task_id = state.get("current_task_id") or ""
    verses = _verses(state)
    warnings: list[str] = []
    errors: list[str] = []

    if not verses:
        return {
            "warnings": ["context_researcher: no selected_verses"],
            "completed_task_ids": [task_id] if task_id else [],
        }

    nuzool_items: list[NuzoolEvidence] = []

    try:
        client = TafsirMCPClient()
        client.initialize()
    except Exception as exc:
        # Session-level failure → ERROR for every requested verse
        for v in verses:
            nuzool_items.append(
                NuzoolEvidence(
                    status="ERROR",
                    content=str(exc),
                    source=None,
                    surah_number=v.ref.surah,
                    ayah_number=v.ref.ayah,
                    raw={"error": str(exc)},
                )
            )
        errors.append(f"context_researcher: MCP init failed: {exc}")
        return {
            "nuzool_evidence": nuzool_items,
            "errors": errors,
            "completed_task_ids": [task_id] if task_id else [],
        }

    try:
        for verse in verses:
            surah, ayah = verse.ref.surah, verse.ref.ayah
            try:
                raw = client.call_tool(
                    "fetch_nuzool_reason",
                    {"surah": surah, "ayah": ayah},
                )
                payload = mcp_payload(raw)
                if not isinstance(payload, dict):
                    nuzool_items.append(
                        NuzoolEvidence(
                            status="ERROR",
                            content="Unexpected MCP payload shape",
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
                    nuzool_items.append(
                        NuzoolEvidence(
                            status="NOT_AVAILABLE",
                            content="No nuzool sources returned for this ayah",
                            source=None,
                            surah_number=surah,
                            ayah_number=ayah,
                            raw=payload if isinstance(payload, dict) else {},
                        )
                    )
            except TafsirMCPError as exc:
                nuzool_items.append(
                    NuzoolEvidence(
                        status="ERROR",
                        content=str(exc),
                        source=None,
                        surah_number=surah,
                        ayah_number=ayah,
                        raw={"error": str(exc)},
                    )
                )
            except Exception as exc:
                nuzool_items.append(
                    NuzoolEvidence(
                        status="ERROR",
                        content=str(exc),
                        source=None,
                        surah_number=surah,
                        ayah_number=ayah,
                        raw={"error": str(exc)},
                    )
                )
    finally:
        client.close()

    evidence = [
        e for e in (_nuzool_to_evidence(n) for n in nuzool_items) if e is not None
    ]
    counts = {
        "FOUND": sum(1 for n in nuzool_items if n.status == "FOUND"),
        "NOT_AVAILABLE": sum(1 for n in nuzool_items if n.status == "NOT_AVAILABLE"),
        "ERROR": sum(1 for n in nuzool_items if n.status == "ERROR"),
    }
    updates: dict[str, Any] = {
        "nuzool_evidence": nuzool_items,
        "evidence_items": evidence,
        "warnings": warnings
        + [f"context_researcher: nuzool status counts={counts}"],
    }
    if task_id:
        updates["completed_task_ids"] = [task_id]
    if errors:
        updates["errors"] = errors
    return updates
