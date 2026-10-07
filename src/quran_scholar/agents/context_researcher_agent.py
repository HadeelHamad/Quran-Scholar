"""Context Researcher — asbab al-nuzool with FOUND / NOT_AVAILABLE / ERROR."""

from __future__ import annotations

from typing import Any

from quran_scholar.agents.helpers import (
    make_evidence,
    pack,
    selected_verses,
    session_fail,
)
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import NuzoolEvidence
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace


def _classify(surah: int, ayah: int, entry: dict[str, Any]) -> NuzoolEvidence:
    source = str(
        entry.get("attribution") or entry.get("source") or "nuzool"
    )
    text = entry.get("text")
    reason = entry.get("reason")

    if entry.get("available") is False:
        return NuzoolEvidence(
            status="NOT_AVAILABLE",
            content=str(reason) if reason else "لم يثبت سبب نزول لهذه الآية",
            source=source,
            surah_number=surah,
            ayah_number=ayah,
            raw=entry,
        )
    if isinstance(text, str) and text.strip():
        return NuzoolEvidence(
            status="FOUND",
            content=text,
            source=source,
            surah_number=surah,
            ayah_number=ayah,
            isnad=str(entry["isnad"]) if entry.get("isnad") else None,
            raw=entry,
        )
    return NuzoolEvidence(
        status="NOT_AVAILABLE",
        content=str(reason) if reason else "لا تتوفر بيانات سبب نزول لهذه الآية",
        source=source,
        surah_number=surah,
        ayah_number=ayah,
        raw=entry,
    )


def run_context_research(state: ResearchState) -> dict:
    """Fetch أسباب النزول. ERROR = MCP failed; NOT_AVAILABLE = empty success."""
    tid = state.get("current_task_id") or ""
    verses = selected_verses(state)
    warnings: list[str] = []
    lines = [trace("context_researcher", "Checking asbab al-nuzool...", blank_before=True)]

    if not verses:
        lines.append(trace("context_researcher", "Skipped — no selected verses."))
        return pack(
            tid,
            lines=lines,
            warnings=["context_researcher: no selected_verses"],
        )

    items: list[NuzoolEvidence] = []

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
                    items.append(
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
                            f"NO_EVIDENCE: unexpected nuzool payload for {surah}:{ayah}",
                        )
                    )
                    items.append(
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
                            items.append(_classify(surah, ayah, entry))
                else:
                    warnings.extend(
                        mark_empty(
                            outcome,
                            f"NO_EVIDENCE: no nuzool sources for {surah}:{ayah}",
                        )
                    )
                    items.append(
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
        for v in verses:
            items.append(
                NuzoolEvidence(
                    status="ERROR",
                    content=str(exc),
                    source=None,
                    surah_number=v.ref.surah,
                    ayah_number=v.ref.ayah,
                    raw={"error": str(exc)},
                )
            )
        return session_fail(
            "context_researcher",
            "Nuzool retrieval",
            exc,
            tid,
            lines,
            nuzool_evidence=items,
        )

    counts = {
        "FOUND": sum(1 for n in items if n.status == "FOUND"),
        "NOT_AVAILABLE": sum(1 for n in items if n.status == "NOT_AVAILABLE"),
        "ERROR": sum(1 for n in items if n.status == "ERROR"),
    }
    lines.append(
        trace(
            "context_researcher",
            f"Nuzool results — FOUND={counts['FOUND']}, "
            f"NOT_AVAILABLE={counts['NOT_AVAILABLE']}, ERROR={counts['ERROR']}.",
        )
    )
    evidence = [
        make_evidence(
            kind="nuzool",
            content=n.content or "",
            refs=[n.ref],
            id_prefix=f"nuzool-{n.surah_number}-{n.ayah_number}",
            metadata={
                "status": n.status,
                "source_id": "nuzool",
                "attribution": n.source,
                "source_tool": n.source_tool,
                "raw": n.raw,
            },
        )
        for n in items
        if n.status == "FOUND" and n.content
    ]
    return pack(
        tid,
        lines=lines,
        warnings=warnings + [f"context_researcher: nuzool status counts={counts}"],
        nuzool_evidence=items,
        evidence_items=evidence,
    )
