"""Context Researcher — create_agent + fetch_nuzool_reason (asbab al-nuzool)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.helpers import (
    make_evidence,
    pack,
    selected_verses,
    session_fail,
)
from quran_scholar.agents.mcp_agent import (
    has_llm_credentials,
    parse_tool_json,
    run_researcher_agent,
    tool_had_mcp_error,
)
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import NuzoolEvidence
from quran_scholar.state import ResearchState

CONTEXT_SYSTEM = """You are the Context Researcher for Quran Scholar (asbab al-nuzool).

You have the MCP tool: fetch_nuzool_reason.
Call it for the verse(s) in the brief that need revelation-context evidence.
You may skip a verse only if the brief says nuzool is irrelevant for it.
NEVER invent reasons of revelation — only tool results.
"""


class ContextAgentSummary(BaseModel):
    notes: str = ""
    verses_checked: list[str] = Field(default_factory=list)


def _classify(surah: int, ayah: int, entry: dict[str, Any]) -> NuzoolEvidence:
    source = str(entry.get("attribution") or entry.get("source") or "nuzool")
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


def _items_from_tool_calls(tool_calls: list) -> tuple[list[NuzoolEvidence], list[str]]:
    items: list[NuzoolEvidence] = []
    warnings: list[str] = []
    for call in tool_calls:
        if call.name != "fetch_nuzool_reason":
            continue
        surah = call.args.get("surah")
        ayah = call.args.get("ayah")
        if surah is None or ayah is None:
            warnings.append("context_researcher: fetch_nuzool_reason missing surah/ayah")
            continue
        surah_i, ayah_i = int(surah), int(ayah)
        payload = parse_tool_json(call.content)
        if tool_had_mcp_error(payload):
            items.append(
                NuzoolEvidence(
                    status="ERROR",
                    content=(
                        payload.get("error")
                        if isinstance(payload, dict)
                        else "MCP request failed"
                    ),
                    source=None,
                    surah_number=surah_i,
                    ayah_number=ayah_i,
                    raw={"error": payload, "mcp_failed": True},
                )
            )
            continue
        data = mcp_payload(payload) if not isinstance(payload, str) else payload
        if not isinstance(data, dict):
            warnings.append(
                f"NO_EVIDENCE: unexpected nuzool payload for {surah_i}:{ayah_i}"
            )
            items.append(
                NuzoolEvidence(
                    status="NOT_AVAILABLE",
                    content="Unexpected MCP payload shape (request succeeded)",
                    source=None,
                    surah_number=surah_i,
                    ayah_number=ayah_i,
                    raw={"payload": data},
                )
            )
            continue
        sources = data.get("sources")
        if isinstance(sources, list) and sources:
            for entry in sources:
                if isinstance(entry, dict):
                    items.append(_classify(surah_i, ayah_i, entry))
        else:
            warnings.append(f"NO_EVIDENCE: no nuzool sources for {surah_i}:{ayah_i}")
            items.append(
                NuzoolEvidence(
                    status="NOT_AVAILABLE",
                    content="No nuzool sources returned for this ayah",
                    source=None,
                    surah_number=surah_i,
                    ayah_number=ayah_i,
                    raw=data,
                )
            )
    return items, warnings


def _deterministic_nuzool(verses: list) -> tuple[list[NuzoolEvidence], list[str]]:
    items: list[NuzoolEvidence] = []
    warnings: list[str] = []
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
    return items, warnings


def run_context_research(state: ResearchState) -> dict:
    tid = state.get("current_task_id") or ""
    verses = selected_verses(state)
    question = state.get("user_question") or ""
    warnings: list[str] = []
    if not verses:
        return pack(
            tid,
            warnings=["context_researcher: no selected_verses"],
        )

    brief = {
        "question": question,
        "verses": [{"surah": v.ref.surah, "ayah": v.ref.ayah} for v in verses],
    }
    user_msg = (
        f"Brief:\n{json.dumps(brief, ensure_ascii=False)}\n\n"
        "Call fetch_nuzool_reason for relevant verses."
    )

    items: list[NuzoolEvidence] = []
    tools_used: list[str] = []

    try:
        if has_llm_credentials():
            agent_out = run_researcher_agent(
                role="context",
                system_prompt=CONTEXT_SYSTEM,
                user_message=user_msg,
                response_format=ContextAgentSummary,
                name="context_researcher",
            )
            warnings.extend(agent_out.warnings)
            tools_used = agent_out.tools_used
            items, w2 = _items_from_tool_calls(agent_out.tool_calls)
            warnings.extend(w2)
            if not items:
                warnings.append(
                    "context_researcher: agent empty — deterministic fallback"
                )
                items, w3 = _deterministic_nuzool(verses)
                warnings.extend(w3)
        else:
            items, w = _deterministic_nuzool(verses)
            warnings.extend(w)
            tools_used = ["fetch_nuzool_reason"]
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
            nuzool_evidence=items,
        )

    counts = {
        "FOUND": sum(1 for n in items if n.status == "FOUND"),
        "NOT_AVAILABLE": sum(1 for n in items if n.status == "NOT_AVAILABLE"),
        "ERROR": sum(1 for n in items if n.status == "ERROR"),
    }
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
        warnings=warnings
        + [f"context_researcher: nuzool status counts={counts} tools={tools_used}"],
        nuzool_evidence=items,
        evidence_items=evidence,
    )
