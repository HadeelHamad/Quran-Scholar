"""Tafsir Researcher — create_agent + MCP tools (fetch_tafsir / search_in_tafsir)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.graph.nodes.helpers import make_evidence, pack, selected_verses
from quran_scholar.graph.nodes.mcp_agent import (
    has_llm_credentials,
    parse_tool_json,
    run_researcher_agent,
    tool_had_mcp_error,
)
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.models import ResearchPlan, TafsirEvidence, VerseRef
from quran_scholar.state import ResearchState

_ATTRIB_RE = re.compile(
    r"^(?P<title>[^،,]+)[،,]?\s*(?P<author>.*?)\s*\(ت\.\s*(?P<death>[^)]+)\)\s*$"
)

TAFSIR_SYSTEM = """You are the Tafsir Researcher for Quran Scholar.

You run only when the plan needs classical commentary. MCP tools:
fetch_tafsir, search_in_tafsir, list_tafsir_sources, list_sources_for_ayah.

Guidelines:
- Known verses → fetch_tafsir (pass requested source ids when given).
- Thematic commentary search → search_in_tafsir.
- Use list_* tools if you need to discover available sources for an ayah.
- NEVER invent tafsir text — only tool results.
- Summarize retrieval (counts/sources) in structured notes; do not rewrite tafsir.
"""


class TafsirAgentSummary(BaseModel):
    notes: str = Field(
        default="",
        description="Brief note on which tools were used and why",
    )
    verse_refs_covered: list[str] = Field(
        default_factory=list,
        description="surah:ayah strings covered from tool results",
    )


def _parse_attribution(attr: Any) -> tuple[str | None, str | None, str | None]:
    title = author = death = None
    if isinstance(attr, str) and attr.strip():
        match = _ATTRIB_RE.match(attr.strip())
        if match:
            title = match.group("title").strip() or None
            author = match.group("author").strip() or None
            death = match.group("death").strip() or None
        else:
            title = attr
    return title, author, death


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("tafsirs"), list):
            return [x for x in payload["tafsirs"] if isinstance(x, dict)]
        if isinstance(payload.get("results"), list):
            return [x for x in payload["results"] if isinstance(x, dict)]
        if isinstance(payload.get("hits"), list):
            return [x for x in payload["hits"] if isinstance(x, dict)]
    return [x for x in as_list(payload) if isinstance(x, dict)]


def _record_to_tafsir(
    rec: dict[str, Any],
    *,
    source_tool: str,
    fallback_ref: VerseRef | None = None,
) -> TafsirEvidence | None:
    text = str(rec.get("text") or rec.get("snippet") or "").strip()
    if not text:
        return None
    surah = rec.get("surah")
    ayah = rec.get("ayah")
    if surah is not None and ayah is not None:
        ref = VerseRef(surah=int(surah), ayah=int(ayah))
    elif fallback_ref is not None:
        ref = fallback_ref
    else:
        return None
    title, author, death = _parse_attribution(rec.get("attribution"))
    return TafsirEvidence(
        ref=ref,
        source_id=str(rec.get("source") or rec.get("source_id") or "unknown"),
        source_title=title,
        author=author,
        death_year_hijri=death,
        text=text,
        source_tool=source_tool,
        raw=rec,
    )


def _items_from_tool_calls(tool_calls: list) -> tuple[list[TafsirEvidence], list[str], int]:
    items: list[TafsirEvidence] = []
    warnings: list[str] = []
    mcp_failures = 0
    for call in tool_calls:
        if call.name not in ("fetch_tafsir", "search_in_tafsir"):
            continue
        payload = parse_tool_json(call.content)
        if tool_had_mcp_error(payload):
            mcp_failures += 1
            warnings.append(
                f"tafsir_researcher: MCP error on {call.name}: "
                f"{payload.get('error') if isinstance(payload, dict) else call.content}"
            )
            continue
        data = mcp_payload(payload) if not isinstance(payload, str) else payload
        fallback: VerseRef | None = None
        if call.name == "fetch_tafsir" and call.args:
            try:
                fallback = VerseRef(
                    surah=int(call.args["surah"]), ayah=int(call.args["ayah"])
                )
            except (KeyError, TypeError, ValueError):
                fallback = None
        records = _records_from_payload(data)
        if not records:
            warnings.append(
                f"NO_EVIDENCE: {call.name} succeeded but returned no tafsir records"
            )
            continue
        for rec in records:
            item = _record_to_tafsir(rec, source_tool=call.name, fallback_ref=fallback)
            if item:
                items.append(item)
    return items, warnings, mcp_failures


def run_tafsir_research(state: ResearchState) -> dict:
    """Retrieve tafsir via create_agent tool-calling only."""
    tid = state.get("current_task_id") or ""
    verses = selected_verses(state)
    plan: ResearchPlan | None = state.get("research_plan")
    if plan and plan.target_tafsir_sources:
        sources = list(plan.target_tafsir_sources)
    else:
        env = os.getenv("TAFSIR_SOURCES", "").strip()
        sources = (
            [s.strip() for s in env.split(",") if s.strip()]
            if env
            else ["tabary", "katheer", "baghawy", "saadi", "moyassar"]
        )
    question = state.get("user_question") or ""
    warnings: list[str] = []
    if not verses:
        return pack(
            tid,
            warnings=["tafsir_researcher: no selected_verses to fetch"],
        )

    if not has_llm_credentials():
        return pack(
            tid,
            warnings=["tafsir_researcher: no OPENAI_API_KEY — cannot run agent"],
            errors=["tafsir_researcher: no OPENAI_API_KEY — cannot run agent"],
        )

    verse_brief = [
        {
            "surah": v.ref.surah,
            "ayah": v.ref.ayah,
            "text": (v.text_uthmani or "")[:200],
        }
        for v in verses
    ]
    user_msg = (
        f"User question (Arabic):\n{question}\n\n"
        f"Selected verses:\n{json.dumps(verse_brief, ensure_ascii=False)}\n\n"
        f"Preferred tafsir source ids: {sources}\n\n"
        "Call the appropriate MCP tools to gather tafsir evidence."
    )

    agent_out = run_researcher_agent(
        role="tafsir",
        system_prompt=TAFSIR_SYSTEM,
        user_message=user_msg,
        response_format=TafsirAgentSummary,
        name="tafsir_researcher",
    )
    warnings.extend(agent_out.warnings)
    tools_used = agent_out.tools_used
    items, w2, mcp_failures = _items_from_tool_calls(agent_out.tool_calls)
    warnings.extend(w2)
    if not items:
        warnings.append("tafsir_researcher: agent found nothing")

    return pack(
        tid,
        warnings=warnings
        + [
            f"tafsir_researcher: fetched={len(items)} "
            f"mcp_failures={mcp_failures} "
            f"sources={sources} tools={tools_used}"
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


def tafsir_researcher_node(state: ResearchState) -> dict:
    from quran_scholar.graph.routing import task_id_for_action
    patched = {**state, "current_task_id": task_id_for_action(state, "tafsir_research")}
    return run_tafsir_research(patched)  # type: ignore[arg-type]
