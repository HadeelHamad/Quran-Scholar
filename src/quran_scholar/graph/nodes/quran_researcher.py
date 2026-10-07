"""Quran Researcher — create_agent + MCP tools for text, search, surah/meta info."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.graph.nodes.helpers import make_evidence, pack, session_fail
from quran_scholar.graph.nodes.mcp_agent import (
    has_llm_credentials,
    parse_tool_json,
    run_researcher_agent,
    tool_had_mcp_error,
)
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import ResearchPlan, VerseEvidence, VerseRef
from quran_scholar.graph.nodes.planner import normalize_question_text, parse_verse_ref
from quran_scholar.state import ResearchState

QURAN_SYSTEM = """You are the Quran Researcher for Quran Scholar.

You answer Quran-related needs using MCP tools (not only verse search):
fetch_ayah, search_quran_text, fetch_surah_info, get_quran_overview,
get_surah_statistics, get_qeraat_variants, get_page_fawaed.

Decide which tool(s) fit the Arabic user question.

Guidelines:
- Known surah:ayah → fetch_ayah (and get_qeraat_variants if qira'at asked).
- Thematic / wording search → search_quran_text with 1–4 concise ARABIC stems.
- Surah metadata / makki-madani / verse counts → fetch_surah_info / get_surah_statistics.
- Broad corpus questions → get_quran_overview.
- Page benefits → get_page_fawaed when a page is relevant.
- Prefer precision; NEVER invent verse text or numbers — only tool results.
- In structured output, list selected_refs (surah:ayah) when verses matter;
  leave selected_refs empty for pure meta/stats answers and summarize in notes.
"""


class QuranAgentResult(BaseModel):
    selected_refs: list[str] = Field(
        default_factory=list,
        description="surah:ayah strings to keep as selected verses",
    )
    relevance_notes: list[str] = Field(default_factory=list)
    notes: str = ""


_THEME_STEMS: dict[str, list[str]] = {
    "صبر": ["صبر"],
    "يتيم": ["يتيم", "يتامى"],
    "رحمة": ["رحمة", "رحم"],
    "صلاة": ["صلاة"],
    "تقوى": ["تقوى"],
    "كرسي": ["كرسي"],
}


def _heuristic_concepts(question: str) -> list[str]:
    q = normalize_question_text(question)
    stems: list[str] = []
    for key, values in _THEME_STEMS.items():
        if key in q:
            stems.extend(values)
    stems.extend(re.findall(r"[\u0600-\u06FF]{2,}", q))
    seen: set[str] = set()
    out: list[str] = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:4] or ["صبر"]


def _record_to_verse(rec: dict[str, Any], source_tool: str) -> VerseEvidence | None:
    surah = rec.get("surah")
    ayah = rec.get("ayah")
    if surah is None or ayah is None:
        return None
    text = (
        rec.get("text_uthmani")
        or rec.get("text")
        or rec.get("text_simple")
        or rec.get("snippet")
        or ""
    )
    return VerseEvidence(
        ref=VerseRef(surah=int(surah), ayah=int(ayah)),
        text_uthmani=str(text),
        source_tool=source_tool,
        raw=rec,
    )


def _dedupe_verses(verses: list[VerseEvidence]) -> list[VerseEvidence]:
    seen: set[tuple[int, int]] = set()
    out: list[VerseEvidence] = []
    for v in verses:
        key = (v.ref.surah, v.ref.ayah)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _parse_ref_str(ref: str) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d{1,3})\s*:\s*(\d{1,3})\s*$", ref or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


_META_TOOLS = frozenset(
    {
        "fetch_surah_info",
        "get_quran_overview",
        "get_surah_statistics",
        "get_qeraat_variants",
        "get_page_fawaed",
    }
)


def _verses_from_tool_calls(tool_calls: list) -> tuple[list[VerseEvidence], list[str], int]:
    discovered: list[VerseEvidence] = []
    warnings: list[str] = []
    mcp_failures = 0
    for call in tool_calls:
        if call.name not in ("fetch_ayah", "search_quran_text"):
            continue
        payload = parse_tool_json(call.content)
        if tool_had_mcp_error(payload):
            mcp_failures += 1
            warnings.append(
                f"quran_researcher: MCP error on {call.name}: "
                f"{payload.get('error') if isinstance(payload, dict) else call.content}"
            )
            continue
        data = mcp_payload(payload) if not isinstance(payload, str) else payload
        if call.name == "fetch_ayah":
            rec = data if isinstance(data, dict) else None
            if rec:
                ve = _record_to_verse(rec, "fetch_ayah")
                if ve:
                    discovered.append(ve)
                else:
                    warnings.append("NO_EVIDENCE: fetch_ayah returned unusable ayah")
            else:
                warnings.append("NO_EVIDENCE: fetch_ayah returned no ayah payload")
        else:
            hits = as_list(data)
            if not hits:
                warnings.append(
                    f"NO_EVIDENCE: search_quran_text zero hits args={call.args}"
                )
            for rec in hits:
                if isinstance(rec, dict):
                    ve = _record_to_verse(rec, "search_quran_text")
                    if ve:
                        discovered.append(ve)
    return _dedupe_verses(discovered), warnings, mcp_failures


def _meta_from_tool_calls(tool_calls: list) -> tuple[list, list[str], int]:
    """Turn surah/overview/qira'at/stats tool results into evidence items."""
    items: list = []
    warnings: list[str] = []
    mcp_failures = 0
    for call in tool_calls:
        if call.name not in _META_TOOLS:
            continue
        payload = parse_tool_json(call.content)
        if tool_had_mcp_error(payload):
            mcp_failures += 1
            warnings.append(
                f"quran_researcher: MCP error on {call.name}: "
                f"{payload.get('error') if isinstance(payload, dict) else call.content}"
            )
            continue
        data = mcp_payload(payload) if not isinstance(payload, str) else payload
        if data is None or data == "" or data == [] or data == {}:
            warnings.append(f"NO_EVIDENCE: {call.name} returned empty payload")
            continue
        text = (
            data
            if isinstance(data, str)
            else json.dumps(data, ensure_ascii=False)[:6000]
        )
        refs: list[VerseRef] = []
        if isinstance(call.args.get("surah"), int) or str(
            call.args.get("surah") or ""
        ).isdigit():
            try:
                refs = [VerseRef(surah=int(call.args["surah"]), ayah=1)]
            except (KeyError, TypeError, ValueError):
                refs = []
        items.append(
            make_evidence(
                kind="quran_meta",
                content=text,
                refs=refs,
                id_prefix=f"meta-{call.name}",
                metadata={"source_tool": call.name, "args": call.args, "raw": data},
            )
        )
    return items, warnings, mcp_failures


def _select_from_agent(
    discovered: list[VerseEvidence],
    structured: QuranAgentResult | None,
) -> list[VerseEvidence]:
    if not discovered:
        return []
    if structured and structured.selected_refs:
        by_key = {(v.ref.surah, v.ref.ayah): v for v in discovered}
        selected: list[VerseEvidence] = []
        notes = structured.relevance_notes or []
        for i, ref_s in enumerate(structured.selected_refs):
            key = _parse_ref_str(ref_s)
            if key and key in by_key:
                note = notes[i] if i < len(notes) else "agent-selected"
                selected.append(by_key[key].model_copy(update={"relevance": note}))
        if selected:
            return selected
    # Prefer fetch_ayah hits, else top search hits
    fetched = [v for v in discovered if v.source_tool == "fetch_ayah"]
    if fetched:
        return [
            v.model_copy(update={"relevance": "fetched primary"}) for v in fetched
        ]
    return [
        v.model_copy(update={"relevance": "top search hit"})
        for v in discovered[:8]
    ]


def _deterministic_quran(
    question: str,
    plan: ResearchPlan | None,
    tid: str,
) -> tuple[list[VerseEvidence], list[VerseEvidence], list[str], int]:
    warnings: list[str] = []
    discovered: list[VerseEvidence] = []
    selected: list[VerseEvidence] = []
    mcp_failures = 0
    primary = plan.primary_verse if plan else None
    if primary is None:
        primary = parse_verse_ref(question)
    task_kind = ""
    if plan and tid:
        for t in plan.tasks:
            if t.id == tid:
                task_kind = t.kind
                break
    with ScopedTafsirMCPClient("quran") as client:
        if primary and task_kind in ("", "fetch_ayah", "verse_search"):
            outcome = safe_call_tool(
                client,
                "fetch_ayah",
                {"surah": primary.surah, "ayah": primary.ayah},
                label=f"fetch_ayah {primary.surah}:{primary.ayah}",
            )
            warnings.extend(outcome.warnings)
            if outcome.failed:
                mcp_failures += 1
            else:
                payload = mcp_payload(outcome.data)
                ve = (
                    _record_to_verse(payload, "fetch_ayah")
                    if isinstance(payload, dict)
                    else None
                )
                if ve:
                    discovered.append(ve)
                    selected.append(
                        ve.model_copy(
                            update={"relevance": "primary verse from question/plan"}
                        )
                    )
                else:
                    warnings.extend(
                        mark_empty(
                            outcome,
                            f"NO_EVIDENCE: no ayah for {primary.surah}:{primary.ayah}",
                        )
                    )
        if task_kind in ("", "quran_search", "verse_search") or not selected:
            try:
                limit = max(1, min(50, int(os.getenv("MAX_QURAN_SEARCH_VERSES", "15"))))
            except ValueError:
                limit = 15
            for concept in _heuristic_concepts(question):
                outcome = safe_call_tool(
                    client,
                    "search_quran_text",
                    {"query": concept, "limit": limit},
                    label=f"search_quran_text '{concept}'",
                )
                warnings.extend(outcome.warnings)
                if outcome.failed:
                    mcp_failures += 1
                    continue
                hits = as_list(outcome.data)
                if not hits:
                    warnings.extend(
                        mark_empty(
                            outcome,
                            f"NO_EVIDENCE: zero hits for '{concept}'",
                        )
                    )
                    continue
                for rec in hits:
                    ve = _record_to_verse(rec, "search_quran_text")
                    if ve:
                        discovered.append(ve)
    discovered = _dedupe_verses(discovered)
    if not selected:
        selected = [
            v.model_copy(update={"relevance": "heuristic top hit"})
            for v in discovered[:5]
        ]
    return discovered, selected, warnings, mcp_failures


def run_quran_research(state: ResearchState) -> dict:
    question = state.get("user_question") or ""
    plan: ResearchPlan | None = state.get("research_plan")
    tid = state.get("current_task_id") or ""
    warnings: list[str] = []
    primary = plan.primary_verse if plan else None
    if primary is None:
        primary = parse_verse_ref(question)
    try:
        limit = max(1, min(50, int(os.getenv("MAX_QURAN_SEARCH_VERSES", "15"))))
    except ValueError:
        limit = 15
    brief = {
        "question": question,
        "primary_verse": (
            f"{primary.surah}:{primary.ayah}" if primary else None
        ),
        "search_limit": limit,
    }
    user_msg = (
        f"Research brief:\n{json.dumps(brief, ensure_ascii=False)}\n\n"
        "Call the MCP tools needed for this Quran-related question "
        "(verses and/or surah/overview/qira'at/stats). "
        "Return selected_refs when verses are part of the answer."
    )

    discovered: list[VerseEvidence] = []
    selected: list[VerseEvidence] = []
    meta_evidence: list = []
    mcp_failures = 0
    tools_used: list[str] = []

    try:
        if has_llm_credentials():
            agent_out = run_researcher_agent(
                role="quran",
                system_prompt=QURAN_SYSTEM,
                user_message=user_msg,
                response_format=QuranAgentResult,
                name="quran_researcher",
            )
            warnings.extend(agent_out.warnings)
            tools_used = agent_out.tools_used
            discovered, w2, fail_v = _verses_from_tool_calls(agent_out.tool_calls)
            meta_evidence, w_meta, fail_m = _meta_from_tool_calls(agent_out.tool_calls)
            mcp_failures = fail_v + fail_m
            warnings.extend(w2)
            warnings.extend(w_meta)
            structured = (
                agent_out.structured
                if isinstance(agent_out.structured, QuranAgentResult)
                else None
            )
            selected = _select_from_agent(discovered, structured)
            if not discovered and not meta_evidence:
                warnings.append(
                    "quran_researcher: agent found nothing — deterministic fallback"
                )
                discovered, selected, w3, mcp_failures = _deterministic_quran(
                    question, plan, tid
                )
                warnings.extend(w3)
        else:
            discovered, selected, w, mcp_failures = _deterministic_quran(
                question, plan, tid
            )
            warnings.extend(w)
            tools_used = ["fetch_ayah", "search_quran_text"]
    except MCPError as exc:
        return session_fail("quran_researcher", "Quran retrieval", exc, tid)

    verse_evidence = [
        make_evidence(
            kind="verse",
            content=v.text_uthmani,
            refs=[v.ref],
            id_prefix=f"verse-{v.ref.surah}-{v.ref.ayah}",
            metadata={
                "source_tool": v.source_tool,
                "relevance": v.relevance,
                "raw": v.raw,
            },
        )
        for v in selected
    ]

    return pack(
        tid,
        warnings=warnings
        + [
            f"quran_researcher: discovered={len(discovered)} "
            f"selected={len(selected)} meta={len(meta_evidence)} "
            f"mcp_failures={mcp_failures} tools={tools_used}"
        ],
        discovered_verses=discovered,
        selected_verses=selected,
        evidence_items=verse_evidence + meta_evidence,
    )


def quran_researcher_node(state: ResearchState) -> dict:
    from quran_scholar.graph.routing import task_id_for_action
    patched = {**state, "current_task_id": task_id_for_action(state, "quran_research")}
    return run_quran_research(patched)  # type: ignore[arg-type]
