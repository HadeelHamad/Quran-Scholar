"""Linguistic Researcher — create_agent + MCP linguistic tools."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.helpers import make_evidence, pack, session_fail
from quran_scholar.agents.mcp_agent import (
    has_llm_credentials,
    parse_tool_json,
    run_researcher_agent,
    tool_had_mcp_error,
)
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.mcp.safe import safe_call_tool
from quran_scholar.models import LinguisticEvidence, ResearchPlan, VerseRef
from quran_scholar.state import ResearchState

LING_SYSTEM = """You are the Linguistic Researcher for Quran Scholar.

You have MCP tools: analyze_word, find_root_occurrences, get_root_stats.
Decide which to call for the Arabic question and selected verses.

Guidelines:
- Use analyze_word for important content words (word_no is 1-based in the ayah).
  Prefer nouns/verbs; skip particles.
- After discovering a root, you may call get_root_stats and/or find_root_occurrences.
- NEVER invent morphological analysis — only use tool results.
- Keep calls focused (typically a few words / roots).
"""


class LinguisticAgentSummary(BaseModel):
    notes: str = ""
    roots_studied: list[str] = Field(default_factory=list)


_ROOT_FROM_SARF = re.compile(
    r"مَادَّ[ةه][\u064B-\u0652]*\s*[:：]?\s*\(?\s*([ء-ي]{2,5})\s*\)?",
)
_ROOT_IN_PARENS = re.compile(r"\(([ء-ي]{2,5})\)")


def _extract_root(analysis: dict[str, Any]) -> str | None:
    root = analysis.get("root")
    if isinstance(root, str) and root.strip():
        return root.strip()
    sarf = analysis.get("sarf") or ""
    if isinstance(sarf, str):
        match = _ROOT_FROM_SARF.search(sarf)
        if match:
            return match.group(1)
        parens = _ROOT_IN_PARENS.findall(sarf)
        if parens:
            return parens[-1]
    return None


def _items_from_tool_calls(tool_calls: list) -> tuple[list[LinguisticEvidence], list[str]]:
    items: list[LinguisticEvidence] = []
    warnings: list[str] = []
    for call in tool_calls:
        if call.name not in (
            "analyze_word",
            "get_root_stats",
            "find_root_occurrences",
        ):
            continue
        payload = parse_tool_json(call.content)
        if tool_had_mcp_error(payload):
            warnings.append(
                f"linguistic_researcher: MCP error on {call.name}: "
                f"{payload.get('error') if isinstance(payload, dict) else call.content}"
            )
            continue
        data = mcp_payload(payload) if not isinstance(payload, str) else payload

        if call.name == "analyze_word":
            if not isinstance(data, dict):
                warnings.append("NO_EVIDENCE: analyze_word unusable payload")
                continue
            root = _extract_root(data)
            word = str(data.get("word") or call.args.get("word_no") or "word")
            surah = call.args.get("surah")
            ayah = call.args.get("ayah")
            related: list[VerseRef] = []
            if surah is not None and ayah is not None:
                related = [VerseRef(surah=int(surah), ayah=int(ayah))]
            items.append(
                LinguisticEvidence(
                    query=f"{surah}:{ayah}:{word}" if surah else word,
                    root=root,
                    analysis=json.dumps(data, ensure_ascii=False),
                    related_verses=related,
                    source_tool="analyze_word",
                    raw=data,
                )
            )
        elif call.name == "get_root_stats":
            root = str(call.args.get("root") or "")
            items.append(
                LinguisticEvidence(
                    query=f"root_stats:{root}",
                    root=root or None,
                    analysis=json.dumps({"stats": data}, ensure_ascii=False),
                    source_tool="get_root_stats",
                    raw={"stats": data if isinstance(data, dict) else {"value": data}},
                )
            )
        else:  # find_root_occurrences
            root = str(call.args.get("root") or "")
            occ = as_list(data)
            related = [
                VerseRef(surah=int(h["surah"]), ayah=int(h["ayah"]))
                for h in occ[:20]
                if isinstance(h, dict) and "surah" in h and "ayah" in h
            ]
            items.append(
                LinguisticEvidence(
                    query=f"root_occ:{root}",
                    root=root or None,
                    analysis=json.dumps(
                        {"occurrences_sample": occ[:10]}, ensure_ascii=False
                    ),
                    related_verses=related,
                    source_tool="find_root_occurrences",
                    raw={"occurrences": occ},
                )
            )
    return items, warnings


def _deterministic_linguistic(
    verses: list, question: str
) -> tuple[list[LinguisticEvidence], list[str]]:
    """Simple fallback: analyze first content-like word of up to 3 verses."""
    items: list[LinguisticEvidence] = []
    warnings: list[str] = []
    skip = {"إن", "في", "من", "على", "إلى", "عن", "ما", "لا", "أن", "يا", "و", "ف"}
    with ScopedTafsirMCPClient("linguistic") as client:
        for verse in verses[:3]:
            words = [w for w in (verse.text_uthmani or "").split() if w.strip()]
            idxs = [
                i + 1
                for i, w in enumerate(words)
                if len(re.sub(r"[^\u0600-\u06FF]", "", w)) >= 3
                and re.sub(r"[^\u0600-\u06FF/]", "", w) not in skip
            ][:2]
            for word_no in idxs:
                outcome = safe_call_tool(
                    client,
                    "analyze_word",
                    {
                        "surah": verse.ref.surah,
                        "ayah": verse.ref.ayah,
                        "word_no": word_no,
                        "aspects": ["meaning", "sarf", "root", "irab"],
                    },
                    label=f"analyze_word {verse.ref.surah}:{verse.ref.ayah}#{word_no}",
                )
                warnings.extend(outcome.warnings)
                if outcome.failed:
                    continue
                payload = mcp_payload(outcome.data)
                if not isinstance(payload, dict):
                    continue
                root = _extract_root(payload)
                items.append(
                    LinguisticEvidence(
                        query=f"{verse.ref.surah}:{verse.ref.ayah}#{word_no}",
                        root=root,
                        analysis=json.dumps(payload, ensure_ascii=False),
                        related_verses=[verse.ref],
                        source_tool="analyze_word",
                        raw=payload,
                    )
                )
                if root:
                    for tool, args in (
                        ("get_root_stats", {"root": root}),
                        ("find_root_occurrences", {"root": root, "limit": 20}),
                    ):
                        out = safe_call_tool(client, tool, args, label=f"{tool} {root}")
                        warnings.extend(out.warnings)
    return items, warnings


def run_linguistic_research(state: ResearchState) -> dict:
    tid = state.get("current_task_id") or ""
    plan: ResearchPlan | None = state.get("research_plan")
    kind = ""
    if plan and tid:
        for t in plan.tasks:
            if t.id == tid:
                kind = t.kind
                break
    if kind in ("linguistic", "linguistic_analysis"):
        reason = "explicit linguistic task"
    elif plan and plan.needs_linguistic_analysis:
        reason = "plan.needs_linguistic_analysis"
    else:
        reason = "linguistic analysis not required for this question"
        return pack(
            tid,
            warnings=[f"linguistic_researcher: skipped ({reason})"],
        )

    verses = list(state.get("selected_verses") or [])
    if not verses:
        return pack(
            tid,
            warnings=["linguistic_researcher: no selected_verses"],
        )

    question = state.get("user_question") or ""
    verse_brief = [
        {
            "surah": v.ref.surah,
            "ayah": v.ref.ayah,
            "text": v.text_uthmani,
            "words": [
                {"word_no": i + 1, "word": w}
                for i, w in enumerate((v.text_uthmani or "").split())
            ],
        }
        for v in verses[:5]
    ]
    user_msg = (
        f"User question:\n{question}\n\n"
        f"Selected verses (with word indices):\n"
        f"{json.dumps(verse_brief, ensure_ascii=False)}\n\n"
        "Call linguistic MCP tools as needed."
    )

    items: list[LinguisticEvidence] = []
    warnings: list[str] = []
    tools_used: list[str] = []

    try:
        if has_llm_credentials():
            agent_out = run_researcher_agent(
                role="linguistic",
                system_prompt=LING_SYSTEM,
                user_message=user_msg,
                response_format=LinguisticAgentSummary,
                name="linguistic_researcher",
            )
            warnings.extend(agent_out.warnings)
            tools_used = agent_out.tools_used
            items, w2 = _items_from_tool_calls(agent_out.tool_calls)
            warnings.extend(w2)
            if not items:
                warnings.append(
                    "linguistic_researcher: agent empty — deterministic fallback"
                )
                items, w3 = _deterministic_linguistic(verses, question)
                warnings.extend(w3)
        else:
            items, w = _deterministic_linguistic(verses, question)
            warnings.extend(w)
            tools_used = ["analyze_word"]
    except MCPError as exc:
        return session_fail(
            "linguistic_researcher", "Linguistic retrieval", exc, tid)

    return pack(
        tid,
        warnings=warnings
        + [f"linguistic_researcher: items={len(items)} tools={tools_used}"],
        linguistic_evidence=items,
        evidence_items=[
            make_evidence(
                kind="linguistic",
                content=x.analysis,
                refs=x.related_verses,
                id_prefix="ling",
                metadata={
                    "source_tool": x.source_tool,
                    "root": x.root,
                    "raw": x.raw,
                },
            )
            for x in items
        ],
    )
