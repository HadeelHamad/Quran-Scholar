"""Quran Researcher — search concepts → MCP search → evaluate → select."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.llm import get_llm
from quran_scholar.config import quran_search_limit
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.models import Evidence, ResearchPlan, VerseEvidence, VerseRef
from quran_scholar.state import ResearchState

SEARCH_CONCEPTS_SYSTEM = """You generate short Quran search queries for Tafsir MCP FTS5.
Return 1–4 concise ARABIC keyword stems (no English; avoid leading ال when possible).
Examples: صبر، يتيم، تقوى، رحمة — not full sentences.
Do NOT answer the question. Do NOT invent verse numbers."""


class SearchConcepts(BaseModel):
    concepts: list[str] = Field(
        description="1–4 Arabic keyword stems for Quran FTS (not English)"
    )


# Heuristic English/Arabic theme → Arabic FTS stems
_THEME_STEMS: dict[str, list[str]] = {
    "patience": ["صبر"],
    "sabr": ["صبر"],
    "صبر": ["صبر"],
    "orphan": ["يتيم", "يتامى"],
    "يتيم": ["يتيم"],
    "mercy": ["رحمة", "رحم"],
    "رحمة": ["رحمة"],
    "prayer": ["صلاة", "صلو"],
    "صلاة": ["صلاة"],
    "tawhid": ["الله", "اله"],
    "forgive": ["غفر", "مغفرة"],
}


class VerseSelection(BaseModel):
    """LLM relevance judgment — candidates are NOT auto-selected."""

    selected_indices: list[int] = Field(
        description="0-based indices into discovered candidate list that are relevant"
    )
    relevance_notes: list[str] = Field(
        default_factory=list,
        description="Short note per selected index explaining relevance",
    )


EVALUATE_SYSTEM = """You evaluate Quran search candidates for relevance to the user question.
You receive a numbered list of candidate verses (search hits).
Select ONLY verses that materially help answer the question.
Do not select all hits by default. Prefer precision over recall (typically 1–8 verses).
Return selected_indices (0-based). Do NOT invent verses not in the list."""


def _parse_verse_ref(text: str) -> VerseRef | None:
    match = re.search(r"\b(\d{1,3})\s*:\s*(\d{1,3})\b", text)
    if not match:
        return None
    surah, ayah = int(match.group(1)), int(match.group(2))
    if 1 <= surah <= 114 and ayah >= 1:
        return VerseRef(surah=surah, ayah=ayah)
    return None


def _heuristic_concepts(question: str) -> list[str]:
    q = question.lower()
    stems: list[str] = []
    for key, values in _THEME_STEMS.items():
        if key in q or key in question:
            stems.extend(values)
    # Any Arabic tokens in the question (length ≥ 2)
    arabic_tokens = re.findall(r"[\u0600-\u06FF]{2,}", question)
    stems.extend(arabic_tokens)
    # Deduplicate preserve order
    seen: set[str] = set()
    out: list[str] = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:4] or ["صبر"]


def _generate_search_concepts(question: str, language: str) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return _heuristic_concepts(question)

    llm = get_llm()
    structured = llm.with_structured_output(SearchConcepts)
    try:
        out = structured.invoke(
            [
                {"role": "system", "content": SEARCH_CONCEPTS_SYSTEM},
                {
                    "role": "user",
                    "content": f"Language: {language}\nQuestion: {question}",
                },
            ]
        )
        if not isinstance(out, SearchConcepts):
            out = SearchConcepts.model_validate(out)
        concepts = [c.strip() for c in out.concepts if c and c.strip()]
        # Prefer Arabic-only concepts for FTS
        arabic = [c for c in concepts if re.search(r"[\u0600-\u06FF]", c)]
        return (arabic or concepts)[:4] or _heuristic_concepts(question)
    except Exception:
        return _heuristic_concepts(question)


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


def _evaluate_selection(
    question: str,
    discovered: list[VerseEvidence],
) -> list[VerseEvidence]:
    if not discovered:
        return []
    if len(discovered) == 1:
        v = discovered[0].model_copy(update={"relevance": "sole candidate"})
        return [v]

    api_key = os.getenv("OPENAI_API_KEY", "")
    catalog = []
    for i, v in enumerate(discovered):
        catalog.append(
            {
                "index": i,
                "ref": f"{v.ref.surah}:{v.ref.ayah}",
                "text": v.text_uthmani[:280],
            }
        )

    if not api_key or api_key.startswith("your_"):
        # Heuristic: take top 5 search hits (still separate from full discovered set)
        selected = discovered[:5]
        return [
            s.model_copy(update={"relevance": "heuristic top hit"})
            for s in selected
        ]

    llm = get_llm()
    structured = llm.with_structured_output(VerseSelection)
    try:
        out = structured.invoke(
            [
                {"role": "system", "content": EVALUATE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nCandidates:\n"
                        + json.dumps(catalog, ensure_ascii=False)
                    ),
                },
            ]
        )
        if not isinstance(out, VerseSelection):
            out = VerseSelection.model_validate(out)
        selected: list[VerseEvidence] = []
        notes = out.relevance_notes or []
        for idx in out.selected_indices:
            if 0 <= idx < len(discovered):
                note = notes[len(selected)] if len(selected) < len(notes) else "relevant"
                selected.append(
                    discovered[idx].model_copy(update={"relevance": note})
                )
        return selected or [
            discovered[0].model_copy(update={"relevance": "fallback first hit"})
        ]
    except Exception:
        return [
            s.model_copy(update={"relevance": "fallback top hit"})
            for s in discovered[:5]
        ]


def _verse_to_evidence(v: VerseEvidence) -> Evidence:
    from quran_scholar.services.citation_manager import citation_manager

    ev = Evidence(
        id=f"verse-{v.ref.surah}-{v.ref.ayah}-{uuid.uuid4().hex[:8]}",
        kind="verse",
        content=v.text_uthmani,
        citation="",  # filled by citation_manager
        refs=[v.ref],
        metadata={
            "source_tool": v.source_tool,
            "relevance": v.relevance,
            "raw": v.raw,
        },
    )
    return ev.model_copy(update={"citation": citation_manager.format(ev).label})


def run_quran_research(state: ResearchState) -> dict:
    """
    Find relevant verses.

    discovered_verses = all search/fetch hits
    selected_verses   = relevance-filtered subset only
    evidence_items    = only from selected_verses (raw text, not LLM paraphrase)
    """
    question = state.get("user_question") or ""
    language = state.get("language") or "ar"
    plan: ResearchPlan | None = state.get("research_plan")
    task_id = state.get("current_task_id") or ""
    warnings: list[str] = []
    errors: list[str] = []

    discovered: list[VerseEvidence] = []
    selected: list[VerseEvidence] = []

    try:
        with ScopedTafsirMCPClient("quran") as client:
            # Verse-specific: fetch primary ayah first
            primary = plan.primary_verse if plan else None
            if primary is None:
                primary = _parse_verse_ref(question)

            task_kind = ""
            if plan and task_id:
                for t in plan.tasks:
                    if t.id == task_id:
                        task_kind = t.kind
                        break

            if primary and task_kind in ("", "fetch_ayah", "verse_search"):
                raw = client.call_tool(
                    "fetch_ayah",
                    {"surah": primary.surah, "ayah": primary.ayah},
                )
                payload = mcp_payload(raw)
                if isinstance(payload, dict):
                    ve = _record_to_verse(payload, "fetch_ayah")
                    if ve:
                        discovered.append(ve)
                        selected.append(
                            ve.model_copy(
                                update={"relevance": "primary verse from question/plan"}
                            )
                        )

            # Thematic / additional search
            if task_kind in ("", "quran_search", "verse_search") or not selected:
                concepts = _generate_search_concepts(question, language)
                limit = quran_search_limit()
                for concept in concepts:
                    raw = client.call_tool(
                        "search_quran_text",
                        {"query": concept, "limit": limit},
                    )
                    for rec in as_list(raw):
                        ve = _record_to_verse(rec, "search_quran_text")
                        if ve:
                            discovered.append(ve)

            discovered = _dedupe_verses(discovered)

            # Evaluate: search hits are NOT automatically evidence
            if not selected:
                selected = _evaluate_selection(question, discovered)
            else:
                # Still evaluate extra search hits for thematic expansion
                extras = [
                    d
                    for d in discovered
                    if (d.ref.surah, d.ref.ayah)
                    not in {(s.ref.surah, s.ref.ayah) for s in selected}
                ]
                if extras and task_kind == "quran_search":
                    more = _evaluate_selection(question, extras)
                    selected = _dedupe_verses(selected + more)

    except Exception as exc:
        errors.append(f"quran_researcher: MCP error: {exc}")
        return {
            "errors": errors,
            "warnings": warnings,
            "completed_task_ids": [task_id] if task_id else [],
        }

    evidence = [_verse_to_evidence(v) for v in selected]
    updates: dict[str, Any] = {
        "discovered_verses": discovered,
        "selected_verses": selected,
        "evidence_items": evidence,
        "warnings": [
            f"quran_researcher: discovered={len(discovered)} selected={len(selected)}"
        ],
    }
    if task_id:
        updates["completed_task_ids"] = [task_id]
    if errors:
        updates["errors"] = errors
    return updates
