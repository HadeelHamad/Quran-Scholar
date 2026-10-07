"""Linguistic Researcher — optional root/word study via Tafsir MCP."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.llm import get_llm
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import (
    Evidence,
    LinguisticEvidence,
    ResearchPlan,
    VerseEvidence,
    VerseRef,
)
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines

TERM_PICKER_SYSTEM = """You pick important Quranic content words for linguistic study.
Given a verse and the user question, return up to 3 word_no values (1-based positions
in the whitespace-split ayah text). Prefer thematic content words (nouns/verbs),
not particles (إن، في، من، على، و، ف، ال…).
Only choose terms that help answer the question. If none warranted, return []."""


class ImportantTerms(BaseModel):
    word_numbers: list[int] = Field(
        default_factory=list,
        description="1-based word positions in the ayah (max 3)",
    )
    roots_hint: list[str] = Field(
        default_factory=list,
        description="Optional Arabic roots to study if known (e.g. صبر)",
    )


_ROOT_FROM_SARF = re.compile(
    r"مَادَّ[ةه][\u064B-\u0652]*\s*[:：]?\s*\(?\s*([ء-ي]{2,5})\s*\)?",
)
_ROOT_IN_PARENS = re.compile(r"\(([ء-ي]{2,5})\)")


def _should_run_linguistic(state: ResearchState) -> tuple[bool, str]:
    """Do not auto-run deep linguistics for every theological question."""
    plan: ResearchPlan | None = state.get("research_plan")
    task_id = state.get("current_task_id") or ""
    task_kind = ""
    if plan and task_id:
        for t in plan.tasks:
            if t.id == task_id:
                task_kind = t.kind
                break

    if task_kind in ("linguistic", "linguistic_analysis"):
        return True, "explicit linguistic task"

    if plan and plan.needs_linguistic_analysis:
        return True, "plan.needs_linguistic_analysis"

    return False, "linguistic analysis not required for this question"


def _tokenize_words(text: str) -> list[str]:
    return [w for w in (text or "").split() if w.strip()]


def _pick_terms(
    question: str,
    verse: VerseEvidence,
) -> ImportantTerms:
    words = _tokenize_words(verse.text_uthmani)
    if not words:
        return ImportantTerms()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        # Heuristic: middle content words, skip short particles
        skip = {"إن", "في", "من", "على", "إلى", "عن", "ما", "لا", "أن", "يا", "و", "ف"}
        idxs = [
            i + 1
            for i, w in enumerate(words)
            if len(re.sub(r"[^\u0600-\u06FF]", "", w)) >= 3
            and re.sub(r"[^\u0600-\u06FF/]", "", w) not in skip
        ]
        return ImportantTerms(word_numbers=idxs[:3])

    llm = get_llm()
    structured = llm.with_structured_output(ImportantTerms)
    try:
        catalog = [{"word_no": i + 1, "word": w} for i, w in enumerate(words)]
        out = structured.invoke(
            [
                {"role": "system", "content": TERM_PICKER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Verse {verse.ref.surah}:{verse.ref.ayah}\n"
                        f"Words: {json.dumps(catalog, ensure_ascii=False)}"
                    ),
                },
            ]
        )
        if not isinstance(out, ImportantTerms):
            out = ImportantTerms.model_validate(out)
        out.word_numbers = [n for n in out.word_numbers if 1 <= n <= len(words)][:3]
        return out
    except Exception:
        return ImportantTerms(word_numbers=[min(5, len(words))])


def _extract_root(analysis: dict[str, Any]) -> str | None:
    root = analysis.get("root")
    if isinstance(root, str) and root.strip():
        return root.strip()
    sarf = analysis.get("sarf") or ""
    if isinstance(sarf, str):
        match = _ROOT_FROM_SARF.search(sarf)
        if match:
            return match.group(1)
        # Fallback: last parenthesized Arabic token in sarf (often the root)
        parens = _ROOT_IN_PARENS.findall(sarf)
        if parens:
            return parens[-1]
    word = str(analysis.get("word") or "")
    # Light heuristic for common stems embedded in the surface form
    for stem in ("صبر", "رحم", "أمن", "صلى", "غفر"):
        if stem in word:
            return stem
    return None


def _ling_to_evidence(item: LinguisticEvidence) -> Evidence:
    from quran_scholar.services.citation_manager import citation_manager

    ev = Evidence(
        id=f"ling-{uuid.uuid4().hex[:10]}",
        kind="linguistic",
        content=item.analysis,
        citation="",
        refs=item.related_verses,
        metadata={
            "source_tool": item.source_tool,
            "root": item.root,
            "raw": item.raw,
        },
    )
    return ev.model_copy(update={"citation": citation_manager.format(ev).label})


def run_linguistic_research(state: ResearchState) -> dict:
    """
    Investigate terminology when relevant.

    Flow: selected verses → important terms → analyze_word → roots →
    find_root_occurrences → get_root_stats.
    Skips when the plan does not call for linguistic work.
    """
    task_id = state.get("current_task_id") or ""
    lines: list[str] = [
        trace("linguistic_researcher", "Analyzing roots...", blank_before=True)
    ]
    should, reason = _should_run_linguistic(state)
    if not should:
        lines.append(
            trace("linguistic_researcher", f"Skipped ({reason}).")
        )
        return {
            "warnings": [f"linguistic_researcher: skipped ({reason})"],
            "completed_task_ids": [task_id] if task_id else [],
            **trace_lines(*lines),
        }

    verses = list(state.get("selected_verses") or [])
    if not verses:
        lines.append(
            trace("linguistic_researcher", "Skipped — no selected verses.")
        )
        return {
            "warnings": ["linguistic_researcher: no selected_verses"],
            "completed_task_ids": [task_id] if task_id else [],
            **trace_lines(*lines),
        }

    question = state.get("user_question") or ""
    linguistic_items: list[LinguisticEvidence] = []
    warnings: list[str] = []
    errors: list[str] = []
    roots_seen: set[str] = set()

    try:
        with ScopedTafsirMCPClient("linguistic") as client:
            for verse in verses[:5]:
                picks = _pick_terms(question, verse)
                for word_no in picks.word_numbers:
                    outcome = safe_call_tool(
                        client,
                        "analyze_word",
                        {
                            "surah": verse.ref.surah,
                            "ayah": verse.ref.ayah,
                            "word_no": word_no,
                            "aspects": ["meaning", "sarf", "root", "irab"],
                        },
                        label=(
                            f"analyze_word {verse.ref.surah}:"
                            f"{verse.ref.ayah}#{word_no}"
                        ),
                    )
                    warnings.extend(outcome.warnings)
                    if outcome.failed:
                        continue
                    payload = mcp_payload(outcome.data)
                    if not isinstance(payload, dict):
                        warnings.extend(
                            mark_empty(
                                outcome,
                                "NO_EVIDENCE: analyze_word succeeded but payload "
                                f"unusable for {verse.ref.surah}:"
                                f"{verse.ref.ayah}#{word_no}",
                            )
                        )
                        continue
                    root = _extract_root(payload)
                    word = str(payload.get("word") or f"word_no={word_no}")
                    linguistic_items.append(
                        LinguisticEvidence(
                            query=(
                                f"{verse.ref.surah}:{verse.ref.ayah}"
                                f"#{word_no}:{word}"
                            ),
                            root=root,
                            analysis=json.dumps(payload, ensure_ascii=False),
                            related_verses=[verse.ref],
                            source_tool="analyze_word",
                            raw=payload,
                        )
                    )
                    if root:
                        roots_seen.add(root)

                for hint in picks.roots_hint:
                    if hint and hint.strip():
                        roots_seen.add(hint.strip())

            for root in sorted(roots_seen)[:5]:
                stats_out = safe_call_tool(
                    client,
                    "get_root_stats",
                    {"root": root},
                    label=f"get_root_stats {root}",
                )
                warnings.extend(stats_out.warnings)
                occ_out = safe_call_tool(
                    client,
                    "find_root_occurrences",
                    {"root": root, "limit": 20},
                    label=f"find_root_occurrences {root}",
                )
                warnings.extend(occ_out.warnings)
                if stats_out.failed and occ_out.failed:
                    continue
                stats = mcp_payload(stats_out.data) if stats_out.ok else None
                occ = as_list(occ_out.data) if occ_out.ok else []
                related = [
                    VerseRef(surah=int(h["surah"]), ayah=int(h["ayah"]))
                    for h in occ[:20]
                    if "surah" in h and "ayah" in h
                ]
                linguistic_items.append(
                    LinguisticEvidence(
                        query=f"root:{root}",
                        root=root,
                        analysis=json.dumps(
                            {
                                "stats": stats,
                                "occurrences_sample": occ[:10],
                                "stats_mcp_failed": stats_out.failed,
                                "occ_mcp_failed": occ_out.failed,
                            },
                            ensure_ascii=False,
                        ),
                        related_verses=related[:20],
                        source_tool="get_root_stats+find_root_occurrences",
                        raw={
                            "stats": (
                                stats if isinstance(stats, dict) else {"value": stats}
                            ),
                            "occurrences": occ,
                        },
                    )
                )
    except MCPError as exc:
        msg = f"Linguistic retrieval failed (MCP session — not 'no analysis'): {exc}"
        lines.append(trace("linguistic_researcher", f"MCP session failed: {exc}"))
        return {
            "warnings": [msg],
            "errors": [msg],
            "completed_task_ids": [task_id] if task_id else [],
            **trace_lines(*lines),
        }

    evidence = [_ling_to_evidence(x) for x in linguistic_items]
    root_list = sorted(r for r in roots_seen if r)
    lines.append(
        trace(
            "linguistic_researcher",
            f"Analyzed {len(linguistic_items)} item(s)"
            + (f"; roots: {', '.join(root_list)}." if root_list else "."),
        )
    )
    updates: dict[str, Any] = {
        "linguistic_evidence": linguistic_items,
        "evidence_items": evidence,
        "warnings": warnings
        + [
            f"linguistic_researcher: ran ({reason}); "
            f"items={len(linguistic_items)} roots={sorted(roots_seen)}"
        ],
        **trace_lines(*lines),
    }
    if task_id:
        updates["completed_task_ids"] = [task_id]
    if errors:
        updates["errors"] = errors
    return updates
