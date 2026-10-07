"""Linguistic Researcher — optional root/word study via Tafsir MCP."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.helpers import (
    make_evidence,
    pack,
    session_fail,
)
from quran_scholar.agents.llm import get_llm
from quran_scholar.mcp.client import ScopedTafsirMCPClient
from quran_scholar.mcp.errors import MCPError
from quran_scholar.mcp.parse import as_list, mcp_payload
from quran_scholar.mcp.safe import mark_empty, safe_call_tool
from quran_scholar.models import LinguisticEvidence, ResearchPlan, VerseEvidence, VerseRef
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace

TERM_PICKER_SYSTEM = """You pick important Quranic content words for linguistic study.
Given a verse and the user question, return up to 3 word_no values (1-based positions
in the whitespace-split ayah text). Prefer thematic content words (nouns/verbs),
not particles (إن، في، من، على، و، ف، ال…).
Only choose terms that help answer the question. If none warranted, return []."""


class ImportantTerms(BaseModel):
    word_numbers: list[int] = Field(default_factory=list)
    roots_hint: list[str] = Field(default_factory=list)


_ROOT_FROM_SARF = re.compile(
    r"مَادَّ[ةه][\u064B-\u0652]*\s*[:：]?\s*\(?\s*([ء-ي]{2,5})\s*\)?",
)
_ROOT_IN_PARENS = re.compile(r"\(([ء-ي]{2,5})\)")


def _pick_terms(question: str, verse: VerseEvidence) -> ImportantTerms:
    words = [w for w in (verse.text_uthmani or "").split() if w.strip()]
    if not words:
        return ImportantTerms()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        skip = {"إن", "في", "من", "على", "إلى", "عن", "ما", "لا", "أن", "يا", "و", "ف"}
        idxs = [
            i + 1
            for i, w in enumerate(words)
            if len(re.sub(r"[^\u0600-\u06FF]", "", w)) >= 3
            and re.sub(r"[^\u0600-\u06FF/]", "", w) not in skip
        ]
        return ImportantTerms(word_numbers=idxs[:3])

    try:
        out = get_llm().with_structured_output(ImportantTerms).invoke(
            [
                {"role": "system", "content": TERM_PICKER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Verse {verse.ref.surah}:{verse.ref.ayah}\n"
                        f"Words: {json.dumps([{'word_no': i + 1, 'word': w} for i, w in enumerate(words)], ensure_ascii=False)}"
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
        parens = _ROOT_IN_PARENS.findall(sarf)
        if parens:
            return parens[-1]
    word = str(analysis.get("word") or "")
    for stem in ("صبر", "رحم", "أمن", "صلى", "غفر"):
        if stem in word:
            return stem
    return None


def run_linguistic_research(state: ResearchState) -> dict:
    tid = state.get("current_task_id") or ""
    lines = [trace("linguistic_researcher", "Analyzing roots...", blank_before=True)]
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
        lines.append(trace("linguistic_researcher", f"Skipped ({reason})."))
        return pack(
            tid,
            lines=lines,
            warnings=[f"linguistic_researcher: skipped ({reason})"],
        )

    verses = list(state.get("selected_verses") or [])
    if not verses:
        lines.append(trace("linguistic_researcher", "Skipped — no selected verses."))
        return pack(
            tid,
            lines=lines,
            warnings=["linguistic_researcher: no selected_verses"],
        )

    question = state.get("user_question") or ""
    items: list[LinguisticEvidence] = []
    warnings: list[str] = []
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
                    items.append(
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
                    client, "get_root_stats", {"root": root}, label=f"get_root_stats {root}"
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
                items.append(
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
                            "stats": stats if isinstance(stats, dict) else {"value": stats},
                            "occurrences": occ,
                        },
                    )
                )
    except MCPError as exc:
        return session_fail(
            "linguistic_researcher", "Linguistic retrieval", exc, tid, lines
        )

    root_list = sorted(r for r in roots_seen if r)
    lines.append(
        trace(
            "linguistic_researcher",
            f"Analyzed {len(items)} item(s)"
            + (f"; roots: {', '.join(root_list)}." if root_list else "."),
        )
    )
    evidence = [
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
    ]
    return pack(
        tid,
        lines=lines,
        warnings=warnings
        + [
            f"linguistic_researcher: ran ({reason}); "
            f"items={len(items)} roots={sorted(roots_seen)}"
        ],
        linguistic_evidence=items,
        evidence_items=evidence,
    )
