"""Tafsir Comparator — evidence-grounded agreements/differences only."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.graph.nodes.llm import get_llm
from quran_scholar.models import (
    Evidence,
    TafsirComparison,
    TafsirEvidence,
    VerseRef,
)
from quran_scholar.state import ResearchState

COMPARATOR_SYSTEM = """You compare classical tafsir excerpts for ONE Quranic verse.

Rules (critical):
- Only identify differences that are clearly supported by the retrieved evidence text.
- Do NOT infer disagreement merely because two sources use different wording.
- If sources express the same interpretation differently, classify that as AGREEMENT.
- Do not invent theological disputes not present in the excerpts.
- difference_types should name real interpretive categories when differences exist
  (e.g. linguistic nuance, legal implication, historical context) — empty if none.
- evidence_ids must be taken from the provided evidence id list only.

Do not answer the user question beyond comparing these sources."""


class ComparisonLLMOut(BaseModel):
    verse_reference: str
    agreements: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    difference_types: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    summary: str = ""


def _evidence_id_for_tafsir(t: TafsirEvidence, items: list[Evidence]) -> str | None:
    for e in items:
        if e.kind != "tafsir":
            continue
        meta = e.metadata or {}
        if meta.get("source_id") == t.source_id and e.refs:
            if e.refs[0].surah == t.ref.surah and e.refs[0].ayah == t.ref.ayah:
                return e.id
    # Stable fallback id (not in store, but ties comparison to source)
    return f"tafsir-{t.source_id}-{t.ref.surah}-{t.ref.ayah}"


def _group_by_verse(
    tafsirs: list[TafsirEvidence],
) -> dict[tuple[int, int], list[TafsirEvidence]]:
    groups: dict[tuple[int, int], list[TafsirEvidence]] = defaultdict(list)
    for t in tafsirs:
        groups[(t.ref.surah, t.ref.ayah)].append(t)
    return groups


def _deterministic_comparison(
    verse_ref: str,
    ref: VerseRef,
    group: list[TafsirEvidence],
    evidence_ids: list[str],
) -> TafsirComparison:
    """Fallback when LLM unavailable: note source coverage, no invented differences."""
    source_ids = [t.source_id for t in group]
    if len(group) < 2:
        return TafsirComparison(
            verse_reference=verse_ref,
            ref=ref,
            agreements=["Fewer than two sources — comparison deferred"],
            differences=[],
            difference_types=[],
            evidence_ids=evidence_ids,
            source_ids=source_ids,
            summary="Need ≥2 tafsir sources for a meaningful comparison.",
        )
    return TafsirComparison(
        verse_reference=verse_ref,
        ref=ref,
        agreements=[
            f"Retrieved {len(group)} sources ({', '.join(source_ids)}); "
            "wording differences not treated as disagreement without LLM review."
        ],
        differences=[],
        difference_types=[],
        evidence_ids=evidence_ids,
        source_ids=source_ids,
        summary="Deterministic placeholder — sources present; no fabricated differences.",
    )


def _llm_compare(
    verse_ref: str,
    ref: VerseRef,
    group: list[TafsirEvidence],
    evidence_ids: list[str],
) -> TafsirComparison:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return _deterministic_comparison(verse_ref, ref, group, evidence_ids)

    excerpts = []
    for t, eid in zip(group, evidence_ids):
        excerpts.append(
            {
                "evidence_id": eid,
                "source_id": t.source_id,
                "attribution": t.source_title or t.author,
                "text": t.text[:2500],
            }
        )

    try:
        llm = get_llm()
        structured = llm.with_structured_output(ComparisonLLMOut)
        out = structured.invoke(
            [
                {"role": "system", "content": COMPARATOR_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Verse: {verse_ref}\n"
                        f"Allowed evidence_ids: {evidence_ids}\n"
                        f"Excerpts:\n{json.dumps(excerpts, ensure_ascii=False)}"
                    ),
                },
            ]
        )
        if not isinstance(out, ComparisonLLMOut):
            out = ComparisonLLMOut.model_validate(out)
        # Keep only known evidence ids
        allowed = set(evidence_ids)
        eids = [e for e in out.evidence_ids if e in allowed] or evidence_ids
        return TafsirComparison(
            verse_reference=out.verse_reference or verse_ref,
            ref=ref,
            agreements=out.agreements,
            differences=out.differences,
            difference_types=out.difference_types,
            evidence_ids=eids,
            source_ids=[t.source_id for t in group],
            summary=out.summary,
        )
    except Exception:
        return _deterministic_comparison(verse_ref, ref, group, evidence_ids)


def run_tafsir_comparison(state: ResearchState) -> dict:
    """Optional: compare tafsirs only when the research plan asks for it."""
    plan = state.get("research_plan")
    wants = bool(
        plan
        and (
            plan.needs_tafsir_comparison
            or "tafsir_comparison" in (plan.required_evidence_types or [])
        )
    )
    if not wants:
        return {
            "warnings": [
                "tafsir_comparator: skipped (needs_tafsir_comparison=false)"
            ],
        }

    tafsirs = list(state.get("tafsir_evidence") or [])
    evidence_items = list(state.get("evidence_items") or [])
    selected = list(state.get("selected_verses") or [])
    if not tafsirs:
        return {
            "warnings": ["tafsir_comparator: no tafsir_evidence to compare"],
        }

    # Restrict to selected verses when present
    selected_keys = {(v.ref.surah, v.ref.ayah) for v in selected} if selected else None
    groups = _group_by_verse(tafsirs)
    comparisons: list[TafsirComparison] = []

    for (surah, ayah), group in sorted(groups.items()):
        if selected_keys is not None and (surah, ayah) not in selected_keys:
            continue
        verse_ref = f"{surah}:{ayah}"
        ref = VerseRef(surah=surah, ayah=ayah)
        eids = [_evidence_id_for_tafsir(t, evidence_items) or "" for t in group]
        eids = [e for e in eids if e]
        comparisons.append(_llm_compare(verse_ref, ref, group, eids))

    if not comparisons:
        # Fall back to all groups if selected filter emptied everything
        for (surah, ayah), group in sorted(groups.items()):
            verse_ref = f"{surah}:{ayah}"
            ref = VerseRef(surah=surah, ayah=ayah)
            eids = [_evidence_id_for_tafsir(t, evidence_items) or "" for t in group]
            eids = [e for e in eids if e]
            comparisons.append(_llm_compare(verse_ref, ref, group, eids))
    return {
        "tafsir_comparisons": comparisons,
        "warnings": [
            f"tafsir_comparator: produced {len(comparisons)} comparison(s) "
            f"from {len(tafsirs)} tafsir excerpt(s)"
        ],
    }


def tafsir_comparator_node(state: ResearchState) -> dict:
    return run_tafsir_comparison(state)
