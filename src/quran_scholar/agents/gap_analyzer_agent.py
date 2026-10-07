"""Gap Analyzer — deterministic evidence checks, limited LLM semantics."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.llm import get_llm
from quran_scholar.models import (
    ResearchGap,
    ResearchPlan,
    ResearchTask,
    TaskStatus,
)
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines

SEMANTIC_SYSTEM = """You assess whether collected Quranic research evidence can answer the user question.
You receive deterministic missing_evidence already found. Only ADD gaps that are clearly
still missing for a scholarly answer. Do not invent requirements not implied by the plan.
Prefer empty additions if deterministic checks already cover the gaps.
Respond with additional_missing (strings) and optional recommended task kinds only when needed.
Do NOT answer the religious question itself."""


class SemanticGapDelta(BaseModel):
    additional_missing: list[str] = Field(default_factory=list)
    recommend_linguistic: bool = False
    recommend_nuzool: bool = False
    notes: str = ""


def _requires_quran(plan: ResearchPlan) -> bool:
    """Quran Scholar answers always need verse evidence unless plan says otherwise."""
    types = plan.required_evidence_types or []
    if types and not any(
        t in types for t in ("quran_text", "quran", "verse", "ayah")
    ):
        # Explicit types omit Quran — rare; still usually need verses as anchors
        pass
    if any(t in types for t in ("quran_text", "quran", "verse", "ayah")):
        return True
    if any(
        t.kind in ("fetch_ayah", "quran_search", "verse_search", "quran")
        for t in plan.tasks
    ):
        return True
    if plan.primary_verse is not None:
        return True
    return True  # default for this project


def _requires_tafsir(plan: ResearchPlan) -> bool:
    types = plan.required_evidence_types or []
    return (
        any("tafsir" in t for t in types)
        or any(
            t.kind in ("tafsir_fetch", "fetch_tafsir", "tafsir", "tafsir_research")
            for t in plan.tasks
        )
        or bool(plan.target_tafsir_sources)
    )


def _requires_comparison(plan: ResearchPlan) -> bool:
    types = plan.required_evidence_types or []
    return plan.needs_tafsir_comparison or "tafsir_comparison" in types


def _requires_linguistic(plan: ResearchPlan) -> bool:
    types = plan.required_evidence_types or []
    return plan.needs_linguistic_analysis or "linguistic" in types or any(
        t.kind in ("linguistic", "linguistic_analysis") for t in plan.tasks
    )


def _requires_nuzool(plan: ResearchPlan) -> bool:
    types = plan.required_evidence_types or []
    return plan.needs_sabab_nuzool or "nuzool" in types or any(
        t.kind in ("nuzool", "nuzool_research", "context") for t in plan.tasks
    )


def _nuzool_was_searched(state: ResearchState) -> bool:
    """True if context researcher produced any statused nuzool records."""
    return bool(state.get("nuzool_evidence"))


def _distinct_tafsir_sources(state: ResearchState) -> set[str]:
    return {
        t.source_id
        for t in (state.get("tafsir_evidence") or [])
        if getattr(t, "source_id", None)
    }


def _pending_tasks(plan: ResearchPlan | None, done: set[str]) -> list[ResearchTask]:
    if plan is None:
        return []
    return [
        t
        for t in plan.tasks
        if t.id not in done and t.status != TaskStatus.SKIPPED
    ]


def _new_task(task_id: str, description: str, kind: str, depends_on: list[str] | None = None) -> ResearchTask:
    return ResearchTask(
        id=task_id,
        description=description,
        kind=kind,
        depends_on=depends_on or [],
    )


def deterministic_gap_check(state: ResearchState) -> ResearchGap:
    """Plan/evidence checklist — no LLM."""
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    missing: list[str] = []
    recommended: list[ResearchTask] = []

    if plan is None:
        return ResearchGap(
            sufficient=False,
            missing_evidence=["No research_plan"],
            recommended_tasks=[],
        )

    pending = _pending_tasks(plan, done)
    has_quran = bool(state.get("selected_verses") or state.get("discovered_verses"))
    has_tafsir = bool(state.get("tafsir_evidence"))
    has_linguistic = bool(state.get("linguistic_evidence"))
    nuzool_searched = _nuzool_was_searched(state)
    sources = _distinct_tafsir_sources(state)
    has_comparisons = bool(state.get("tafsir_comparisons"))

    if _requires_quran(plan) and not has_quran:
        missing.append("Quran evidence required but no selected/discovered verses")
        if not any(t.kind in ("fetch_ayah", "quran_search", "verse_search") for t in pending):
            recommended.append(
                _new_task(
                    "gap_quran_search",
                    "Search or fetch Quran verses for the question",
                    "quran_search",
                )
            )

    if _requires_tafsir(plan) and not has_tafsir:
        missing.append("Tafsir evidence required but none retrieved")
        if not any(t.kind in ("tafsir_fetch", "fetch_tafsir", "tafsir") for t in pending):
            recommended.append(
                _new_task(
                    "gap_tafsir_fetch",
                    "Fetch tafsir for selected verses",
                    "tafsir_fetch",
                    depends_on=[],
                )
            )

    if _requires_comparison(plan):
        if len(sources) < 2 and not has_comparisons:
            missing.append(
                "Tafsir comparison required but fewer than two distinct tafsir sources"
            )
            if has_tafsir and len(sources) < 2:
                recommended.append(
                    _new_task(
                        "gap_tafsir_more_sources",
                        "Fetch additional tafsir sources for comparison",
                        "tafsir_fetch",
                    )
                )
        elif has_tafsir and len(sources) >= 2 and not has_comparisons:
            # Enough sources — comparison node still to run; not a research gap
            pass

    if _requires_linguistic(plan) and not has_linguistic:
        missing.append("Linguistic analysis required but not collected")
        if not any(t.kind in ("linguistic", "linguistic_analysis") for t in pending):
            recommended.append(
                _new_task(
                    "gap_linguistic",
                    "Analyze key Quranic terms/roots",
                    "linguistic",
                )
            )

    if _requires_nuzool(plan) and not nuzool_searched:
        missing.append("Sabab al-nuzool required but not searched")
        if not any(t.kind in ("nuzool", "context", "nuzool_research") for t in pending):
            recommended.append(
                _new_task(
                    "gap_nuzool",
                    "Fetch asbab al-nuzool for selected verses",
                    "nuzool",
                )
            )

    # Pending planned tasks still count as incomplete research
    if pending:
        missing.append(
            "Pending planned tasks: " + ", ".join(t.id for t in pending)
        )

    sufficient = not missing
    return ResearchGap(
        sufficient=sufficient,
        missing_evidence=missing,
        recommended_tasks=recommended,
    )


def _semantic_enrich(state: ResearchState, gap: ResearchGap) -> ResearchGap:
    """Optional LLM pass — only adds semantic gaps when API available."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return gap

    plan = state.get("research_plan")
    payload = {
        "user_question": state.get("user_question"),
        "plan_summary": plan.question_summary if plan else None,
        "required_evidence_types": plan.required_evidence_types if plan else [],
        "deterministic_missing": gap.missing_evidence,
        "counts": {
            "selected_verses": len(state.get("selected_verses") or []),
            "tafsir_evidence": len(state.get("tafsir_evidence") or []),
            "linguistic_evidence": len(state.get("linguistic_evidence") or []),
            "nuzool_evidence": len(state.get("nuzool_evidence") or []),
            "evidence_items": len(state.get("evidence_items") or []),
            "tafsir_sources": sorted(_distinct_tafsir_sources(state)),
        },
        "completed_task_ids": list(state.get("completed_task_ids") or []),
    }

    try:
        llm = get_llm()
        structured = llm.with_structured_output(SemanticGapDelta)
        delta = structured.invoke(
            [
                {"role": "system", "content": SEMANTIC_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ]
        )
        if not isinstance(delta, SemanticGapDelta):
            delta = SemanticGapDelta.model_validate(delta)
    except Exception:
        return gap

    missing = list(gap.missing_evidence)
    for item in delta.additional_missing:
        if item and item not in missing:
            missing.append(item)

    recommended = list(gap.recommended_tasks)
    existing_kinds = {t.kind for t in recommended}
    if delta.recommend_linguistic and "linguistic" not in existing_kinds:
        recommended.append(
            _new_task("gap_semantic_linguistic", "Semantic gap: linguistic study", "linguistic")
        )
    if delta.recommend_nuzool and "nuzool" not in existing_kinds:
        recommended.append(
            _new_task("gap_semantic_nuzool", "Semantic gap: asbab al-nuzool", "nuzool")
        )

    return ResearchGap(
        sufficient=len(missing) == 0,
        missing_evidence=missing,
        recommended_tasks=recommended,
    )


def analyze_research_gaps(state: ResearchState) -> ResearchGap:
    gap = deterministic_gap_check(state)
    # Only call LLM when deterministic path looks sufficient but we want a semantic sanity check,
    # or when there is evidence but still borderline.
    has_some_evidence = bool(
        state.get("selected_verses") or state.get("tafsir_evidence") or state.get("evidence_items")
    )
    if gap.sufficient and has_some_evidence:
        return _semantic_enrich(state, gap)
    if (not gap.sufficient) and has_some_evidence:
        # Limited LLM: may refine missing list; still cheap structured call
        return _semantic_enrich(state, gap)
    return gap


def apply_gap_to_state_updates(state: ResearchState, gap: ResearchGap) -> dict[str, Any]:
    """Translate ResearchGap into ResearchState field updates."""
    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    budget_left = iteration < max_iters

    if gap.sufficient:
        t_line = trace(
            "gap_analyzer",
            "Evidence sufficient.",
            blank_before=True,
        )
    else:
        t_line = trace(
            "gap_analyzer",
            f"Evidence insufficient — {len(gap.missing_evidence)} gap(s).",
            blank_before=True,
        )

    updates: dict[str, Any] = {
        "research_gap": gap,
        "unresolved_gaps": list(gap.missing_evidence),
        "gap_status": "sufficient" if gap.sufficient else "insufficient",
        "research_complete": bool(gap.sufficient or not budget_left),
        "research_iteration": iteration + 1,
        "warnings": [
            f"gap_analyzer: sufficient={gap.sufficient} "
            f"missing={len(gap.missing_evidence)} "
            f"recommended={len(gap.recommended_tasks)}"
        ],
        **trace_lines(t_line),
    }

    # Merge recommended tasks into plan (replace plan object)
    plan = state.get("research_plan")
    if plan and gap.recommended_tasks:
        existing_ids = {t.id for t in plan.tasks}
        new_tasks = [t for t in gap.recommended_tasks if t.id not in existing_ids]
        if new_tasks:
            updates["research_plan"] = plan.model_copy(
                update={"tasks": list(plan.tasks) + new_tasks}
            )

    if gap.sufficient:
        updates["research_iteration"] = iteration  # don't burn budget when done

    return updates
