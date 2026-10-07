"""Gap Analyzer — deterministic checklist against the research plan."""

from __future__ import annotations

from typing import Any

from quran_scholar.models import ResearchGap, ResearchPlan, ResearchTask, TaskStatus
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines


def run_gap_analysis(state: ResearchState) -> dict[str, Any]:
    """Check whether planned evidence types are present; recommend follow-up tasks."""
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    missing: list[str] = []
    recommended: list[ResearchTask] = []

    def add_task(tid: str, description: str, kind: str) -> None:
        recommended.append(ResearchTask(id=tid, description=description, kind=kind))

    if plan is None:
        gap = ResearchGap(
            sufficient=False,
            missing_evidence=["No research_plan"],
            recommended_tasks=[],
        )
    else:
        pending = [
            t
            for t in plan.tasks
            if t.id not in done and t.status != TaskStatus.SKIPPED
        ]
        types = plan.required_evidence_types or []
        has_quran = bool(
            state.get("selected_verses")
            or state.get("discovered_verses")
            or any(
                getattr(e, "kind", None) == "quran_meta"
                for e in (state.get("evidence_items") or [])
            )
        )
        has_tafsir = bool(state.get("tafsir_evidence"))
        sources = {
            t.source_id
            for t in (state.get("tafsir_evidence") or [])
            if getattr(t, "source_id", None)
        }

        needs_quran = (
            not types
            or any(
                t in types
                for t in ("quran_text", "surah_info", "qiraat", "statistics")
            )
            or any(
                t.kind in ("fetch_ayah", "quran_search", "verse_search")
                for t in plan.tasks
            )
        )
        needs_tafsir = any("tafsir" in t for t in types) or any(
            t.kind in ("tafsir_fetch", "fetch_tafsir", "tafsir") for t in plan.tasks
        )
        needs_ling = plan.needs_linguistic_analysis or "linguistic" in types
        needs_nuzool = plan.needs_sabab_nuzool or "nuzool" in types
        needs_compare = plan.needs_tafsir_comparison or "tafsir_comparison" in types

        if needs_quran and not has_quran:
            missing.append("Quran evidence required but none collected")
            if not any(
                t.kind in ("fetch_ayah", "quran_search", "verse_search")
                for t in pending
            ):
                add_task("gap_quran_search", "Search or fetch Quran verses", "quran_search")

        if needs_tafsir and not has_tafsir:
            missing.append("Tafsir evidence required but none retrieved")
            if not any(
                t.kind in ("tafsir_fetch", "fetch_tafsir", "tafsir") for t in pending
            ):
                add_task(
                    "gap_tafsir_fetch",
                    "Fetch tafsir for selected verses",
                    "tafsir_fetch",
                )

        if needs_compare and len(sources) < 2 and not state.get("tafsir_comparisons"):
            missing.append(
                "Tafsir comparison required but fewer than two distinct sources"
            )
            if has_tafsir:
                add_task(
                    "gap_tafsir_more_sources",
                    "Fetch additional tafsir sources for comparison",
                    "tafsir_fetch",
                )

        if needs_ling and not state.get("linguistic_evidence"):
            missing.append("Linguistic analysis required but not collected")
            if not any(
                t.kind in ("linguistic", "linguistic_analysis") for t in pending
            ):
                add_task(
                    "gap_linguistic", "Analyze key Quranic terms/roots", "linguistic"
                )

        if needs_nuzool and not state.get("nuzool_evidence"):
            missing.append("Sabab al-nuzool required but not searched")
            if not any(
                t.kind in ("nuzool", "context", "nuzool_research") for t in pending
            ):
                add_task(
                    "gap_nuzool",
                    "Fetch asbab al-nuzool for selected verses",
                    "nuzool",
                )

        if pending:
            missing.append(
                "Pending planned tasks: " + ", ".join(t.id for t in pending)
            )

        gap = ResearchGap(
            sufficient=not missing,
            missing_evidence=missing,
            recommended_tasks=recommended,
        )

    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    t_line = trace(
        "gap_analyzer",
        "Evidence sufficient."
        if gap.sufficient
        else f"Evidence insufficient — {len(gap.missing_evidence)} gap(s).",
        blank_before=True,
    )
    updates: dict[str, Any] = {
        "research_gap": gap,
        "unresolved_gaps": list(gap.missing_evidence),
        "gap_status": "sufficient" if gap.sufficient else "insufficient",
        "research_complete": bool(gap.sufficient or iteration >= max_iters),
        "research_iteration": iteration if gap.sufficient else iteration + 1,
        "warnings": [
            f"gap_analyzer: sufficient={gap.sufficient} "
            f"missing={len(gap.missing_evidence)} "
            f"recommended={len(gap.recommended_tasks)}"
        ],
        **trace_lines(t_line),
    }
    if plan and gap.recommended_tasks:
        existing = {t.id for t in plan.tasks}
        new_tasks = [t for t in gap.recommended_tasks if t.id not in existing]
        if new_tasks:
            updates["research_plan"] = plan.model_copy(
                update={"tasks": list(plan.tasks) + new_tasks}
            )
    return updates
