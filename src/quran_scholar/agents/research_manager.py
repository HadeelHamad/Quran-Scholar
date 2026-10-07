"""Research Manager — next researcher wave or finish (deterministic)."""

from __future__ import annotations

from quran_scholar.models import (
    ExecutionPattern,
    QuestionFocus,
    RESEARCHER_ACTIONS,
    ResearchDecision,
    ResearchDispatch,
    ResearchPlan,
    ResearchTask,
    TaskStatus,
)
from quran_scholar.state import ResearchState

KIND_TO_ACTION = {
    "fetch_ayah": "quran_research",
    "quran_search": "quran_research",
    "verse_search": "quran_research",
    "quran": "quran_research",
    "tafsir_fetch": "tafsir_research",
    "fetch_tafsir": "tafsir_research",
    "tafsir_research": "tafsir_research",
    "tafsir": "tafsir_research",
    "linguistic": "linguistic_research",
    "linguistic_analysis": "linguistic_research",
    "nuzool": "context_research",
    "nuzool_research": "context_research",
    "context": "context_research",
}

PATTERN_ORDER = {
    ExecutionPattern.THEMATIC: [
        "quran_research",
        "linguistic_research",
        "tafsir_research",
        "context_research",
    ],
    ExecutionPattern.VERSE_SPECIFIC: [
        "quran_research",
        "tafsir_research",
        "linguistic_research",
        "context_research",
    ],
    ExecutionPattern.TAFSIR_COMPARISON: ["quran_research", "tafsir_research"],
}

_NEEDS_VERSES = frozenset(
    {"tafsir_research", "linguistic_research", "context_research"}
)


def choose_execution_pattern(plan: ResearchPlan | None) -> ExecutionPattern:
    if plan is None:
        return ExecutionPattern.THEMATIC
    if plan.question_focus in (QuestionFocus.THEMATIC, QuestionFocus.MIXED):
        return ExecutionPattern.THEMATIC
    if (
        plan.needs_tafsir_comparison
        and not plan.needs_linguistic_analysis
        and not plan.needs_sabab_nuzool
    ):
        return ExecutionPattern.TAFSIR_COMPARISON
    return ExecutionPattern.VERSE_SPECIFIC


def decide_next_action(state: ResearchState) -> ResearchDecision:
    """Pick next researcher wave(s), gap check, optional comparison, or finish."""
    plan = state.get("research_plan")
    pattern = choose_execution_pattern(plan)
    done = set(state.get("completed_task_ids") or [])

    def decision(
        action: str,
        reasoning: str,
        *,
        task_id: str | None = None,
        dispatches: list[ResearchDispatch] | None = None,
    ) -> ResearchDecision:
        return ResearchDecision(
            action=action,  # type: ignore[arg-type]
            task_id=task_id,
            dispatches=dispatches or [],
            execution_pattern=pattern,
            reasoning=reasoning,
        )

    has_verses = bool(state.get("selected_verses"))
    by_action: dict[str, ResearchTask] = {}
    if plan:
        for task in plan.tasks:
            if task.id in done or task.status == TaskStatus.SKIPPED:
                continue
            if not all(dep in done for dep in task.depends_on):
                continue
            action = KIND_TO_ACTION.get(task.kind)
            if action not in RESEARCHER_ACTIONS or action in by_action:
                continue
            if action in _NEEDS_VERSES and not has_verses:
                continue
            by_action[action] = task

    order = PATTERN_ORDER.get(pattern, PATTERN_ORDER[ExecutionPattern.THEMATIC])
    ordered = [(a, by_action[a]) for a in order if a in by_action]
    seen = {a for a, _ in ordered}
    ordered += [(a, t) for a, t in by_action.items() if a not in seen]

    if ordered:
        parallel = pattern == ExecutionPattern.THEMATIC and len(ordered) > 1
        picks = ordered if parallel else [ordered[0]]
        dispatches = [
            ResearchDispatch(action=a, task_id=t.id)  # type: ignore[arg-type]
            for a, t in picks
        ]
        mode = "parallel" if parallel else "sequential"
        return decision(
            dispatches[0].action,
            f"Pattern {pattern.value}: {mode} {[d.action for d in dispatches]}",
            task_id=dispatches[0].task_id,
            dispatches=dispatches,
        )

    pending_research = bool(
        plan
        and any(
            t.id not in done
            and t.status != TaskStatus.SKIPPED
            and KIND_TO_ACTION.get(t.kind) in RESEARCHER_ACTIONS
            for t in plan.tasks
        )
    )
    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    gaps = list(state.get("unresolved_gaps") or [])

    if pending_research:
        return decision("gap_analysis", "Researcher tasks still pending.")
    if gaps and iteration < max_iters:
        return decision("gap_analysis", "Unresolved gaps remain.")
    if plan and plan.needs_tafsir_comparison and not state.get("tafsir_comparisons"):
        return decision("comparison", "Run optional tafsir comparison.")
    return decision("finish", "Research complete — write the answer.")
