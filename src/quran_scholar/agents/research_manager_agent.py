"""Research Manager — pick execution pattern, then next wave or stage."""

from __future__ import annotations

import json
import os

from quran_scholar.agents.llm import get_llm
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

# Order used when picking the next sequential step (and sorting parallel waves)
PATTERN_ORDER: dict[ExecutionPattern, list[str]] = {
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
    ExecutionPattern.TAFSIR_COMPARISON: [
        "quran_research",
        "tafsir_research",
    ],
}

_NEEDS_VERSES = frozenset(
    {"tafsir_research", "linguistic_research", "context_research"}
)

MANAGER_SYSTEM = """You are the Research Manager for Quran Scholar.
Choose ONE next action: gap_analysis, comparison, verification, or finish.
Do not choose researcher actions. Do not invent node names."""


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


def _decision(
    action: str,
    pattern: ExecutionPattern,
    reasoning: str,
    *,
    task_id: str | None = None,
    dispatches: list[ResearchDispatch] | None = None,
) -> ResearchDecision:
    dispatches = dispatches or []
    return ResearchDecision(
        action=action,  # type: ignore[arg-type]
        task_id=task_id,
        dispatches=dispatches,
        execution_pattern=pattern,
        reasoning=reasoning,
    )


def _ready_tasks(plan: ResearchPlan | None, done: set[str]) -> list[ResearchTask]:
    if plan is None:
        return []
    return [
        t
        for t in plan.tasks
        if t.id not in done
        and t.status != TaskStatus.SKIPPED
        and all(dep in done for dep in t.depends_on)
    ]


def _pending_researchers(plan: ResearchPlan | None, done: set[str]) -> list[ResearchTask]:
    if plan is None:
        return []
    return [
        t
        for t in plan.tasks
        if t.id not in done
        and t.status != TaskStatus.SKIPPED
        and KIND_TO_ACTION.get(t.kind) in RESEARCHER_ACTIONS
    ]


def _research_wave(
    state: ResearchState, pattern: ExecutionPattern
) -> ResearchDecision | None:
    """Next researcher step(s). Thematic may fan out; others stay sequential."""
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    has_verses = bool(state.get("selected_verses"))

    # One ready task per action
    by_action: dict[str, ResearchTask] = {}
    for task in _ready_tasks(plan, done):
        action = KIND_TO_ACTION.get(task.kind)
        if action in RESEARCHER_ACTIONS and action not in by_action:
            # Tafsir / linguistic / context wait for selected verses
            if action in _NEEDS_VERSES and not has_verses:
                continue
            by_action[action] = task

    order = PATTERN_ORDER.get(pattern, PATTERN_ORDER[ExecutionPattern.THEMATIC])
    ordered = [(a, by_action[a]) for a in order if a in by_action]
    seen = {a for a, _ in ordered}
    ordered += [(a, t) for a, t in by_action.items() if a not in seen]
    if not ordered:
        return None

    parallel = pattern == ExecutionPattern.THEMATIC and len(ordered) > 1
    picks = ordered if parallel else [ordered[0]]
    dispatches = [
        ResearchDispatch(action=a, task_id=t.id)  # type: ignore[arg-type]
        for a, t in picks
    ]
    names = [d.action for d in dispatches]
    mode = "parallel" if parallel else "sequential"
    return _decision(
        dispatches[0].action,
        pattern,
        f"Pattern {pattern.value}: {mode} {names}",
        task_id=dispatches[0].task_id,
        dispatches=dispatches,
    )


def _after_research(
    state: ResearchState, pattern: ExecutionPattern
) -> ResearchDecision:
    """Gap → comparison → verification → finish once researchers are done."""
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    gaps = list(state.get("unresolved_gaps") or [])

    if state.get("verification_passed"):
        return _decision("finish", pattern, "Verification passed.")

    # Still have planned researcher work (e.g. waiting on verses)
    if _pending_researchers(plan, done):
        return _decision(
            "gap_analysis",
            pattern,
            "Researcher tasks still pending.",
        )

    if gaps and iteration < max_iters:
        return _decision("gap_analysis", pattern, "Unresolved gaps remain.")

    if plan and plan.needs_tafsir_comparison and not state.get("tafsir_comparisons"):
        return _decision("comparison", pattern, "Run tafsir comparison.")

    if not state.get("verification_passed"):
        if iteration >= max_iters:
            return _decision("finish", pattern, "Iteration budget exhausted.")
        return _decision("verification", pattern, "Run evidence verification.")

    return _decision("finish", pattern, "Proceed to report.")


def _llm_after_research(
    state: ResearchState, pattern: ExecutionPattern
) -> ResearchDecision | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None
    plan = state.get("research_plan")
    payload = {
        "execution_pattern": pattern.value,
        "gap_status": state.get("gap_status"),
        "unresolved_gaps": list(state.get("unresolved_gaps") or []),
        "needs_tafsir_comparison": bool(plan and plan.needs_tafsir_comparison),
        "has_comparisons": bool(state.get("tafsir_comparisons")),
        "verification_passed": state.get("verification_passed"),
        "research_iteration": state.get("research_iteration"),
        "max_research_iterations": state.get("max_research_iterations"),
    }
    try:
        decision = get_llm().with_structured_output(ResearchDecision).invoke(
            [
                {"role": "system", "content": MANAGER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ]
        )
        if not isinstance(decision, ResearchDecision):
            decision = ResearchDecision.model_validate(decision)
        if decision.action in RESEARCHER_ACTIONS:
            return None
        return decision.model_copy(
            update={"execution_pattern": pattern, "dispatches": []}
        )
    except Exception:
        return None


def decide_next_action(state: ResearchState) -> ResearchDecision:
    """Deterministic research waves; optional LLM only for post-research stages."""
    pattern = choose_execution_pattern(state.get("research_plan"))

    if state.get("verification_passed"):
        return _decision("finish", pattern, "Verification passed.")

    wave = _research_wave(state, pattern)
    if wave is not None:
        return wave

    return _llm_after_research(state, pattern) or _after_research(state, pattern)
