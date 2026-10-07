"""Research Manager supervisor — pattern selection + parallel/sequential dispatch."""

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

MANAGER_SYSTEM = """You are the Research Manager (supervisor) for Quran Scholar.

Decide the next action(s) for the investigation. You do NOT answer the question.

Choose action ONLY from this fixed enum:
- quran_research — fetch/search Quran verses
- tafsir_research — fetch tafsir (ONLY after verses are selected)
- linguistic_research — root/word analysis (needs selected verses)
- context_research — sabab al-nuzool (needs selected verses)
- gap_analysis — assess what evidence is still missing
- comparison — compare tafsir sources
- verification — verify claims against evidence
- finish — generate final report

Execution patterns (chosen from the plan; respect them):
- thematic: Quran first; then Linguistic/Tafsir/Context may run in PARALLEL when
  their depends_on are satisfied (never run tafsir before verses exist).
- verse_specific: Fetch ayah → Tafsir → optional Linguistic → optional Context (sequential).
- tafsir_comparison: Fetch ayah → Tafsir → then comparison/verification (sequential).

Rules:
- Prefer pending plan tasks whose depends_on are satisfied.
- Never dispatch tafsir/linguistic/context until selected_verses exist.
- Do not parallelize dependent tasks.
- If pending research tasks remain, do NOT jump to comparison/verification/finish.
- Never invent actions outside the enum. Never invent graph node names.
"""

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

# Preferred order within each pattern (used for sequential pick + wave sorting)
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

# Actions that require selected verses (never parallelize ahead of Quran research)
_VERSE_DEPENDENT_ACTIONS = frozenset(
    {"tafsir_research", "linguistic_research", "context_research"}
)


def choose_execution_pattern(plan: ResearchPlan | None) -> ExecutionPattern:
    """Pick thematic / verse_specific / tafsir_comparison from the plan."""
    if plan is None:
        return ExecutionPattern.THEMATIC

    focus = plan.question_focus
    if focus in (QuestionFocus.THEMATIC, QuestionFocus.MIXED):
        return ExecutionPattern.THEMATIC

    # Verse-specific: comparison-only path when no optional supporting research
    if (
        plan.needs_tafsir_comparison
        and not plan.needs_linguistic_analysis
        and not plan.needs_sabab_nuzool
    ):
        return ExecutionPattern.TAFSIR_COMPARISON

    return ExecutionPattern.VERSE_SPECIFIC


def allows_parallel_wave(pattern: ExecutionPattern) -> bool:
    """Only thematic investigations fan out independent researchers."""
    return pattern == ExecutionPattern.THEMATIC


def _pending_ready_tasks(plan: ResearchPlan | None, done: set[str]) -> list[ResearchTask]:
    if plan is None:
        return []
    ready: list[ResearchTask] = []
    for task in plan.tasks:
        if task.id in done or task.status == TaskStatus.SKIPPED:
            continue
        if all(dep in done for dep in task.depends_on):
            ready.append(task)
    return ready


def _has_selected_verses(state: ResearchState) -> bool:
    """Tafsir / linguistic / context must wait for verse selection."""
    return bool(state.get("selected_verses"))


def _evidence_summary(state: ResearchState) -> dict:
    return {
        "discovered_verses": len(state.get("discovered_verses") or []),
        "selected_verses": len(state.get("selected_verses") or []),
        "tafsir_evidence": len(state.get("tafsir_evidence") or []),
        "linguistic_evidence": len(state.get("linguistic_evidence") or []),
        "nuzool_evidence": len(state.get("nuzool_evidence") or []),
        "evidence_items": len(state.get("evidence_items") or []),
        "claims": len(state.get("claims") or []),
        "tafsir_comparisons": len(state.get("tafsir_comparisons") or []),
        "findings": len(state.get("findings") or []),
    }


def _tasks_by_action(ready: list[ResearchTask]) -> dict[str, ResearchTask]:
    """One task per researcher action (first ready wins)."""
    by_action: dict[str, ResearchTask] = {}
    for task in ready:
        action = KIND_TO_ACTION.get(task.kind)
        if action is None or action not in RESEARCHER_ACTIONS:
            continue
        if action not in by_action:
            by_action[action] = task
    return by_action


def _filter_verse_dependent(
    by_action: dict[str, ResearchTask],
    state: ResearchState,
) -> dict[str, ResearchTask]:
    """Drop tafsir/linguistic/context when verses are not yet selected."""
    if _has_selected_verses(state):
        return by_action
    return {
        action: task
        for action, task in by_action.items()
        if action not in _VERSE_DEPENDENT_ACTIONS
    }


def _ordered_actions(
    by_action: dict[str, ResearchTask],
    pattern: ExecutionPattern,
) -> list[tuple[str, ResearchTask]]:
    order = PATTERN_ORDER.get(pattern, PATTERN_ORDER[ExecutionPattern.THEMATIC])
    ordered: list[tuple[str, ResearchTask]] = []
    seen: set[str] = set()
    for action in order:
        if action in by_action:
            ordered.append((action, by_action[action]))
            seen.add(action)
    for action, task in by_action.items():
        if action not in seen:
            ordered.append((action, task))
    return ordered


def _decision_from_dispatches(
    dispatches: list[ResearchDispatch],
    pattern: ExecutionPattern,
    reasoning: str,
) -> ResearchDecision:
    primary = dispatches[0]
    return ResearchDecision(
        action=primary.action,
        task_id=primary.task_id,
        dispatches=dispatches,
        execution_pattern=pattern,
        reasoning=reasoning,
    )


def _dispatch_research_wave(
    state: ResearchState,
    pattern: ExecutionPattern,
) -> ResearchDecision | None:
    """
    Build the next researcher wave.

    - Thematic: fan out all independent ready researchers (after verse gate).
    - Verse / comparison: one sequential step in pattern order.
    Returns None when no researcher work is ready.
    """
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    ready = _pending_ready_tasks(plan, done)
    by_action = _filter_verse_dependent(_tasks_by_action(ready), state)
    ordered = _ordered_actions(by_action, pattern)
    if not ordered:
        return None

    if allows_parallel_wave(pattern) and len(ordered) > 1:
        dispatches = [
            ResearchDispatch(action=action, task_id=task.id)  # type: ignore[arg-type]
            for action, task in ordered
        ]
        names = [d.action for d in dispatches]
        return _decision_from_dispatches(
            dispatches,
            pattern,
            reasoning=(
                f"Pattern {pattern.value}: parallel independent wave {names} "
                f"(depends_on satisfied; verses selected={_has_selected_verses(state)})."
            ),
        )

    action, task = ordered[0]
    return _decision_from_dispatches(
        [ResearchDispatch(action=action, task_id=task.id)],  # type: ignore[arg-type]
        pattern,
        reasoning=(
            f"Pattern {pattern.value}: sequential next={action} "
            f"(task {task.id}, kind={task.kind})."
        ),
    )


def _pending_researcher_tasks(
    plan: ResearchPlan | None,
    done: set[str],
) -> list[ResearchTask]:
    if plan is None:
        return []
    out: list[ResearchTask] = []
    for task in plan.tasks:
        if task.id in done or task.status == TaskStatus.SKIPPED:
            continue
        if KIND_TO_ACTION.get(task.kind) in RESEARCHER_ACTIONS:
            out.append(task)
    return out


def _post_research_decision(
    state: ResearchState,
    pattern: ExecutionPattern,
) -> ResearchDecision:
    """Gap / comparison / verification / finish after researchers are done."""
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    gaps = list(state.get("unresolved_gaps") or [])

    if state.get("verification_passed"):
        return ResearchDecision(
            action="finish",
            task_id=None,
            dispatches=[],
            execution_pattern=pattern,
            reasoning="Verification passed; generate the final report.",
        )

    # Pending researcher work still exists (e.g. tafsir blocked until verses selected)
    blocked = _pending_researcher_tasks(plan, done)
    if blocked:
        ready = _pending_ready_tasks(plan, done)
        ready_actions = _tasks_by_action(ready)
        if ready_actions and not _filter_verse_dependent(ready_actions, state):
            return ResearchDecision(
                action="gap_analysis",
                task_id=None,
                dispatches=[],
                execution_pattern=pattern,
                reasoning=(
                    "Verse-dependent tasks are ready by depends_on but blocked: "
                    "no selected_verses yet (will not treat as 'no tafsir'). "
                    f"Pending: {[t.id for t in blocked]}"
                ),
            )
        if any(not all(d in done for d in t.depends_on) for t in blocked):
            return ResearchDecision(
                action="gap_analysis",
                task_id=None,
                dispatches=[],
                execution_pattern=pattern,
                reasoning=(
                    "Dependent researcher tasks still pending; "
                    f"ids={[t.id for t in blocked]}"
                ),
            )
        # Ready + verses present but wave empty shouldn't happen; gap anyway
        return ResearchDecision(
            action="gap_analysis",
            task_id=None,
            dispatches=[],
            execution_pattern=pattern,
            reasoning=f"Pending researcher tasks remain: {[t.id for t in blocked]}",
        )

    if gaps and iteration < max_iters:
        return ResearchDecision(
            action="gap_analysis",
            task_id=None,
            dispatches=[],
            execution_pattern=pattern,
            reasoning="Unresolved gaps remain; run gap analysis.",
        )

    needs_compare = bool(plan and plan.needs_tafsir_comparison)
    has_compare = bool(state.get("tafsir_comparisons"))
    if needs_compare and not has_compare:
        return ResearchDecision(
            action="comparison",
            task_id=None,
            dispatches=[],
            execution_pattern=pattern,
            reasoning="Research tasks complete; run tafsir comparison.",
        )

    if not state.get("verification_passed"):
        if iteration >= max_iters:
            return ResearchDecision(
                action="finish",
                task_id=None,
                dispatches=[],
                execution_pattern=pattern,
                reasoning="Iteration budget exhausted; finish with available evidence.",
            )
        return ResearchDecision(
            action="verification",
            task_id=None,
            dispatches=[],
            execution_pattern=pattern,
            reasoning="Evidence gathered; run evidence verification.",
        )

    return ResearchDecision(
        action="finish",
        task_id=None,
        dispatches=[],
        execution_pattern=pattern,
        reasoning="No pending research; proceed to report.",
    )


def _heuristic_decision(state: ResearchState) -> ResearchDecision:
    """Deterministic supervisor: pattern → wave → post-research stages."""
    plan = state.get("research_plan")
    pattern = choose_execution_pattern(plan)

    if state.get("verification_passed"):
        return ResearchDecision(
            action="finish",
            task_id=None,
            dispatches=[],
            execution_pattern=pattern,
            reasoning="Verification passed; generate the final report.",
        )

    wave = _dispatch_research_wave(state, pattern)
    if wave is not None:
        return wave

    return _post_research_decision(state, pattern)


def decide_next_action(state: ResearchState) -> ResearchDecision:
    """
    Produce ResearchDecision.

    Researcher dispatch is always deterministic (patterns + depends_on + verse gate)
    so parallel/sequential behavior stays reliable. LLM may refine post-research
    stages when an API key is available.
    """
    plan = state.get("research_plan")
    pattern = choose_execution_pattern(plan)

    if state.get("verification_passed"):
        return ResearchDecision(
            action="finish",
            task_id=None,
            dispatches=[],
            execution_pattern=pattern,
            reasoning="Verification passed; generate the final report.",
        )

    wave = _dispatch_research_wave(state, pattern)
    if wave is not None:
        return wave

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return _post_research_decision(state, pattern)

    done = list(state.get("completed_task_ids") or [])
    payload = {
        "user_question": state.get("user_question"),
        "language": state.get("language"),
        "execution_pattern": pattern.value,
        "research_plan": plan.model_dump() if plan else None,
        "completed_task_ids": done,
        "ready_tasks": [
            t.model_dump() for t in _pending_ready_tasks(plan, set(done))
        ],
        "selected_verses": _has_selected_verses(state),
        "evidence_counts": _evidence_summary(state),
        "unresolved_gaps": list(state.get("unresolved_gaps") or []),
        "gap_status": state.get("gap_status"),
        "research_iteration": state.get("research_iteration"),
        "max_research_iterations": state.get("max_research_iterations"),
        "verification_passed": state.get("verification_passed"),
        "needs_tafsir_comparison": bool(plan.needs_tafsir_comparison) if plan else False,
        "note": (
            "No researcher wave is ready. Choose gap_analysis, comparison, "
            "verification, or finish only."
        ),
    }

    llm = get_llm()
    structured = llm.with_structured_output(ResearchDecision)
    try:
        decision = structured.invoke(
            [
                {"role": "system", "content": MANAGER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Decide the next non-researcher action from this state:\n"
                        + json.dumps(payload, ensure_ascii=False, default=str)
                    ),
                },
            ]
        )
        if not isinstance(decision, ResearchDecision):
            decision = ResearchDecision.model_validate(decision)
        # Never let the LLM invent a researcher fan-out here — wave already empty
        if decision.action in RESEARCHER_ACTIONS:
            return _post_research_decision(state, pattern)
        return decision.model_copy(
            update={
                "execution_pattern": pattern,
                "dispatches": [],
            }
        )
    except Exception:
        return _post_research_decision(state, pattern)
