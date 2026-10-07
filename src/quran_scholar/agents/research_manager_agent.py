"""Research Manager supervisor — structured next-action decision."""

from __future__ import annotations

import json
import os

from quran_scholar.agents.llm import get_llm
from quran_scholar.models import (
    ResearchDecision,
    ResearchPlan,
    ResearchTask,
    TaskStatus,
)
from quran_scholar.state import ResearchState

MANAGER_SYSTEM = """You are the Research Manager (supervisor) for Quran Scholar.

Decide the SINGLE next action for the investigation. You do NOT answer the question.

Choose action ONLY from this fixed enum:
- quran_research — fetch/search Quran verses (tasks: fetch_ayah, quran_search, …)
- tafsir_research — fetch tafsir (tasks: tafsir_fetch, …)
- linguistic_research — root/word analysis
- context_research — sabab al-nuzool / surah context
- gap_analysis — assess what evidence is still missing
- comparison — compare tafsir sources (when research is sufficient and comparison needed)
- verification — verify claims against evidence
- finish — generate final report (only when verification passed or budget exhausted)

Rules:
- Prefer pending plan tasks whose depends_on are satisfied; set task_id to that task.
- Map task kinds: fetch_ayah/quran_search → quran_research; tafsir_fetch → tafsir_research;
  linguistic → linguistic_research; nuzool → context_research.
- If pending research tasks remain, do NOT jump to comparison/verification/finish.
- If all researcher tasks are done and gaps remain → gap_analysis.
- If evidence is sufficient and needs_tafsir_comparison → comparison (then pipeline continues).
- If comparison already done (tafsir_comparisons or findings present) and claims need check → verification.
- If verification_passed or max iterations exhausted with nothing left → finish.
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


def _heuristic_decision(state: ResearchState) -> ResearchDecision:
    """Deterministic supervisor when LLM is unavailable."""
    plan = state.get("research_plan")
    done = set(state.get("completed_task_ids") or [])
    ready = _pending_ready_tasks(plan, done)
    iteration = int(state.get("research_iteration") or 0)
    max_iters = int(state.get("max_research_iterations") or 3)
    gaps = list(state.get("unresolved_gaps") or [])
    gap_status = state.get("gap_status") or "insufficient"

    if state.get("verification_passed"):
        return ResearchDecision(
            action="finish",
            task_id=None,
            reasoning="Verification passed; generate the final report.",
        )

    if ready:
        task = ready[0]
        action = KIND_TO_ACTION.get(task.kind, "quran_research")
        return ResearchDecision(
            action=action,  # type: ignore[arg-type]
            task_id=task.id,
            reasoning=f"Next ready plan task {task.id} ({task.kind}).",
        )

    # No ready researcher tasks
    if gaps and iteration < max_iters:
        return ResearchDecision(
            action="gap_analysis",
            task_id=None,
            reasoning="Unresolved gaps remain; run gap analysis.",
        )

    needs_compare = bool(plan and plan.needs_tafsir_comparison)
    has_compare = bool(state.get("tafsir_comparisons"))
    if needs_compare and not has_compare:
        return ResearchDecision(
            action="comparison",
            task_id=None,
            reasoning="Research tasks complete; run tafsir comparison.",
        )

    if not state.get("verification_passed"):
        if iteration >= max_iters:
            return ResearchDecision(
                action="finish",
                task_id=None,
                reasoning="Iteration budget exhausted; finish with available evidence.",
            )
        return ResearchDecision(
            action="verification",
            task_id=None,
            reasoning="Evidence gathered; run evidence verification.",
        )

    return ResearchDecision(
        action="finish",
        task_id=None,
        reasoning="No pending research; proceed to report.",
    )


def decide_next_action(state: ResearchState) -> ResearchDecision:
    """Produce ResearchDecision via structured LLM, with heuristic fallback."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return _heuristic_decision(state)

    plan = state.get("research_plan")
    done = list(state.get("completed_task_ids") or [])
    payload = {
        "user_question": state.get("user_question"),
        "language": state.get("language"),
        "research_plan": plan.model_dump() if plan else None,
        "completed_task_ids": done,
        "ready_tasks": [
            t.model_dump()
            for t in _pending_ready_tasks(plan, set(done))
        ],
        "evidence_counts": _evidence_summary(state),
        "unresolved_gaps": list(state.get("unresolved_gaps") or []),
        "gap_status": state.get("gap_status"),
        "research_iteration": state.get("research_iteration"),
        "max_research_iterations": state.get("max_research_iterations"),
        "verification_passed": state.get("verification_passed"),
        "needs_tafsir_comparison": bool(plan.needs_tafsir_comparison) if plan else False,
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
                        "Decide the next action from this research state:\n"
                        + json.dumps(payload, ensure_ascii=False, default=str)
                    ),
                },
            ]
        )
        if not isinstance(decision, ResearchDecision):
            decision = ResearchDecision.model_validate(decision)
        return decision
    except Exception:
        return _heuristic_decision(state)
