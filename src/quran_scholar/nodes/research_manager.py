"""Research Manager — selects the next research task (deterministic)."""

from __future__ import annotations

from quran_scholar.models import TaskStatus
from quran_scholar.state import ResearchState


def research_manager_node(state: ResearchState) -> dict:
    """Pick the next incomplete task whose dependencies are satisfied."""
    plan = state.get("research_plan")
    if plan is None:
        return {
            "current_task_id": "",
            "warnings": ["research_manager: missing research_plan"],
        }

    done = set(state.get("completed_task_ids") or [])
    for task in plan.tasks:
        if task.id in done or task.status == TaskStatus.SKIPPED:
            continue
        if all(dep in done for dep in task.depends_on):
            return {"current_task_id": task.id}

    # Nothing left to dispatch
    return {"current_task_id": "", "research_complete": True}
