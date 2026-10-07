"""Planner node — LLM/agent boundary (stub plan for now)."""

from __future__ import annotations

from quran_scholar.models import ResearchPlan, ResearchTask, TaskStatus
from quran_scholar.state import ResearchState


def planner_node(state: ResearchState) -> dict:
    """Decompose the user question into research tasks.

    Deterministic stub plan; replace with structured LLM output later.
    """
    question = state.get("user_question") or ""
    plan = ResearchPlan(
        question_summary=question[:240],
        approach=(
            "Search relevant verses, fetch classical tafsir, gather linguistic "
            "and nuzool context, then compare and verify claims."
        ),
        tasks=[
            ResearchTask(
                id="t1_verse_search",
                description="Find Quranic verses relevant to the question",
                kind="verse_search",
                status=TaskStatus.PENDING,
            ),
            ResearchTask(
                id="t2_tafsir",
                description="Fetch tafsir for selected verses from multiple sources",
                kind="tafsir_fetch",
                status=TaskStatus.PENDING,
                depends_on=["t1_verse_search"],
            ),
            ResearchTask(
                id="t3_linguistic",
                description="Analyze key roots/words linguistically if needed",
                kind="linguistic",
                status=TaskStatus.PENDING,
                depends_on=["t1_verse_search"],
            ),
            ResearchTask(
                id="t4_context",
                description="Gather asbab al-nuzool / surah context",
                kind="context",
                status=TaskStatus.PENDING,
                depends_on=["t1_verse_search"],
            ),
        ],
        target_tafsir_sources=["saadi", "katheer", "moyassar"],
    )
    return {"research_plan": plan, "research_iteration": 0}
