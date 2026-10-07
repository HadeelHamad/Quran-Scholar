"""LangGraph wiring for Quran Scholar (non-sequential research loop)."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from quran_scholar.nodes.analysis import (
    report_generator_node,
    tafsir_comparator_node,
)
from quran_scholar.nodes.gap_analyzer import gap_analyzer_node
from quran_scholar.nodes.planner import planner_node
from quran_scholar.nodes.research_manager import research_manager_node
from quran_scholar.nodes.researchers import (
    context_researcher_node,
    linguistic_researcher_node,
    quran_researcher_node,
    tafsir_researcher_node,
)
from quran_scholar.nodes.logging_wrap import with_state_logging
from quran_scholar.nodes.routing import (
    route_after_gap_analyzer,
    route_after_research_manager,
)
from quran_scholar.state import ResearchState


def build_graph():
    """Planner → Manager ⇄ Researchers → Gap → (optional Comparator) → Report."""
    graph = StateGraph(ResearchState)
    nodes = {
        "planner": planner_node,
        "research_manager": research_manager_node,
        "quran_researcher": quran_researcher_node,
        "tafsir_researcher": tafsir_researcher_node,
        "linguistic_researcher": linguistic_researcher_node,
        "context_researcher": context_researcher_node,
        "gap_analyzer": gap_analyzer_node,
        "tafsir_comparator": tafsir_comparator_node,
        "report_generator": report_generator_node,
    }
    for name, fn in nodes.items():
        graph.add_node(name, with_state_logging(name, fn))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research_manager")
    graph.add_conditional_edges(
        "research_manager",
        route_after_research_manager,
        {
            "quran_researcher": "quran_researcher",
            "tafsir_researcher": "tafsir_researcher",
            "linguistic_researcher": "linguistic_researcher",
            "context_researcher": "context_researcher",
            "gap_analyzer": "gap_analyzer",
            "tafsir_comparator": "tafsir_comparator",
            "report_generator": "report_generator",
        },
    )
    for researcher in (
        "quran_researcher",
        "tafsir_researcher",
        "linguistic_researcher",
        "context_researcher",
    ):
        graph.add_edge(researcher, "gap_analyzer")
    graph.add_conditional_edges(
        "gap_analyzer",
        route_after_gap_analyzer,
        {
            "research_manager": "research_manager",
            "tafsir_comparator": "tafsir_comparator",
            "report_generator": "report_generator",
        },
    )
    graph.add_edge("tafsir_comparator", "report_generator")
    graph.add_edge("report_generator", END)
    return graph.compile()


build_quran_scholar_graph = build_graph
