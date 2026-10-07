"""LangGraph wiring for Quran Scholar (non-sequential research loop)."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from quran_scholar.nodes.analysis import (
    claim_extractor_node,
    evidence_verifier_node,
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
from quran_scholar.nodes.routing import (
    route_after_gap_analyzer,
    route_after_research_manager,
    verification_route,
)
from quran_scholar.state import ResearchState


def build_graph():
    """Planner → Manager ⇄ Researchers → Gap → (optional Comparator) → Claims → Verify → Report."""
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
        "claim_extractor": claim_extractor_node,
        "evidence_verifier": evidence_verifier_node,
        "report_generator": report_generator_node,
    }
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research_manager")
    graph.add_conditional_edges(
        "research_manager",
        route_after_research_manager,
        {k: k for k in nodes if k != "planner"},
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
            "claim_extractor": "claim_extractor",
        },
    )
    graph.add_edge("tafsir_comparator", "claim_extractor")
    graph.add_edge("claim_extractor", "evidence_verifier")
    graph.add_conditional_edges(
        "evidence_verifier",
        verification_route,
        {"gap_analyzer": "gap_analyzer", "report_generator": "report_generator"},
    )
    graph.add_edge("report_generator", END)
    return graph.compile()


build_quran_scholar_graph = build_graph
