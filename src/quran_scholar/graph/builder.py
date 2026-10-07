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
    route_after_evidence_verifier,
    route_after_gap_analyzer,
    route_after_research_manager,
)
from quran_scholar.state import ResearchState


def build_graph():
    """
    START → Planner → Research Manager ⇄ Researchers → Gap Analyzer
         ↘ insufficient                              ↙
           sufficient → Tafsir Comparator → Claim Extractor
             → Evidence Verifier ⇄ Gap Analyzer (retry)
             → Report Generator → END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("research_manager", research_manager_node)
    graph.add_node("quran_researcher", quran_researcher_node)
    graph.add_node("tafsir_researcher", tafsir_researcher_node)
    graph.add_node("linguistic_researcher", linguistic_researcher_node)
    graph.add_node("context_researcher", context_researcher_node)
    graph.add_node("gap_analyzer", gap_analyzer_node)
    graph.add_node("tafsir_comparator", tafsir_comparator_node)
    graph.add_node("claim_extractor", claim_extractor_node)
    graph.add_node("evidence_verifier", evidence_verifier_node)
    graph.add_node("report_generator", report_generator_node)

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
            "evidence_verifier": "evidence_verifier",
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
        },
    )

    graph.add_edge("tafsir_comparator", "claim_extractor")
    graph.add_edge("claim_extractor", "evidence_verifier")

    graph.add_conditional_edges(
        "evidence_verifier",
        route_after_evidence_verifier,
        {
            "research_manager": "research_manager",
            "report_generator": "report_generator",
        },
    )

    graph.add_edge("report_generator", END)
    return graph.compile()


def build_quran_scholar_graph():
    """Alias used by notebooks / entrypoints."""
    return build_graph()
