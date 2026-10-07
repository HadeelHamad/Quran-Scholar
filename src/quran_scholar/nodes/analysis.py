"""Analysis / verification / report nodes."""

from __future__ import annotations

from quran_scholar.agents.claim_extractor_agent import run_claim_extraction
from quran_scholar.agents.evidence_verifier_agent import run_evidence_verification
from quran_scholar.agents.report_generator_agent import run_report_generation
from quran_scholar.agents.tafsir_comparator_agent import run_tafsir_comparison
from quran_scholar.state import ResearchState


def tafsir_comparator_node(state: ResearchState) -> dict:
    """Compare retrieved tafsir — agreement unless differences are evidence-backed."""
    return run_tafsir_comparison(state)


def claim_extractor_node(state: ResearchState) -> dict:
    """Convert findings / comparisons / evidence into auditable claims."""
    return run_claim_extraction(state)


def evidence_verifier_node(state: ResearchState) -> dict:
    """Strictly verify claims (DIRECT / SYNTHESIS / UNSUPPORTED / CONFLICTING)."""
    return run_evidence_verification(state)


def report_generator_node(state: ResearchState) -> dict:
    """Arabic report from verified claims/evidence only — no unsupported material."""
    return run_report_generation(state)
