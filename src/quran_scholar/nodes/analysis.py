"""Analysis / verification / report nodes."""

from __future__ import annotations

from quran_scholar.agents.claim_extractor import run_claim_extraction
from quran_scholar.agents.evidence_verifier import run_evidence_verification
from quran_scholar.agents.report_generator import run_report_generation
from quran_scholar.agents.tafsir_comparator import run_tafsir_comparison
from quran_scholar.state import ResearchState


def tafsir_comparator_node(state: ResearchState) -> dict:
    return run_tafsir_comparison(state)


def claim_extractor_node(state: ResearchState) -> dict:
    return run_claim_extraction(state)


def evidence_verifier_node(state: ResearchState) -> dict:
    return run_evidence_verification(state)


def report_generator_node(state: ResearchState) -> dict:
    return run_report_generation(state)
