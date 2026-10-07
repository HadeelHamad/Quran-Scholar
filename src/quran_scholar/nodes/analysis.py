"""Analysis / verification / report nodes."""

from __future__ import annotations

from quran_scholar.agents.claim_extractor_agent import run_claim_extraction
from quran_scholar.agents.evidence_verifier_agent import run_evidence_verification
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
    """Compose the final scholarly report — LLM boundary (stub)."""
    question = state.get("user_question") or ""
    plan = state.get("research_plan")
    result = state.get("verification_result")
    claims = state.get("claims") or []
    report = (
        f"# Quran Scholar Report\n\n"
        f"**Question:** {question}\n\n"
        f"**Plan:** {plan.question_summary if plan else '(none)'}\n\n"
        f"**Completed tasks:** {', '.join(state.get('completed_task_ids') or [])}\n\n"
        f"**Claims:** {len(claims)}\n\n"
        f"**Verification:** "
        f"{'passed' if state.get('verification_passed') else 'not passed'}"
    )
    if result:
        report += (
            f" (coverage={result.evidence_coverage:.2f}, "
            f"verified={len(result.verified_claim_ids)}, "
            f"unsupported={len(result.unsupported_claim_ids)}, "
            f"conflicting={len(result.conflicting_claim_ids)})"
        )
    report += (
        "\n\n_Stub report body — report generator LLM polish can be added later._\n"
    )
    return {
        "final_report": report,
        "research_complete": True,
    }
