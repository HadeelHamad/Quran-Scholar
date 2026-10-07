"""Analysis / verification / report nodes."""

from __future__ import annotations

from quran_scholar.agents.tafsir_comparator_agent import run_tafsir_comparison
from quran_scholar.models import ClaimSupport, VerificationResult
from quran_scholar.state import ResearchState


def tafsir_comparator_node(state: ResearchState) -> dict:
    """Compare retrieved tafsir — agreement unless differences are evidence-backed."""
    return run_tafsir_comparison(state)


def claim_extractor_node(state: ResearchState) -> dict:
    """Extract claims grounded in evidence — LLM boundary (stub)."""
    return {
        "warnings": ["claim_extractor: stub (no claims yet)"],
    }


def evidence_verifier_node(state: ResearchState) -> dict:
    """Verify claims against evidence store — LLM + deterministic checks."""
    claims = state.get("claims") or []
    evidence = state.get("evidence_items") or []

    # Stub: pass when no unsupported claims exist (including empty claim set).
    unsupported = [c for c in claims if c.support == ClaimSupport.UNSUPPORTED]
    passed = len(unsupported) == 0

    result = VerificationResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        summary=(
            "Stub verification passed"
            if passed
            else f"{len(unsupported)} unsupported claim(s)"
        ),
        unsupported_claim_ids=[c.id for c in unsupported],
        needs_more_research=not passed,
    )
    return {
        "verification_result": result,
        "verification_passed": passed,
        "unsupported_claims": unsupported,
        "warnings": [
            f"evidence_verifier: {'passed' if passed else 'failed'} "
            f"(claims={len(claims)}, evidence={len(evidence)})"
        ],
    }


def report_generator_node(state: ResearchState) -> dict:
    """Compose the final scholarly report — LLM boundary (stub)."""
    question = state.get("user_question") or ""
    plan = state.get("research_plan")
    report = (
        f"# Quran Scholar Report\n\n"
        f"**Question:** {question}\n\n"
        f"**Plan:** {plan.question_summary if plan else '(none)'}\n\n"
        f"**Completed tasks:** {', '.join(state.get('completed_task_ids') or [])}\n\n"
        f"**Verification:** "
        f"{'passed' if state.get('verification_passed') else 'not passed'}\n\n"
        f"_Stub report — researcher MCP + LLM nodes not fully wired yet._\n"
    )
    return {
        "final_report": report,
        "research_complete": True,
    }
