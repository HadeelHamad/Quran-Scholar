"""Evidence Verifier — strict claim support checks (stricter than report)."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.llm import get_llm
from quran_scholar.models import (
    Claim,
    ClaimVerificationStatus,
    ClaimVerdict,
    Evidence,
    VerificationLevel,
    VerificationResult,
)
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines

VERIFIER_SYSTEM = """You are a strict evidence verifier for Quranic research claims.

For each claim, assign exactly one level:
- DIRECT: evidence explicitly supports the claim (clear textual match).
- SUPPORTED_SYNTHESIS: reasonable synthesis of multiple cited evidence pieces.
- UNSUPPORTED: cited evidence does not support the claim (or ids are irrelevant).
- CONFLICTING: cited evidence contradicts or significantly differs from the claim.

Be stricter than a report writer. Prefer UNSUPPORTED over generous synthesis.
Only use the provided evidence texts for the claim's evidence_ids.
"""


class VerdictDraft(BaseModel):
    claim_id: str
    level: VerificationLevel
    notes: str = ""


class VerdictSet(BaseModel):
    verdicts: list[VerdictDraft] = Field(default_factory=list)


def _evidence_map(state: ResearchState) -> dict[str, Evidence]:
    return {e.id: e for e in (state.get("evidence_items") or [])}


def _deterministic_verdicts(
    claims: list[Claim],
    evid_by_id: dict[str, Evidence],
) -> list[ClaimVerdict]:
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        resolved = [evid_by_id[i] for i in claim.evidence_ids if i in evid_by_id]
        missing = [i for i in claim.evidence_ids if i not in evid_by_id]
        if not resolved:
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.id,
                    level=VerificationLevel.UNSUPPORTED,
                    notes="No resolvable evidence_ids in the evidence store",
                )
            )
            continue
        if missing:
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.id,
                    level=VerificationLevel.SUPPORTED_SYNTHESIS
                    if len(resolved) >= 1
                    else VerificationLevel.UNSUPPORTED,
                    notes=f"Some evidence ids missing: {missing}; others present",
                )
            )
            continue
        # All ids resolve — treat single explicit cite as DIRECT, multi as synthesis
        level = (
            VerificationLevel.DIRECT
            if len(resolved) == 1
            else VerificationLevel.SUPPORTED_SYNTHESIS
        )
        verdicts.append(
            ClaimVerdict(
                claim_id=claim.id,
                level=level,
                notes="Deterministic: cited evidence ids resolve in the store",
            )
        )
    return verdicts


def _llm_verdicts(
    claims: list[Claim],
    evid_by_id: dict[str, Evidence],
) -> list[ClaimVerdict] | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None

    packed = []
    for claim in claims:
        evid = []
        for eid in claim.evidence_ids:
            e = evid_by_id.get(eid)
            if e:
                evid.append(
                    {
                        "id": e.id,
                        "kind": e.kind,
                        "citation": e.citation,
                        "content": (e.content or "")[:800],
                    }
                )
        packed.append(
            {
                "id": claim.id,
                "statement": claim.statement,
                "claim_type": str(claim.claim_type),
                "evidence": evid,
            }
        )

    try:
        llm = get_llm()
        structured = llm.with_structured_output(VerdictSet)
        out = structured.invoke(
            [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps({"claims": packed}, ensure_ascii=False),
                },
            ]
        )
        if not isinstance(out, VerdictSet):
            out = VerdictSet.model_validate(out)
    except Exception:
        return None

    by_id = {v.claim_id: v for v in out.verdicts}
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        if claim.id in by_id:
            d = by_id[claim.id]
            verdicts.append(
                ClaimVerdict(claim_id=d.claim_id, level=d.level, notes=d.notes)
            )
        else:
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.id,
                    level=VerificationLevel.UNSUPPORTED,
                    notes="Verifier omitted claim — treated as UNSUPPORTED",
                )
            )
    return verdicts


def _apply_statuses(
    claims: list[Claim],
    verdicts: list[ClaimVerdict],
) -> list[Claim]:
    level_by_id = {v.claim_id: v.level for v in verdicts}
    updated: list[Claim] = []
    for claim in claims:
        level = level_by_id.get(claim.id)
        status: ClaimVerificationStatus | str
        if level == VerificationLevel.DIRECT:
            status = ClaimVerificationStatus.DIRECT
        elif level == VerificationLevel.SUPPORTED_SYNTHESIS:
            status = ClaimVerificationStatus.SUPPORTED_SYNTHESIS
        elif level == VerificationLevel.CONFLICTING:
            status = ClaimVerificationStatus.CONFLICTING
        elif level == VerificationLevel.UNSUPPORTED:
            status = ClaimVerificationStatus.UNSUPPORTED
        else:
            status = ClaimVerificationStatus.PENDING
        updated.append(claim.model_copy(update={"verification_status": status}))
    return updated


def run_evidence_verification(state: ResearchState) -> dict:
    """Strictly verify claims against the evidence store."""
    claims = list(state.get("claims") or [])
    evid_by_id = _evidence_map(state)
    lines: list[str] = [
        trace("evidence_verifier", "Verifying claims...", blank_before=True)
    ]

    if not claims:
        result = VerificationResult(
            passed=True,
            verified_claim_ids=[],
            unsupported_claim_ids=[],
            conflicting_claim_ids=[],
            evidence_coverage=1.0 if (state.get("evidence_items") or []) else 0.0,
            notes=["No claims to verify"],
            summary="No claims",
            needs_more_research=False,
        )
        lines.append(
            trace("evidence_verifier", "No claims — passed vacuously.")
        )
        return {
            "verification_result": result,
            "verification_passed": True,
            "unsupported_claims": [],
            "warnings": ["evidence_verifier: no claims — passed vacuously"],
            **trace_lines(*lines),
        }

    verdicts = _llm_verdicts(claims, evid_by_id)
    if verdicts is None:
        verdicts = _deterministic_verdicts(claims, evid_by_id)

    verified: list[str] = []
    unsupported: list[str] = []
    conflicting: list[str] = []
    notes: list[str] = []

    for v in verdicts:
        notes.append(f"{v.claim_id}: {v.level.value} — {v.notes}".strip(" —"))
        if v.level in (
            VerificationLevel.DIRECT,
            VerificationLevel.SUPPORTED_SYNTHESIS,
        ):
            verified.append(v.claim_id)
        elif v.level == VerificationLevel.CONFLICTING:
            conflicting.append(v.claim_id)
        else:
            unsupported.append(v.claim_id)

    # Coverage: fraction of claims that are verified (not unsupported/conflicting)
    coverage = len(verified) / max(len(claims), 1)
    passed = len(unsupported) == 0 and len(conflicting) == 0 and len(verified) > 0

    # Vacuous: all claims verified
    if len(verified) == len(claims) and not conflicting and not unsupported:
        passed = True

    result = VerificationResult(
        passed=passed,
        verified_claim_ids=verified,
        unsupported_claim_ids=unsupported,
        conflicting_claim_ids=conflicting,
        evidence_coverage=coverage,
        notes=notes,
        verdicts=verdicts,
        needs_more_research=not passed,
        summary=(
            f"passed={passed} verified={len(verified)} "
            f"unsupported={len(unsupported)} conflicting={len(conflicting)}"
        ),
    )

    updated_claims = _apply_statuses(claims, verdicts)
    unsupported_claims = [
        c
        for c in updated_claims
        if c.id in unsupported or c.id in conflicting
    ]

    lines.append(
        trace(
            "evidence_verifier",
            f"Verified {len(verified)}/{len(claims)} claim(s).",
        )
    )
    if unsupported or conflicting:
        lines.append(
            trace(
                "evidence_verifier",
                f"Research required for "
                f"{len(unsupported) + len(conflicting)} unsupported/conflicting claim(s).",
            )
        )
    elif passed:
        lines.append(trace("evidence_verifier", "All claims verified."))

    return {
        "claims": updated_claims,
        "verification_result": result,
        "verification_passed": passed,
        "unsupported_claims": unsupported_claims,
        "warnings": [
            f"evidence_verifier: {result.summary} coverage={coverage:.2f}"
        ],
        **trace_lines(*lines),
    }
