"""Claim Extractor — auditable claims with required evidence_ids."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from quran_scholar.agents.llm import get_llm
from quran_scholar.models import (
    Claim,
    ClaimType,
    ClaimVerificationStatus,
    Evidence,
    Finding,
    TafsirComparison,
)
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines

EXTRACTOR_SYSTEM = """You extract auditable claims from Quranic research materials.

Rules:
- Every claim MUST cite one or more evidence_ids from the provided catalog.
- Do NOT invent evidence ids.
- Prefer precise, checkable statements (direct_fact, synthesis, comparison, interpretation).
- Do not write the final report; only list claims.
- If evidence is thin, extract fewer claims — never unsupported speculation.
"""


class DraftClaim(BaseModel):
    id: str
    statement: str
    claim_type: str = "direct_fact"
    evidence_ids: list[str] = Field(default_factory=list)


class DraftClaimSet(BaseModel):
    claims: list[DraftClaim] = Field(default_factory=list)


def _reject_unverified(claims: list[Claim]) -> list[Claim]:
    """Reject claims that have no evidence IDs."""
    kept = [c for c in claims if c.evidence_ids]
    assert all(claim.evidence_ids for claim in kept)
    return kept


def _catalog(state: ResearchState) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for e in state.get("evidence_items") or []:
        items.append(
            {
                "id": e.id,
                "kind": e.kind,
                "citation": e.citation,
                "content": (e.content or "")[:500],
            }
        )
    return items


def _deterministic_claims(state: ResearchState) -> list[Claim]:
    """Build minimal grounded claims when LLM is unavailable."""
    claims: list[Claim] = []
    evidence = list(state.get("evidence_items") or [])
    by_kind: dict[str, list[Evidence]] = {}
    for e in evidence:
        by_kind.setdefault(e.kind, []).append(e)

    n = 1
    for e in by_kind.get("verse", [])[:5]:
        ref = e.refs[0] if e.refs else None
        ref_s = f"{ref.surah}:{ref.ayah}" if ref else e.citation
        claims.append(
            Claim(
                id=f"claim_{n:03d}",
                statement=f"The research includes Quranic verse {ref_s} as primary evidence.",
                claim_type=ClaimType.DIRECT_FACT,
                evidence_ids=[e.id],
                verification_status=ClaimVerificationStatus.PENDING,
            )
        )
        n += 1

    for e in by_kind.get("tafsir", [])[:5]:
        claims.append(
            Claim(
                id=f"claim_{n:03d}",
                statement=(
                    f"Tafsir source cited as {e.citation} provides interpretation "
                    f"for the selected verse(s)."
                ),
                claim_type=ClaimType.INTERPRETATION,
                evidence_ids=[e.id],
                verification_status=ClaimVerificationStatus.PENDING,
            )
        )
        n += 1

    for cmp in state.get("tafsir_comparisons") or []:
        if not cmp.evidence_ids:
            continue
        if cmp.agreements:
            claims.append(
                Claim(
                    id=f"claim_{n:03d}",
                    statement=(
                        f"On {cmp.verse_reference}, retrieved tafsirs agree that: "
                        f"{cmp.agreements[0]}"
                    ),
                    claim_type=ClaimType.COMPARISON,
                    evidence_ids=list(cmp.evidence_ids),
                    verification_status=ClaimVerificationStatus.PENDING,
                )
            )
            n += 1
        for diff in (cmp.differences or [])[:2]:
            claims.append(
                Claim(
                    id=f"claim_{n:03d}",
                    statement=(
                        f"On {cmp.verse_reference}, retrieved tafsirs differ: {diff}"
                    ),
                    claim_type=ClaimType.COMPARISON,
                    evidence_ids=list(cmp.evidence_ids),
                    verification_status=ClaimVerificationStatus.PENDING,
                )
            )
            n += 1

    for finding in state.get("findings") or []:
        if not finding.evidence_ids:
            continue
        claims.append(
            Claim(
                id=f"claim_{n:03d}",
                statement=finding.summary,
                claim_type=ClaimType.SYNTHESIS,
                evidence_ids=list(finding.evidence_ids),
                verification_status=ClaimVerificationStatus.PENDING,
            )
        )
        n += 1

    return _reject_unverified(claims)


def _llm_claims(state: ResearchState) -> list[Claim] | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None

    catalog = _catalog(state)
    if not catalog:
        return []

    allowed = {e["id"] for e in catalog}
    payload = {
        "user_question": state.get("user_question"),
        "evidence_catalog": catalog,
        "findings": [
            f.model_dump() for f in (state.get("findings") or [])
        ],
        "tafsir_comparisons": [
            {
                "verse_reference": c.verse_reference,
                "agreements": c.agreements,
                "differences": c.differences,
                "evidence_ids": c.evidence_ids,
            }
            for c in (state.get("tafsir_comparisons") or [])
        ],
    }

    try:
        llm = get_llm()
        structured = llm.with_structured_output(DraftClaimSet)
        out = structured.invoke(
            [
                {"role": "system", "content": EXTRACTOR_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ]
        )
        if not isinstance(out, DraftClaimSet):
            out = DraftClaimSet.model_validate(out)
    except Exception:
        return None

    claims: list[Claim] = []
    for i, d in enumerate(out.claims, start=1):
        eids = [e for e in d.evidence_ids if e in allowed]
        if not eids:
            continue
        claims.append(
            Claim(
                id=d.id or f"claim_{i:03d}",
                statement=d.statement,
                claim_type=d.claim_type or ClaimType.DIRECT_FACT,
                evidence_ids=eids,
                verification_status=ClaimVerificationStatus.PENDING,
            )
        )
    return _reject_unverified(claims)


def run_claim_extraction(state: ResearchState) -> dict:
    """Convert findings / comparisons / evidence into auditable claims."""
    evidence = state.get("evidence_items") or []
    t0 = trace("claim_extractor", "Extracting claims...", blank_before=True)
    if not evidence and not (state.get("tafsir_comparisons") or []):
        t1 = trace("claim_extractor", "No evidence to ground claims.")
        return {
            "claims": [],
            "warnings": ["claim_extractor: no evidence to ground claims"],
            **trace_lines(t0, t1),
        }

    claims = _llm_claims(state)
    if claims is None:
        claims = _deterministic_claims(state)

    claims = _reject_unverified(claims)
    t1 = trace("claim_extractor", f"Extracted {len(claims)} claim(s).")
    return {
        "claims": claims,
        "warnings": [
            f"claim_extractor: extracted {len(claims)} claim(s) "
            f"(all with evidence_ids)"
        ],
        **trace_lines(t0, t1),
    }
