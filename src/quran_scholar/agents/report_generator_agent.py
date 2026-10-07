"""Report Generator — constrained Arabic report from verified material only."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from quran_scholar.agents.llm import get_llm
from quran_scholar.models import (
    Claim,
    ClaimVerificationStatus,
    Evidence,
    TafsirComparison,
    VerificationResult,
)
from quran_scholar.services.citation_manager import citation_manager
from quran_scholar.state import ResearchState

REPORT_SYSTEM = """You are the Report Generator for Quran Scholar.

Write the entire report in Modern Standard Arabic (الفصحى).
Do not write the report body in English.

Hard constraints:
1) Use ONLY the supplied materials (verified claims, verified evidence, citations, tafsir comparisons, research warnings).
2) Do NOT introduce new Quranic facts that are absent from the supplied evidence.
3) Do NOT invent citation syntax. Use ONLY the provided citation.label strings
   (e.g. [Quran 2:153], [Tafsir Ibn Kathir — 2:153]). Never invent brackets or source names.
4) Do NOT attribute an interpretation to a mufassir/source unless the supplied evidence/citation explicitly identifies that source.
5) Do NOT treat unsupported or conflicting claims as usable evidence (they are excluded from your inputs on purpose).
6) If the verified evidence is insufficient to answer the question, state that explicitly in Arabic and do not fill gaps from your own knowledge.

Suggested report structure (Arabic headings):
- السؤال
- الخلاصة الموثَّقة
- الأدلة المعتمدة
- التفسير والمقارنة (only if supplied)
- حدود المعرفة / التحذيرات
"""


def _status_value(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value)
    return str(status)


def _verified_claims(state: ResearchState) -> list[Claim]:
    """Usable claims only — never unsupported/conflicting."""
    result: VerificationResult | None = state.get("verification_result")
    claims = list(state.get("claims") or [])
    verified_ids = set(result.verified_claim_ids) if result else set()
    blocked = set()
    if result:
        blocked |= set(result.unsupported_claim_ids or [])
        blocked |= set(result.conflicting_claim_ids or [])

    ok_status = {
        ClaimVerificationStatus.DIRECT.value,
        ClaimVerificationStatus.SUPPORTED_SYNTHESIS.value,
        "DIRECT",
        "SUPPORTED_SYNTHESIS",
    }

    usable: list[Claim] = []
    for c in claims:
        if c.id in blocked:
            continue
        if verified_ids and c.id not in verified_ids:
            continue
        if verified_ids and c.id in verified_ids:
            usable.append(c)
            continue
        # Fallback if verifier ids missing: trust status flags
        if _status_value(c.verification_status) in ok_status:
            usable.append(c)
    return usable


def _verified_evidence(
    state: ResearchState,
    claims: list[Claim],
) -> list[Evidence]:
    """Evidence cited by verified claims only."""
    wanted = {eid for c in claims for eid in c.evidence_ids}
    return [e for e in (state.get("evidence_items") or []) if e.id in wanted]


def _safe_comparisons(
    state: ResearchState,
    evidence_ids: set[str],
) -> list[TafsirComparison]:
    """Keep comparisons whose cited evidence is within the verified set (when known)."""
    comps = list(state.get("tafsir_comparisons") or [])
    if not evidence_ids:
        return []
    safe: list[TafsirComparison] = []
    for cmp in comps:
        eids = set(cmp.evidence_ids or [])
        if eids and eids.issubset(evidence_ids):
            safe.append(cmp)
        elif eids and eids & evidence_ids:
            # Partial overlap — include but only overlapping ids are "safe"
            safe.append(cmp)
    return safe


def _pack_inputs(state: ResearchState) -> dict[str, Any]:
    claims = _verified_claims(state)
    evidence = _verified_evidence(state, claims)
    evidence_ids = {e.id for e in evidence}
    evidence_by_id = {e.id: e for e in evidence}
    comparisons = _safe_comparisons(state, evidence_ids)
    warnings = list(state.get("warnings") or [])
    result = state.get("verification_result")

    # Fresh formatting pass for this report (avoid cross-run cache collisions)
    citation_manager.clear_cache()
    citations = citation_manager.format_many(evidence)
    citations_by_id = {c.evidence_id: c for c in citations}

    return {
        "user_question": state.get("user_question") or "",
        "language": state.get("language") or "ar",
        "verified_claims": [
            {
                "id": c.id,
                "statement": c.statement,
                "claim_type": str(c.claim_type),
                "evidence_ids": c.evidence_ids,
                "citation_labels": citation_manager.labels_for_ids(
                    c.evidence_ids, evidence_by_id
                ),
                "verification_status": _status_value(c.verification_status),
            }
            for c in claims
        ],
        "verified_evidence": [
            {
                "id": e.id,
                "kind": e.kind,
                "content": e.content,
                # Canonical citation object — report must use citation.label as-is
                "citation": citations_by_id[e.id].model_dump()
                if e.id in citations_by_id
                else citation_manager.format(e).model_dump(),
            }
            for e in evidence
        ],
        "citations": [c.model_dump() for c in citations],
        "tafsir_comparisons": [
            {
                "verse_reference": c.verse_reference,
                "agreements": c.agreements,
                "differences": c.differences,
                "difference_types": c.difference_types,
                "evidence_ids": c.evidence_ids,
                "citation_labels": citation_manager.labels_for_ids(
                    c.evidence_ids, evidence_by_id
                ),
                "summary": c.summary,
            }
            for c in comparisons
        ],
        "research_warnings": warnings[-30:],  # keep prompt bounded
        "verification_summary": {
            "passed": bool(result.passed) if result else False,
            "evidence_coverage": result.evidence_coverage if result else 0.0,
            "unsupported_excluded": list(result.unsupported_claim_ids)
            if result
            else [],
            "conflicting_excluded": list(result.conflicting_claim_ids)
            if result
            else [],
        },
        # Explicitly NOT including unsupported claims as usable evidence
    }


def _deterministic_arabic_report(payload: dict[str, Any]) -> str:
    """Fallback report without LLM — still Arabic and evidence-bound."""
    question = payload["user_question"]
    claims = payload["verified_claims"]
    evidence = payload["verified_evidence"]
    comparisons = payload["tafsir_comparisons"]
    warnings = payload["research_warnings"]
    vsum = payload["verification_summary"]

    lines = [
        "# تقرير باحث القرآن",
        "",
        "## السؤال",
        question or "(غير محدد)",
        "",
    ]

    if not claims and not evidence:
        lines += [
            "## حدود المعرفة",
            "الأدلة المتحققة غير كافية لإعداد إجابة موثَّقة في هذه المرحلة.",
            "لم تُدرج أي دعاوى غير مدعومة أو متعارضة ضمن مواد التقرير.",
            "",
        ]
    else:
        lines.append("## الخلاصة الموثَّقة")
        if not vsum.get("passed") and not claims:
            lines.append(
                "لم تجتزِ عملية التحقق بالكامل؛ يُعرض فقط ما ثبُت من دعاوى وأدلة."
            )
        for c in claims:
            labels = c.get("citation_labels") or c.get("evidence_ids") or []
            lines.append(f"- {c['statement']} {' '.join(labels)}")
        lines.append("")

        lines.append("## الأدلة المعتمدة")
        for e in evidence:
            cite = e.get("citation") or {}
            label = cite.get("label") if isinstance(cite, dict) else str(cite)
            lines.append(f"### {label}")
            lines.append(e["content"][:1200])
            lines.append("")

        if comparisons:
            lines.append("## مقارنة التفاسير (من المواد المورَدة فقط)")
            for cmp in comparisons:
                lines.append(f"### {cmp['verse_reference']}")
                labels = cmp.get("citation_labels") or []
                if labels:
                    lines.append("المصادر: " + " ".join(labels))
                for a in cmp.get("agreements") or []:
                    lines.append(f"- اتفاق: {a}")
                for d in cmp.get("differences") or []:
                    lines.append(f"- اختلاف مدعوم بالدليل: {d}")
                lines.append("")

    if warnings:
        lines.append("## تحذيرات البحث")
        for w in warnings[-15:]:
            lines.append(f"- {w}")
        lines.append("")

    if vsum.get("unsupported_excluded") or vsum.get("conflicting_excluded"):
        lines.append("## ما استُبعد من التقرير")
        lines.append(
            "لم تُستخدم الدعاوى غير المدعومة أو المتعارضة كأدلة قابلة للاستخدام."
        )
        lines.append("")

    lines.append(
        "_أُنشئ هذا التقرير من المواد المتحققة فقط دون إضافة حقائق قرآنية جديدة._"
    )
    return "\n".join(lines)


def _llm_arabic_report(payload: dict[str, Any]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None

    human = (
        "Generate the Arabic report using ONLY the JSON package below. "
        "Do not add anything outside this package.\n\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )
    try:
        llm = get_llm(temperature=0)
        msg = llm.invoke(
            [
                SystemMessage(content=REPORT_SYSTEM),
                HumanMessage(content=human),
            ]
        )
        text = getattr(msg, "content", None) or str(msg)
        if isinstance(text, list):
            # Some providers return content blocks
            parts = []
            for block in text:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            text = "\n".join(parts)
        return str(text).strip() or None
    except Exception:
        return None


def run_report_generation(state: ResearchState) -> dict:
    """Constrained report from verified claims/evidence only (Arabic)."""
    payload = _pack_inputs(state)
    report = _llm_arabic_report(payload)
    if report is None:
        report = _deterministic_arabic_report(payload)

    n_claims = len(payload["verified_claims"])
    n_evidence = len(payload["verified_evidence"])
    return {
        "final_report": report,
        "research_complete": True,
        "warnings": [
            f"report_generator: arabic report from "
            f"{n_claims} verified claim(s), {n_evidence} evidence item(s); "
            f"unsupported/conflicting claims excluded"
        ],
    }
