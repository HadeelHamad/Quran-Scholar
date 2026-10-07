"""Final answer — short Arabic Q&A grounded in collected evidence only."""

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
from quran_scholar.trace import trace, trace_lines

REPORT_SYSTEM = """You answer the user's Quran-related question in Modern Standard Arabic.

Output format — ONLY these two sections (no long research report):

## الإجابة
A direct answer to the user question (a few clear paragraphs or short bullets).
Use ONLY facts from the supplied evidence / claims / comparisons.
Cite with the exact citation.label strings from the package (e.g. [Quran 2:153]).

## الأدلة
List the evidence items that support your answer. For each:
- the citation.label
- a short excerpt or summary of that item (do not invent text)

Rules:
- Do NOT write sections like «تقرير», «الخلاصة الموثَّقة», «حدود المعرفة» as a full report.
- Do NOT invent Quranic text, tafsir, or citations.
- If evidence is insufficient, say so briefly under ## الإجابة and list what little
  evidence exists under ## الأدلة (or say none).
"""


def _status_value(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value)
    return str(status)


def _verified_claims(state: ResearchState) -> list[Claim]:
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
        if _status_value(c.verification_status) in ok_status:
            usable.append(c)
    return usable


def _evidence_for_answer(
    state: ResearchState,
    claims: list[Claim],
) -> list[Evidence]:
    """Prefer claim-linked evidence; otherwise use collected research evidence."""
    all_items = list(state.get("evidence_items") or [])
    if claims:
        wanted = {eid for c in claims for eid in c.evidence_ids}
        linked = [e for e in all_items if e.id in wanted]
        if linked:
            return linked
    # No claims / no links: still allow answering from gathered evidence
    return all_items[:40]


def _safe_comparisons(
    state: ResearchState,
    evidence_ids: set[str],
) -> list[TafsirComparison]:
    comps = list(state.get("tafsir_comparisons") or [])
    if not evidence_ids:
        return comps[:10]
    safe: list[TafsirComparison] = []
    for cmp in comps:
        eids = set(cmp.evidence_ids or [])
        if not eids or eids & evidence_ids:
            safe.append(cmp)
    return safe


def _pack_inputs(state: ResearchState) -> dict[str, Any]:
    claims = _verified_claims(state)
    evidence = _evidence_for_answer(state, claims)
    evidence_ids = {e.id for e in evidence}
    evidence_by_id = {e.id: e for e in evidence}
    comparisons = _safe_comparisons(state, evidence_ids)
    result = state.get("verification_result")

    citation_manager.clear_cache()
    citations = citation_manager.format_many(evidence)
    citations_by_id = {c.evidence_id: c for c in citations}

    verses = []
    for v in state.get("selected_verses") or []:
        verses.append(
            {
                "ref": f"{v.ref.surah}:{v.ref.ayah}",
                "text": v.text_uthmani,
            }
        )

    return {
        "user_question": state.get("user_question") or "",
        "language": state.get("language") or "ar",
        "selected_verses": verses,
        "verified_claims": [
            {
                "id": c.id,
                "statement": c.statement,
                "evidence_ids": c.evidence_ids,
                "citation_labels": citation_manager.labels_for_ids(
                    c.evidence_ids, evidence_by_id
                ),
            }
            for c in claims
        ],
        "evidence": [
            {
                "id": e.id,
                "kind": e.kind,
                "content": (e.content or "")[:2000],
                "citation": citations_by_id[e.id].model_dump()
                if e.id in citations_by_id
                else citation_manager.format(e).model_dump(),
            }
            for e in evidence
        ],
        "tafsir_comparisons": [
            {
                "verse_reference": c.verse_reference,
                "agreements": c.agreements,
                "differences": c.differences,
                "summary": c.summary,
                "citation_labels": citation_manager.labels_for_ids(
                    c.evidence_ids, evidence_by_id
                ),
            }
            for c in comparisons
        ],
        "verification_passed": bool(result.passed) if result else False,
    }


def _deterministic_answer(payload: dict[str, Any]) -> str:
    """Fallback without LLM: answer from claims + list evidence."""
    question = payload["user_question"]
    claims = payload["verified_claims"]
    evidence = payload["evidence"]
    verses = payload.get("selected_verses") or []

    lines = ["## الإجابة", ""]
    if claims:
        for c in claims:
            labels = c.get("citation_labels") or []
            lines.append(f"{c['statement']} {' '.join(labels)}".strip())
            lines.append("")
    elif verses:
        lines.append("بناءً على الآيات المجموعة:")
        for v in verses[:8]:
            lines.append(f"- ({v['ref']}) {v['text']}")
        lines.append("")
    elif evidence:
        lines.append(
            "فيما يلي ملخص موجز مما جُمع من الأدوات (بدون صياغة إضافية):"
        )
        for e in evidence[:5]:
            cite = e.get("citation") or {}
            label = cite.get("label") if isinstance(cite, dict) else ""
            lines.append(f"- {label}: {(e.get('content') or '')[:400]}")
        lines.append("")
    else:
        lines.append(
            "لا تتوفر أدلة كافية من البحث للإجابة عن السؤال في هذه المرحلة."
        )
        lines.append("")

    lines += ["## الأدلة", ""]
    if not evidence:
        lines.append("(لا توجد عناصر أدلة محمّلة في الحالة.)")
    else:
        for e in evidence:
            cite = e.get("citation") or {}
            label = cite.get("label") if isinstance(cite, dict) else str(cite)
            lines.append(f"### {label}")
            lines.append((e.get("content") or "")[:1200])
            lines.append("")
    return "\n".join(lines).strip()


def _llm_answer(payload: dict[str, Any]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None

    human = (
        "أجب عن سؤال المستخدم بالعربية بالصيغة المطلوبة فقط "
        "(## الإجابة ثم ## الأدلة)، اعتمادًا على حزمة JSON التالية فقط:\n\n"
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
    """Produce final Arabic answer + supporting evidence list."""
    t0 = trace(
        "report_generator",
        "Writing answer + evidence...",
        blank_before=True,
    )
    payload = _pack_inputs(state)
    report = _llm_answer(payload)
    if report is None:
        report = _deterministic_answer(payload)

    n_evidence = len(payload["evidence"])
    t1 = trace(
        "report_generator",
        f"Answer ready ({n_evidence} evidence item(s)).",
    )
    return {
        "final_report": report,
        "research_complete": True,
        "warnings": [
            f"report_generator: Q&A from {len(payload['verified_claims'])} "
            f"claim(s), {n_evidence} evidence item(s)"
        ],
        **trace_lines(t0, t1),
    }
