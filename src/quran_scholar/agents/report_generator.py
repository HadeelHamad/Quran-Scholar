"""Final answer — short Arabic Q&A from collected evidence only."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from quran_scholar.agents.llm import get_llm
from quran_scholar.services.citation_manager import citation_manager
from quran_scholar.state import ResearchState
from quran_scholar.trace import trace, trace_lines

REPORT_SYSTEM = """You answer the user's Quran-related question in Modern Standard Arabic.

Output format — ONLY these two sections:

## الإجابة
A direct answer (short paragraphs or bullets).
Use ONLY facts from the supplied evidence / verses / comparisons.
Cite with exact citation.label strings (e.g. [Quran 2:153]).

## الأدلة
List supporting evidence: citation.label + short excerpt (do not invent text).

Rules:
- Do NOT write a long research report.
- Do NOT invent Quranic text, tafsir, or citations.
- If evidence is insufficient, say so under ## الإجابة.
"""


def _pack_inputs(state: ResearchState) -> dict[str, Any]:
    evidence = list(state.get("evidence_items") or [])[:40]
    citation_manager.clear_cache()
    citations = citation_manager.format_many(evidence)
    citations_by_id = {c.evidence_id: c for c in citations}
    evidence_by_id = {e.id: e for e in evidence}

    verses = [
        {"ref": f"{v.ref.surah}:{v.ref.ayah}", "text": v.text_uthmani}
        for v in (state.get("selected_verses") or [])
    ]
    comparisons = []
    for c in state.get("tafsir_comparisons") or []:
        comparisons.append(
            {
                "verse_reference": c.verse_reference,
                "agreements": c.agreements,
                "differences": c.differences,
                "summary": c.summary,
                "citation_labels": citation_manager.labels_for_ids(
                    c.evidence_ids, evidence_by_id
                ),
            }
        )

    return {
        "user_question": state.get("user_question") or "",
        "selected_verses": verses,
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
        "tafsir_comparisons": comparisons,
        "warnings": list(state.get("warnings") or [])[-15:],
    }


def _deterministic_answer(payload: dict[str, Any]) -> str:
    evidence = payload["evidence"]
    verses = payload.get("selected_verses") or []
    lines = ["## الإجابة", ""]
    if verses:
        lines.append("بناءً على الآيات المجموعة:")
        for v in verses[:8]:
            lines.append(f"- ({v['ref']}) {v['text']}")
        lines.append("")
    elif evidence:
        lines.append("ملخص مما جُمع من الأدلة:")
        for e in evidence[:5]:
            cite = e.get("citation") or {}
            label = cite.get("label") if isinstance(cite, dict) else ""
            lines.append(f"- {label}: {(e.get('content') or '')[:400]}")
        lines.append("")
    else:
        lines.append("لا تتوفر أدلة كافية للإجابة عن السؤال في هذه المرحلة.")
        lines.append("")

    lines += ["## الأدلة", ""]
    if not evidence:
        lines.append("(لا توجد عناصر أدلة.)")
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
        "أجب عن سؤال المستخدم بالعربية (## الإجابة ثم ## الأدلة) "
        "اعتمادًا على JSON التالي فقط:\n\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )
    try:
        msg = get_llm(temperature=0).invoke(
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
    t0 = trace(
        "report_generator",
        "Writing answer + evidence...",
        blank_before=True,
    )
    payload = _pack_inputs(state)
    report = _llm_answer(payload) or _deterministic_answer(payload)
    n = len(payload["evidence"])
    t1 = trace("report_generator", f"Answer ready ({n} evidence item(s)).")
    return {
        "final_report": report,
        "research_complete": True,
        "warnings": [f"report_generator: Q&A from {n} evidence item(s)"],
        **trace_lines(t0, t1),
    }
