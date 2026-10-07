"""Final answer — short Arabic Q&A from collected evidence only."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from quran_scholar.graph.nodes.llm import get_llm
from quran_scholar.state import ResearchState

REPORT_SYSTEM = """You answer the user's Quran-related question in Modern Standard Arabic.

Output format — ONLY these two sections:

## الإجابة
A direct answer (short paragraphs or bullets).
Use ONLY facts from the supplied evidence / verses / comparisons.
Cite with exact citation strings (e.g. [Quran 2:153]).

## الأدلة
List supporting evidence: citation + short excerpt (do not invent text).

Rules:
- Do NOT write a long research report.
- Do NOT invent Quranic text, tafsir, or citations.
- If evidence is insufficient, say so under ## الإجابة.
"""


def _pack_inputs(state: ResearchState) -> dict[str, Any]:
    evidence = list(state.get("evidence_items") or [])[:40]
    evidence_by_id = {e.id: e for e in evidence}

    verses = [
        {"ref": f"{v.ref.surah}:{v.ref.ayah}", "text": v.text_uthmani}
        for v in (state.get("selected_verses") or [])
    ]
    comparisons = []
    for c in state.get("tafsir_comparisons") or []:
        labels = [
            evidence_by_id[eid].citation
            for eid in c.evidence_ids
            if eid in evidence_by_id
        ]
        comparisons.append(
            {
                "verse_reference": c.verse_reference,
                "agreements": c.agreements,
                "differences": c.differences,
                "summary": c.summary,
                "citation_labels": labels,
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
                "citation": e.citation,
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
            label = e.get("citation") or e.get("kind") or ""
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
            label = e.get("citation") or e.get("kind") or ""
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
    payload = _pack_inputs(state)
    report = _llm_answer(payload) or _deterministic_answer(payload)
    n = len(payload["evidence"])
    return {
        "final_report": report,
        "research_complete": True,
        "warnings": [f"report_generator: Q&A from {n} evidence item(s)"],
    }


def report_generator_node(state: ResearchState) -> dict:
    return run_report_generation(state)
