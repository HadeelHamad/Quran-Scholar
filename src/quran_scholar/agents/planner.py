"""Planner — structured ResearchPlan for any Quran-related question (no tools)."""

from __future__ import annotations

import os

from quran_scholar.agents.llm import get_llm
from quran_scholar.models import (
    QuestionFocus,
    ResearchPlan,
    ResearchTask,
)
from quran_scholar.question import (
    parse_verse_ref,
    question_mentions_comparison,
    question_mentions_linguistic,
    question_mentions_meta,
    question_mentions_nuzool,
    question_mentions_tafsir,
)

PLANNER_SYSTEM = """You are the Planner for Quran Scholar — a multi-agent system that
answers ANY Quran-related question using verified MCP tools (not only tafsir).

Users write in Modern Standard Arabic (الفصحى). Questions may be about:
- Quranic text / themes / verse lookup
- classical tafsir or comparison of mufassirin
- linguistic analysis (root, meaning, iʿrāb)
- asbab al-nuzool
- surah info, statistics, overview, qira'at, page benefits
or mixtures of the above.

Your ONLY job is a structured ResearchPlan. Do NOT answer the question,
quote the Quran, summarize tafsir, or give religious rulings.

Identify:
- what the user is asking
- verse-specific vs thematic vs meta (stats/overview/qira'at)
- which evidence types are ACTUALLY needed (do not add tafsir by default)
- the MINIMAL researcher tasks

Evidence types you may list: quran_text, tafsir, linguistic, nuzool,
tafsir_comparison, surah_info, qiraat, statistics

Task kinds (tasks[].kind — English ids only):
- fetch_ayah — known surah:ayah
- quran_search — thematic FTS / overview / surah tools as needed
- tafsir_fetch — only when interpretation/commentary is needed
- linguistic — root/word analysis when needed
- nuzool — asbab al-nuzool when needed

Do NOT add tasks named verification or tafsir_comparison; the graph handles those.
Set needs_tafsir_comparison=true ONLY when the user wants multi-mufassir comparison
or clearly needs contrasting commentaries.

Execution strategies:
- thematic: quran_search first; then ONLY the parallel follow-ups that are needed
  (tafsir / linguistic / nuzool) with depends_on=[quran task id]
- verse_specific: fetch_ayah first; add tafsir/linguistic/nuzool only if needed
- tafsir_comparison: fetch_ayah → tafsir_fetch when comparison is the goal

Keep plans lean:
- "كم عدد آيات سورة البقرة؟" / overview / qira'at → mostly quran_search (meta tools)
- "ما تفسير …؟" → include tafsir_fetch
- "ما جذر كلمة …؟" → linguistic, maybe fetch_ayah
- thematic theme without asking for tafsir → quran_search (+ linguistic if wording/roots matter)

Write question_summary, approach, and task descriptions in Arabic when language is ar.
"""


def _detect_tafsir_sources(question: str) -> list[str]:
    q = question.lower()
    sources: list[str] = []
    if "ibn kathir" in q or "ibn katheer" in q or "ابن كثير" in question:
        sources.append("katheer")
    if "saadi" in q or "sa'di" in q or "السعدي" in question:
        sources.append("saadi")
    if "tabari" in q or "الطبري" in question:
        sources.append("tabary")
    if "muyassar" in q or "moyassar" in q or "الميسر" in question:
        sources.append("moyassar")
    if "baghawy" in q or "البغوي" in question:
        sources.append("baghawy")
    return sources


def _heuristic_plan(user_question: str, language: str) -> ResearchPlan:
    """Deterministic plan when LLM is unavailable."""
    verse = parse_verse_ref(user_question)
    named_sources = _detect_tafsir_sources(user_question)
    ar = language == "ar"
    needs_tafsir = bool(named_sources) or question_mentions_tafsir(user_question)
    needs_ling = question_mentions_linguistic(user_question)
    needs_nuzool = question_mentions_nuzool(user_question)
    needs_compare = question_mentions_comparison(user_question) or (
        len(named_sources) > 1
    )
    needs_meta = question_mentions_meta(user_question)

    # Named mufassir + verse → focused tafsir fetch
    if verse and named_sources:
        return ResearchPlan(
            question_summary=user_question[:240],
            question_focus=QuestionFocus.VERSE_SPECIFIC,
            required_evidence_types=["quran_text", "tafsir"],
            needs_tafsir_comparison=needs_compare,
            needs_linguistic_analysis=False,
            needs_sabab_nuzool=False,
            approach=(
                "جلب الآية ثم التفسير من المصدر المطلوب فقط."
                if ar
                else "Fetch the cited verse then tafsir from the requested source(s) only."
            ),
            primary_verse=verse,
            target_tafsir_sources=named_sources,
            tasks=[
                ResearchTask(
                    id="t1_fetch_ayah",
                    description=f"جلب الآية {verse.surah}:{verse.ayah}",
                    kind="fetch_ayah",
                ),
                ResearchTask(
                    id="t2_tafsir_fetch",
                    description="جلب التفسير من المصادر المطلوبة",
                    kind="tafsir_fetch",
                    depends_on=["t1_fetch_ayah"],
                ),
            ],
        )

    if verse:
        # Pure verse lookup / meta about a verse: do not force tafsir
        if not needs_tafsir and not needs_ling and not needs_nuzool:
            evidence = ["quran_text"]
            if needs_meta:
                evidence.append("surah_info")
            return ResearchPlan(
                question_summary=user_question[:240],
                question_focus=QuestionFocus.VERSE_SPECIFIC,
                required_evidence_types=evidence,
                needs_tafsir_comparison=False,
                needs_linguistic_analysis=False,
                needs_sabab_nuzool=False,
                approach=(
                    "جلب نص الآية وما يلزم من معلومات السورة/القراءات إن وُجدت."
                    if ar
                    else "Fetch the ayah text and any needed surah/qira'at info."
                ),
                primary_verse=verse,
                tasks=[
                    ResearchTask(
                        id="t1_fetch_ayah",
                        description=f"جلب الآية {verse.surah}:{verse.ayah}",
                        kind="fetch_ayah",
                    ),
                ],
            )

        tasks = [
            ResearchTask(
                id="t1_fetch_ayah",
                description=f"جلب الآية {verse.surah}:{verse.ayah}",
                kind="fetch_ayah",
            ),
        ]
        evidence = ["quran_text"]
        prev = "t1_fetch_ayah"
        if needs_tafsir:
            tasks.append(
                ResearchTask(
                    id="t2_tafsir_fetch",
                    description="جلب التفسير للآية",
                    kind="tafsir_fetch",
                    depends_on=[prev],
                )
            )
            evidence.append("tafsir")
            prev = "t2_tafsir_fetch"
        if needs_ling:
            tid = "t3_linguistic"
            tasks.append(
                ResearchTask(
                    id=tid,
                    description="تحليل لغوي للمفردات ذات الصلة",
                    kind="linguistic",
                    depends_on=[prev],
                )
            )
            evidence.append("linguistic")
            prev = tid
        if needs_nuzool:
            tasks.append(
                ResearchTask(
                    id="t4_nuzool",
                    description="جلب سبب النزول إن وُجد",
                    kind="nuzool",
                    depends_on=[prev],
                )
            )
            evidence.append("nuzool")
        if needs_compare:
            evidence.append("tafsir_comparison")
        return ResearchPlan(
            question_summary=user_question[:240],
            question_focus=QuestionFocus.VERSE_SPECIFIC,
            required_evidence_types=evidence,
            needs_tafsir_comparison=needs_compare,
            needs_linguistic_analysis=needs_ling,
            needs_sabab_nuzool=needs_nuzool,
            approach="آية محددة: جلب النص ثم ما يلزم فقط من تفسير/لغة/نزول.",
            primary_verse=verse,
            tasks=tasks,
        )

    # Thematic / meta (no explicit verse)
    tasks = [
        ResearchTask(
            id="t1_quran_search",
            description=(
                "بحث في القرآن وجمع الآيات أو معلومات السورة ذات الصلة"
                if ar
                else "Search Quran / gather relevant verses or surah info"
            ),
            kind="quran_search",
        ),
    ]
    evidence = ["quran_text"]
    if needs_meta:
        evidence.extend(["surah_info", "statistics"])

    # Only attach follow-ups the question actually asks for
    if needs_tafsir or needs_compare:
        tasks.append(
            ResearchTask(
                id="t2_tafsir_fetch",
                description="جلب التفسير للآيات المختارة",
                kind="tafsir_fetch",
                depends_on=["t1_quran_search"],
            )
        )
        evidence.append("tafsir")
    if needs_ling:
        tasks.append(
            ResearchTask(
                id="t3_linguistic",
                description="تحليل لغوي للجذور/الكلمات ذات الصلة",
                kind="linguistic",
                depends_on=["t1_quran_search"],
            )
        )
        evidence.append("linguistic")
    if needs_nuzool:
        tasks.append(
            ResearchTask(
                id="t4_nuzool",
                description="جمع أسباب النزول حيث يلزم",
                kind="nuzool",
                depends_on=["t1_quran_search"],
            )
        )
        evidence.append("nuzool")
    if needs_compare:
        evidence.append("tafsir_comparison")

    return ResearchPlan(
        question_summary=user_question[:240],
        question_focus=QuestionFocus.THEMATIC,
        required_evidence_types=evidence,
        needs_tafsir_comparison=needs_compare,
        needs_linguistic_analysis=needs_ling,
        needs_sabab_nuzool=needs_nuzool,
        approach=(
            "موضوعي: بحث قرآني أولاً، ثم فقط المسارات اللازمة (تفسير/لغة/نزول)."
            if ar
            else "Thematic: Quran research first; only needed follow-ups."
        ),
        tasks=tasks,
    )


def _normalize_task_kinds(plan: ResearchPlan) -> ResearchPlan:
    """Map planner aliases to researcher routing kinds."""
    alias = {
        "tafsir_research": "tafsir_fetch",
        "fetch_tafsir": "tafsir_fetch",
        "verse_search": "quran_search",
        "quran": "quran_search",
        "linguistic_analysis": "linguistic",
        "nuzool_research": "nuzool",
        "context": "nuzool",
        "surah_info": "quran_search",
        "qiraat": "quran_search",
        "statistics": "quran_search",
    }
    tasks = []
    for t in plan.tasks:
        kind = alias.get(t.kind, t.kind)
        if kind in ("verification", "tafsir_comparison"):
            continue
        tasks.append(t.model_copy(update={"kind": kind}))
    return plan.model_copy(update={"tasks": tasks})


def plan_research(user_question: str, language: str = "ar") -> ResearchPlan:
    """Build ResearchPlan via structured LLM, with heuristic fallback."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return _normalize_task_kinds(_heuristic_plan(user_question, language))

    llm = get_llm()
    structured_llm = llm.with_structured_output(ResearchPlan)
    human = (
        f"Language: {language} (user question is in Arabic when language is ar)\n"
        f"User question:\n{user_question}\n\n"
        "Output a minimal ResearchPlan for this Quran-related question. "
        "Do not require tafsir unless the question needs interpretation/commentary."
    )
    try:
        plan = structured_llm.invoke(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": human},
            ]
        )
        if not isinstance(plan, ResearchPlan):
            plan = ResearchPlan.model_validate(plan)
        return _normalize_task_kinds(plan)
    except Exception:
        return _normalize_task_kinds(_heuristic_plan(user_question, language))
