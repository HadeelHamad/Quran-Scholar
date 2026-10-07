"""Planner agent — structured ResearchPlan from user question."""

from __future__ import annotations

import os

from quran_scholar.agents.llm import get_llm
from quran_scholar.models import (
    QuestionFocus,
    ResearchPlan,
    ResearchTask,
    TaskStatus,
)
from quran_scholar.question import (
    parse_verse_ref,
    question_mentions_linguistic,
    question_mentions_nuzool,
)

PLANNER_SYSTEM = """You are the Planner for Quran Scholar, a Quranic research system.

Users write their questions in Modern Standard Arabic (الفصحى). Read the Arabic
question carefully; infer verse-specific vs thematic intent from Arabic phrasing
(e.g. آية الكرسي, سورة 2 آية 255, named mufassirin in Arabic).

Your ONLY job is to produce a structured investigation plan (ResearchPlan).
You must NOT answer the user's question, quote the Quran, summarize tafsir,
or give religious rulings.

Identify:
- what the user is asking
- whether the question is verse-specific (surah:ayah) or thematic
- required evidence types (quran_text, tafsir, linguistic, nuzool)
- the MINIMAL set of researcher tasks
- whether tafsir comparison across multiple mufassirin is needed
- whether linguistic (root/word) analysis is useful
- whether sabab al-nuzool (asbab al-nuzool) is relevant

Task kinds (use only these in tasks[].kind):
- fetch_ayah — retrieve a known surah:ayah
- quran_search — thematic FTS search for relevant verses
- tafsir_fetch — fetch tafsir for verse(s)
- linguistic — root/word analysis
- nuzool — asbab al-nuzool / revelation context

Do NOT add tasks named verification or tafsir_comparison; the graph handles those later.
Use needs_tafsir_comparison=true when multiple tafsir sources should be compared.

Execution strategies the Research Manager will choose:
- thematic: quran_search first; then tafsir_fetch / linguistic / nuzool may run in
  PARALLEL — give them depends_on=[quran task id] only (not each other).
- verse_specific: fetch_ayah → tafsir_fetch → optional linguistic → optional nuzool
  (sequential depends_on chain).
- tafsir_comparison: fetch_ayah → tafsir_fetch only (comparison is a later graph stage).

Keep tasks minimal:
- Verse-specific + one named mufassir (e.g. Ibn Kathir) → fetch_ayah then tafsir_fetch only;
  set target_tafsir_sources to that source (katheer for Ibn Kathir).
- Broad thematic question → quran_search, then parallelizable tafsir/linguistic/nuzool.

Each task needs a unique id (e.g. t1_fetch_ayah), description, kind, and depends_on where order matters.
Write question_summary, approach, and task descriptions in Arabic when language is ar; keep task kind ids in English.
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
    """Deterministic plan when LLM is unavailable (no API key or call failure)."""
    verse = parse_verse_ref(user_question)
    named_sources = _detect_tafsir_sources(user_question)
    ar = language == "ar"

    if verse and named_sources:
        return ResearchPlan(
            question_summary=user_question[:240],
            question_focus=QuestionFocus.VERSE_SPECIFIC,
            required_evidence_types=["quran_text", "tafsir"],
            needs_tafsir_comparison=len(named_sources) > 1,
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
                    description=f"Fetch {verse.surah}:{verse.ayah} (Uthmani text)",
                    kind="fetch_ayah",
                ),
                ResearchTask(
                    id="t2_tafsir_fetch",
                    description="Fetch tafsir for the verse from requested source(s)",
                    kind="tafsir_fetch",
                    depends_on=["t1_fetch_ayah"],
                ),
            ],
        )

    if verse:
        needs_ling = question_mentions_linguistic(user_question)
        needs_nuzool = question_mentions_nuzool(user_question)
        tasks = [
            ResearchTask(
                id="t1_fetch_ayah",
                description=f"Fetch {verse.surah}:{verse.ayah}",
                kind="fetch_ayah",
            ),
            ResearchTask(
                id="t2_tafsir_fetch",
                description="Fetch tafsir for the verse",
                kind="tafsir_fetch",
                depends_on=["t1_fetch_ayah"],
            ),
        ]
        # Sequential optional steps (verse_specific pattern — not parallel with tafsir)
        if needs_ling:
            tasks.append(
                ResearchTask(
                    id="t3_linguistic",
                    description="Optional linguistic analysis of key terms",
                    kind="linguistic",
                    depends_on=["t2_tafsir_fetch"],
                )
            )
        if needs_nuzool:
            prev = "t3_linguistic" if needs_ling else "t2_tafsir_fetch"
            tasks.append(
                ResearchTask(
                    id="t4_nuzool",
                    description="Optional sabab al-nuzool for the verse",
                    kind="nuzool",
                    depends_on=[prev],
                )
            )
        evidence = ["quran_text", "tafsir"]
        if needs_ling:
            evidence.append("linguistic")
        if needs_nuzool:
            evidence.append("nuzool")
        # Comparison path when no optional supporting research
        needs_compare = not needs_ling and not needs_nuzool
        if needs_compare:
            evidence.append("tafsir_comparison")
        return ResearchPlan(
            question_summary=user_question[:240],
            question_focus=QuestionFocus.VERSE_SPECIFIC,
            required_evidence_types=evidence,
            needs_tafsir_comparison=needs_compare,
            needs_linguistic_analysis=needs_ling,
            needs_sabab_nuzool=needs_nuzool,
            approach=(
                "Verse-specific: fetch ayah → tafsir"
                + (" → linguistic" if needs_ling else "")
                + (" → nuzool" if needs_nuzool else "")
                + (" → comparison" if needs_compare else "")
                + "."
            ),
            primary_verse=verse,
            tasks=tasks,
        )

    # Thematic default — after Quran, tafsir/linguistic/nuzool share depends_on
    # so the Research Manager can fan them out in parallel.
    tasks = [
        ResearchTask(
            id="t1_quran_search",
            description="Search Quran for verses relevant to the theme",
            kind="quran_search",
        ),
        ResearchTask(
            id="t2_tafsir_fetch",
            description="Fetch tafsir for selected verses",
            kind="tafsir_fetch",
            depends_on=["t1_quran_search"],
        ),
        ResearchTask(
            id="t3_linguistic",
            description="Analyze key roots/words for the theme (e.g. relevant Arabic roots)",
            kind="linguistic",
            depends_on=["t1_quran_search"],
        ),
    ]
    needs_linguistic = True
    needs_nuzool = question_mentions_nuzool(user_question)
    if needs_nuzool:
        tasks.append(
            ResearchTask(
                id="t4_nuzool",
                description="Gather sabab al-nuzool where relevant",
                kind="nuzool",
                depends_on=["t1_quran_search"],
            )
        )

    evidence = ["quran_text", "tafsir"]
    if needs_linguistic:
        evidence.append("linguistic")
    if needs_nuzool:
        evidence.append("nuzool")
    evidence.append("tafsir_comparison")

    return ResearchPlan(
        question_summary=user_question[:240],
        question_focus=QuestionFocus.THEMATIC,
        required_evidence_types=evidence,
        needs_tafsir_comparison=True,
        needs_linguistic_analysis=needs_linguistic,
        needs_sabab_nuzool=needs_nuzool,
        approach=(
            "Thematic: Quran search first; then parallel tafsir ∥ linguistic"
            + (" ∥ nuzool" if needs_nuzool else "")
            + " once verses are selected."
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
        "Output a minimal ResearchPlan only."
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
