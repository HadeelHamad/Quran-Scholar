"""Planner — structured ResearchPlan for any Quran-related question (no tools)."""

from __future__ import annotations

import re

from quran_scholar.graph.nodes.llm import get_llm
from quran_scholar.models import ResearchPlan, VerseRef
from quran_scholar.state import ResearchState

# --- question parsing helpers (used by Quran researcher) ---

_AR_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_VERSE_ALIASES: tuple[tuple[str, VerseRef], ...] = (
    ("آية الكرسي", VerseRef(surah=2, ayah=255)),
    ("اية الكرسي", VerseRef(surah=2, ayah=255)),
    ("آيةُ الكرسي", VerseRef(surah=2, ayah=255)),
    ("سورة الفاتحة", VerseRef(surah=1, ayah=1)),
)

_COLON_REF = re.compile(r"(\d{1,3})\s*:\s*(\d{1,3})")
_SURAH_AYAH_AR = re.compile(
    r"(?:سورة|سوره)\s*(\d{1,3})\s*(?:،|,|\s|-)+"
    r"(?:ال)?(?:آية|آيه|اية|الآية)\s*(\d{1,3})",
    re.IGNORECASE,
)
_SURAH_AYAH_EN = re.compile(
    r"(?:surah|sura)\s*(\d{1,3})\s*(?:,|\s|-)+"
    r"(?:ayah|aya|verse)\s*(\d{1,3})",
    re.IGNORECASE,
)


def normalize_question_text(question: str) -> str:
    """Normalize digits and strip outer whitespace (questions are Arabic)."""
    return (question or "").strip().translate(_AR_DIGIT_MAP)


def parse_verse_ref(question: str) -> VerseRef | None:
    """Detect surah:ayah or common Arabic verse names in the user question."""
    text = normalize_question_text(question)
    if not text:
        return None

    for phrase, ref in _VERSE_ALIASES:
        if phrase in text:
            return ref

    for pattern in (_COLON_REF, _SURAH_AYAH_AR, _SURAH_AYAH_EN):
        match = pattern.search(text)
        if match:
            surah, ayah = int(match.group(1)), int(match.group(2))
            if 1 <= surah <= 114 and ayah >= 1:
                return VerseRef(surah=surah, ayah=ayah)
    return None


PLANNER_SYSTEM = """You are the Planner for Quran Scholar — a multi-agent system that
answers ANY Quran-related question using verified MCP tools (not only tafsir).

Users write in Modern Standard Arabic (الفصحى). Questions may be about:
- Quranic text / themes / verse lookup
- classical tafsir or comparison of mufassirin
- linguistic analysis (root, meaning, iʿrāb)
- asbab al-nuzool
- surah info, statistics, overview
or mixtures of the above.

Your ONLY job is a structured ResearchPlan. Do NOT answer the question,
quote the Quran, summarize tafsir, or give religious rulings.

Identify:
- what the user is asking
- which evidence types are ACTUALLY needed (do not add tafsir by default)
- the MINIMAL researcher tasks and their depends_on

Evidence types you may list: quran_text, tafsir, linguistic, nuzool,
tafsir_comparison, surah_info, statistics

Task kinds (tasks[].kind — English ids only):
- fetch_ayah — known surah:ayah
- quran_search — thematic FTS / overview / surah tools as needed
- tafsir_fetch — only when interpretation/commentary is needed
- linguistic — root/word analysis when needed
- nuzool — asbab al-nuzool when needed

Do NOT add tasks named verification or tafsir_comparison; the graph handles those.
Set needs_tafsir_comparison=true ONLY when the user wants multi-mufassir comparison
or clearly needs contrasting commentaries.

Task dependencies:
- Known verse: fetch_ayah first; add tafsir/linguistic/nuzool only if needed
  (depends_on=[fetch task id])
- Theme/search: quran_search first; follow-ups that need verses depend on it
- Ready follow-ups with the same depends_on can run in parallel

Keep plans lean:
- "كم عدد آيات سورة البقرة؟" / overview → mostly quran_search (meta tools)
- "ما تفسير …؟" → include tafsir_fetch
- "ما جذر كلمة …؟" → linguistic, maybe fetch_ayah
- theme without asking for tafsir → quran_search (+ linguistic if wording/roots matter)

Write question_summary, approach, and task descriptions in Arabic when language is ar.
"""


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
    """Build ResearchPlan via structured LLM only."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(ResearchPlan)
    human = (
        f"Language: {language} (user question is in Arabic when language is ar)\n"
        f"User question:\n{user_question}\n\n"
        "Output a minimal ResearchPlan for this Quran-related question. "
        "Do not require tafsir unless the question needs interpretation/commentary."
    )
    plan = structured_llm.invoke(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": human},
        ]
    )
    if not isinstance(plan, ResearchPlan):
        plan = ResearchPlan.model_validate(plan)
    return _normalize_task_kinds(plan)


def planner_node(state: ResearchState) -> dict:
    """Convert user_question into research_plan (no answer content)."""
    question = (state.get("user_question") or "").strip()
    language = state.get("language") or "ar"

    if not question:
        return {
            "errors": ["planner: empty user_question"],
            "research_iteration": 0,
        }

    try:
        plan = plan_research(question, language)
    except Exception as exc:
        return {
            "errors": [f"planner: LLM plan failed: {exc}"],
            "research_iteration": 0,
        }

    return {
        "research_plan": plan,
        "research_iteration": 0,
    }


