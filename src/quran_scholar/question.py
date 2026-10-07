"""Parse user research questions (Arabic-first)."""

from __future__ import annotations

import re

from quran_scholar.models import VerseRef

# Eastern Arabic-Indic digits → ASCII
_AR_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Well-known Arabic references without surah:ayah in the text
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


def question_mentions_linguistic(question: str) -> bool:
    q = normalize_question_text(question)
    return any(
        term in q
        for term in (
            "معنى",
            "معني",
            "جذر",
            "لغة",
            "لغوي",
            "كلمة",
            "اللفظ",
            "اشتقاق",
        )
    )


def question_mentions_nuzool(question: str) -> bool:
    q = normalize_question_text(question)
    return any(
        term in q for term in ("سبب", "نزول", "أسباب", "اسباب", "نزلت", "نزولها")
    )
