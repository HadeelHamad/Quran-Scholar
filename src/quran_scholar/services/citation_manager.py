"""Citation manager — format, validate, and dedupe evidence citations."""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

from quran_scholar.models import Evidence, VerseRef

# Display names for Tafsir MCP source ids
TAFSIR_SOURCE_LABELS: dict[str, str] = {
    "tabary": "Tafsir al-Tabari",
    "katheer": "Tafsir Ibn Kathir",
    "baghawy": "Tafsir al-Baghawi",
    "saadi": "Tafsir al-Sa'di",
    "moyassar": "Tafsir al-Muyassar",
    "mukhtasar_ar": "al-Mukhtasar (Arabic)",
    "mukhtasar_en": "Concise Commentary (English)",
    "mukhtasar_bn": "al-Mukhtasar (Bengali)",
    "nuzool": "Asbab al-Nuzool",
}


class Citation(BaseModel):
    """Canonical citation object for reports (do not invent syntax elsewhere)."""

    evidence_id: str
    label: str = Field(description="Bracketed citation string, e.g. [Quran 2:153]")
    kind: str
    verse_reference: str | None = None
    source_id: str | None = None
    source_label: str | None = None
    valid: bool = True
    issues: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return self.label


class CitationManager:
    """Generate and validate citation strings; resolve source metadata; dedupe labels."""

    def __init__(self, source_labels: dict[str, str] | None = None) -> None:
        self._source_labels = dict(TAFSIR_SOURCE_LABELS)
        if source_labels:
            self._source_labels.update(source_labels)
        # Cache: evidence_id → Citation (prevents duplicate formatting work)
        self._by_evidence_id: dict[str, Citation] = {}
        # Cache: normalized label → first evidence_id (dedupe identical labels)
        self._label_owner: dict[str, str] = {}

    def clear_cache(self) -> None:
        self._by_evidence_id.clear()
        self._label_owner.clear()

    def resolve_source_metadata(self, evidence: Evidence) -> dict[str, Any]:
        """Resolve source id / display label / verse ref from evidence + metadata."""
        meta = evidence.metadata or {}
        source_id = meta.get("source_id") or meta.get("source")
        if isinstance(source_id, str):
            source_id = source_id.strip() or None
        else:
            source_id = None

        source_label = None
        if source_id:
            source_label = self._source_labels.get(source_id)
        if not source_label:
            # Prefer explicit attribution fragments from citation/metadata
            for key in ("source_title", "author", "attribution"):
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    source_label = val.strip()
                    break
        if not source_label and evidence.kind == "tafsir" and evidence.citation:
            source_label = self._clean_legacy_citation(evidence.citation)

        verse_reference = self._verse_reference(evidence.refs)
        return {
            "source_id": source_id,
            "source_label": source_label,
            "verse_reference": verse_reference,
            "kind": evidence.kind,
        }

    def validate_evidence_references(self, evidence: Evidence) -> list[str]:
        """Return validation issues (empty list means OK)."""
        issues: list[str] = []
        if not evidence.id:
            issues.append("missing evidence.id")
        if evidence.kind in ("verse", "tafsir", "nuzool") and not evidence.refs:
            issues.append(f"{evidence.kind} evidence missing refs")
        for ref in evidence.refs or []:
            if not (1 <= ref.surah <= 114):
                issues.append(f"invalid surah {ref.surah}")
            if ref.ayah < 1:
                issues.append(f"invalid ayah {ref.ayah}")
        if evidence.kind == "tafsir":
            meta = evidence.metadata or {}
            if not meta.get("source_id") and not evidence.citation:
                issues.append("tafsir evidence missing source_id and citation")
        if not (evidence.content or "").strip():
            issues.append("empty evidence content")
        return issues

    def format(self, evidence: Evidence) -> Citation:
        """
        Format a canonical citation for evidence.

        Examples:
          [Quran 2:153]
          [Tafsir Ibn Kathir — 2:153]
        """
        if evidence.id in self._by_evidence_id:
            return self._by_evidence_id[evidence.id]

        issues = self.validate_evidence_references(evidence)
        meta = self.resolve_source_metadata(evidence)
        kind = evidence.kind
        verse_reference = meta["verse_reference"]
        source_id = meta["source_id"]
        source_label = meta["source_label"]

        if kind == "verse":
            label = (
                f"[Quran {verse_reference}]"
                if verse_reference
                else "[Quran]"
            )
        elif kind == "quran_meta":
            tool = (evidence.metadata or {}).get("source_tool") or "Quran Info"
            label = (
                f"[{tool} — {verse_reference}]"
                if verse_reference
                else f"[{tool}]"
            )
        elif kind == "tafsir":
            name = source_label or source_id or "Tafsir"
            # Normalize common Arabic attribution to short English display when mapped
            if source_id and source_id in self._source_labels:
                name = self._source_labels[source_id]
            label = (
                f"[{name} — {verse_reference}]"
                if verse_reference
                else f"[{name}]"
            )
        elif kind == "nuzool":
            label = (
                f"[Asbab al-Nuzool — {verse_reference}]"
                if verse_reference
                else "[Asbab al-Nuzool]"
            )
        elif kind == "linguistic":
            root = (evidence.metadata or {}).get("root")
            if root and verse_reference:
                label = f"[Linguistic:{root} — {verse_reference}]"
            elif root:
                label = f"[Linguistic:{root}]"
            elif verse_reference:
                label = f"[Linguistic — {verse_reference}]"
            else:
                label = "[Linguistic]"
        else:
            label = f"[{evidence.citation}]" if evidence.citation else f"[{kind}]"

        label = self._dedupe_label(evidence.id, label)

        citation = Citation(
            evidence_id=evidence.id,
            label=label,
            kind=kind,
            verse_reference=verse_reference,
            source_id=source_id if isinstance(source_id, str) else None,
            source_label=source_label if isinstance(source_label, str) else None,
            valid=len(issues) == 0,
            issues=issues,
        )
        self._by_evidence_id[evidence.id] = citation
        return citation

    def format_many(self, evidence_items: Iterable[Evidence]) -> list[Citation]:
        """Format citations for a list of evidence items (deduped by evidence id)."""
        out: list[Citation] = []
        seen: set[str] = set()
        for e in evidence_items:
            if e.id in seen:
                continue
            seen.add(e.id)
            out.append(self.format(e))
        return out

    def labels_for_ids(
        self,
        evidence_ids: list[str],
        evidence_by_id: dict[str, Evidence],
    ) -> list[str]:
        """Resolve citation labels for claim evidence_ids (skip unknown ids)."""
        labels: list[str] = []
        for eid in evidence_ids:
            ev = evidence_by_id.get(eid)
            if ev is None:
                continue
            labels.append(self.format(ev).label)
        return labels

    def _dedupe_label(self, evidence_id: str, label: str) -> str:
        """
        Prevent duplicate citation formatting collisions.

        Same label for the same logical cite is fine; if another evidence_id
        already owns an identical label, suffix with a short id fragment.
        """
        key = label.strip()
        owner = self._label_owner.get(key)
        if owner is None:
            self._label_owner[key] = evidence_id
            return label
        if owner == evidence_id:
            return label
        # Collision: keep stable unique label
        suffix = evidence_id[-6:] if len(evidence_id) >= 6 else evidence_id
        unique = f"{label[:-1]} #{suffix}]" if label.endswith("]") else f"{label} #{suffix}"
        self._label_owner[unique] = evidence_id
        return unique

    @staticmethod
    def _verse_reference(refs: list[VerseRef] | None) -> str | None:
        if not refs:
            return None
        # Prefer primary (first) ref for citation string
        r = refs[0]
        return f"{r.surah}:{r.ayah}"

    @staticmethod
    def _clean_legacy_citation(raw: str) -> str:
        """Strip noisy legacy citation strings down to a short source label."""
        text = raw.strip()
        # Drop trailing "on 2:153" / "— 2:153"
        text = re.sub(r"\s*(on|—|-)\s*\d{1,3}:\d{1,3}\s*$", "", text, flags=re.I)
        # Truncate long attributions
        if len(text) > 80:
            text = text[:77] + "…"
        return text


# Module-level singleton used by agents/report generator
citation_manager = CitationManager()
