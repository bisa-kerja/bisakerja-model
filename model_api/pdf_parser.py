"""Deterministic request-scoped PDF parsing helpers for Model API.

This module intentionally avoids OCR and GenAI. It extracts only text that is
present in PDF text operators, records parser evidence, and returns safe fallback
signals when bytes are empty, scanned, malformed, or too large.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .features import normalized_skill_set

PDF_MAGIC = b"%PDF"
SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summary", re.compile(r"\b(summary|profile|objective|ringkasan|profil)\b", re.I)),
    ("experience", re.compile(r"\b(experience|employment|work history|pengalaman)\b", re.I)),
    ("education", re.compile(r"\b(education|pendidikan)\b", re.I)),
    ("skills", re.compile(r"\b(skills|technical skills|keahlian|kompetensi)\b", re.I)),
    ("projects", re.compile(r"\b(projects|portfolio|proyek)\b", re.I)),
    ("certifications", re.compile(r"\b(certifications?|sertifikasi)\b", re.I)),
)
CONTACT_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|(?:\+?62|0)\d[\d\s\-()]{7,}", re.I)
DATE_RE = re.compile(r"\b(?:20\d{2}|19\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\b", re.I)
TEXT_LITERAL_RE = re.compile(rb"\((?:\\.|[^\\()])*\)\s*T[Jj]")
TEXT_ARRAY_RE = re.compile(rb"\[(.*?)\]\s*TJ", re.S)
ARRAY_LITERAL_RE = re.compile(rb"\((?:\\.|[^\\()])*\)")


@dataclass(frozen=True)
class ParsedPdfEvidence:
    text: str
    page_count: int
    section_names: tuple[str, ...]
    has_contact_signal: bool
    has_date_signal: bool
    formatting_risks: tuple[str, ...]
    parse_quality: str
    detected_issues: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return bool(self.text.strip()) and self.parse_quality in {"good", "partial"}


def _decode_pdf_literal(value: bytes) -> str:
    inner = value[1:-1]
    replacements = {
        rb"\n": b"\n",
        rb"\r": b"\n",
        rb"\t": b"\t",
        rb"\(": b"(",
        rb"\)": b")",
        rb"\\": b"\\",
    }
    for source, target in replacements.items():
        inner = inner.replace(source, target)
    return inner.decode("utf-8", errors="ignore")


def _extract_text_literals(pdf_bytes: bytes) -> str:
    parts: list[str] = []
    for match in TEXT_LITERAL_RE.finditer(pdf_bytes):
        literal = match.group(0).rsplit(b")", 1)[0] + b")"
        parts.append(_decode_pdf_literal(literal))
    for array_match in TEXT_ARRAY_RE.finditer(pdf_bytes):
        literals = ARRAY_LITERAL_RE.findall(array_match.group(1))
        if literals:
            parts.append("".join(_decode_pdf_literal(literal) for literal in literals))
    text = "\n".join(part.strip() for part in parts if part.strip())
    return re.sub(r"[ \t]+", " ", text).strip()


def _page_count(pdf_bytes: bytes) -> int:
    pages = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    if pages:
        return pages
    fallback = len(re.findall(rb"/Page\b", pdf_bytes))
    return max(1, fallback) if pdf_bytes.startswith(PDF_MAGIC) else 0


def _section_names(text: str) -> tuple[str, ...]:
    found = [name for name, pattern in SECTION_PATTERNS if pattern.search(text)]
    return tuple(dict.fromkeys(found))


def parse_pdf_bytes(pdf_bytes: bytes, *, max_bytes: int, max_pages: int) -> ParsedPdfEvidence:
    """Parse PDF bytes into deterministic evidence without OCR/hallucination."""

    issues: list[str] = []
    risks: list[str] = []
    if not pdf_bytes:
        return ParsedPdfEvidence("", 0, (), False, False, (), "failed", ("empty PDF upload",))
    if len(pdf_bytes) > max_bytes:
        return ParsedPdfEvidence("", 0, (), False, False, (), "failed", ("PDF exceeds configured byte limit",))
    if not pdf_bytes.lstrip().startswith(PDF_MAGIC):
        return ParsedPdfEvidence("", 0, (), False, False, (), "failed", ("file is not a PDF",))

    page_count = _page_count(pdf_bytes)
    if page_count > max_pages:
        return ParsedPdfEvidence("", page_count, (), False, False, (), "failed", ("PDF exceeds configured page limit",))

    if re.search(rb"/XObject|/Image", pdf_bytes):
        risks.append("image content present")
    if re.search(rb"/Columns\s+\d", pdf_bytes):
        risks.append("multi-column layout marker present")

    text = _extract_text_literals(pdf_bytes)
    sections = _section_names(text)
    has_contact = bool(CONTACT_RE.search(text))
    has_date = bool(DATE_RE.search(text))
    if not text.strip():
        issues.append("no extractable PDF text; scanned or image-only CV suspected")
    if not sections:
        issues.append("standard CV sections not detected")
    if not has_contact:
        issues.append("contact signal not detected")
    if not has_date:
        issues.append("date or timeline signal not detected")
    if risks:
        issues.extend(risks)

    if not text.strip():
        quality = "empty"
    elif issues:
        quality = "partial"
    else:
        quality = "good"
    return ParsedPdfEvidence(text, page_count, sections, has_contact, has_date, tuple(risks), quality, tuple(issues))


def normalized_skills_from_text(text: str, candidate_skill_hints: Sequence[str] = ()) -> tuple[str, ...]:
    """Extract skill hints by matching known candidate skills against parsed CV text."""

    text_lower = text.lower()
    matched = [skill for skill in normalized_skill_set(candidate_skill_hints) if skill and skill.lower() in text_lower]
    return tuple(sorted(set(matched)))


def ats_score_from_pdf_evidence(evidence: ParsedPdfEvidence) -> tuple[int, tuple[str, ...], bool]:
    """Transparent ATS score from parser evidence."""

    if evidence.parse_quality == "failed":
        return 0, evidence.detected_issues, True
    score = 100
    penalties = {
        "empty": 70,
        "partial": 15,
        "good": 0,
    }
    score -= penalties.get(evidence.parse_quality, 30)
    if not evidence.section_names:
        score -= 15
    if not evidence.has_contact_signal:
        score -= 10
    if not evidence.has_date_signal:
        score -= 10
    score -= min(20, len(evidence.formatting_risks) * 10)
    return max(0, min(100, int(score))), evidence.detected_issues, evidence.parse_quality != "good"
