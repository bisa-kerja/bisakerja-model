"""Deterministic request-scoped PDF parsing helpers for Model API.

This module intentionally avoids OCR and GenAI. It extracts only text that is
present in PDF text operators, records parser evidence, and returns safe fallback
signals when bytes are empty, scanned, malformed, or too large.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence
import zlib

from .features import SKILL_ALIASES, normalize_skill_token

PDF_MAGIC = b"%PDF"
SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summary", re.compile(r"\b(summary|profile|objective|about\s*me|professional\s*summary|ringkasan|profil|tentang\s*saya|deskripsi\s*diri)\b", re.I)),
    ("experience", re.compile(r"\b(experience|employment|work\s*history|work\s*experience|professional\s*experience|pengalaman|riwayat\s*pekerjaan|pengalaman\s*kerja)\b", re.I)),
    ("education", re.compile(r"\b(education|academic|formal\s*education|pendidikan|riwayat\s*pendidikan)\b", re.I)),
    ("skills", re.compile(r"\b(skills|technical\s*skills|core\s*competencies|keahlian|kompetensi|kemampuan|keterampilan)\b", re.I)),
    ("projects", re.compile(r"\b(projects|portfolio|selected\s*work|proyek|portofolio)\b", re.I)),
    ("certifications", re.compile(r"\b(certifications?|licenses?|sertifikasi|lisensi)\b", re.I)),
)
SECTION_COMPACT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "profile", "professionalsummary", "tentangsaya", "deskripsidiri"),
    "experience": ("experience", "workexperience", "professionalexperience", "pengalamankerja", "riwayatpekerjaan"),
    "education": ("education", "formaleducation", "riwayatpendidikan"),
    "skills": ("skills", "technicalskills", "corecompetencies", "keahlian", "kompetensi", "keterampilan"),
    "projects": ("projects", "portfolio", "selectedwork", "portofolio"),
    "certifications": ("certification", "certifications", "licenses", "sertifikasi"),
}
CONTACT_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|(?:\+?62|0)\d[\d\s\-()]{7,}", re.I)
SPACED_CONTACT_RE = re.compile(r"[A-Z0-9._%+-]{3,}\s*@\s*[A-Z0-9.-]+\s*\.\s*[A-Z]{2,}|(?:\+?\s*62|0)(?:[\s\-()_]*\d){8,}", re.I)
DATE_RE = re.compile(r"\b(?:20\d{2}|19\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\b", re.I)
EXPLICIT_EXPERIENCE_RE = re.compile(r"\b(\d{1,2}(?:[.,]\d)?)\+?\s*(?:years?|yrs?|tahun)\s*(?:of\s+)?(?:experience|pengalaman)?\b", re.I)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
QUANTIFIED_IMPACT_RE = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*(?:%|percent|persen|x|k|m|rb|juta|million|billion|users?|pengguna|customers?|clients?|projects?|tickets?|requests?|orders?|transactions?|revenue|sales|cost|latency|ms|seconds?|minutes?|hours?|days?)\b|\b(?:reduced|improved|increased|decreased|optimized|saved|meningkatkan|mengurangi|menghemat|mempercepat)\b)",
    re.I,
)
ROLE_TITLE_RE = re.compile(
    r"\b(?:(?:senior|sr\.?|lead|principal|staff|junior|jr\.?|mid(?:-level)?)\s+)?(?:backend|front[- ]?end|full[- ]?stack|mobile|android|ios|data|machine learning|ml|devops|site reliability|sre|product|project|qa|quality assurance|software|cloud|security|web)\s+(?:engineer|developer|scientist|analyst|manager|specialist|architect|consultant|lead|intern|instructor|coordinator)\b|\b(?:assistant\s+lecturer|lab\s+assistant)\b",
    re.I,
)
COMPACT_ROLE_TITLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("fullstackdeveloperintern", "Full Stack Developer Intern"),
    ("fullstackwebdeveloper", "Full Stack Web Developer"),
    ("backendwebprogramminginstructorcoordinator", "Backend Web Programming Instructor Coordinator"),
    ("backendwebprogramminginstructor", "Backend Web Programming Instructor"),
    ("assistantlecturer", "Assistant Lecturer"),
    ("labassistant", "Lab Assistant"),
    ("softwareengineer", "Software Engineer"),
    ("softwareengineering", "Software Engineering"),
)
COMPANY_RE = re.compile(r"\b(?:at|@|di|pt\.?|cv\.?)\s+([A-Z][A-Za-z0-9&.,' -]{2,60})")
DOMAIN_ORG_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,})\s*(?:\||_)?\s*\.\s*(?:\||_)?\s*(ID|COM|CO)\b", re.I)
GENERIC_DOMAIN_EXCLUSIONS = frozenset({"gmail", "linkedin", "github", "www", "http", "https"})
COMPACT_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("WORKEXPERIENCE", "WORK EXPERIENCE"),
    ("ORGANIZATIONEXPERIENCE", "ORGANIZATION EXPERIENCE"),
    ("FORMALEDUCATION", "FORMAL EDUCATION"),
    ("PROJECTHIGHLIGHT", "PROJECT HIGHLIGHT"),
    ("TECHNICALSKILLS", "TECHNICAL SKILLS"),
    ("FULLSTACKWEBDEVELOPMENT", "FULL STACK WEB DEVELOPMENT"),
    ("FULLSTACKDEVELOPERINTERN", "FULL STACK DEVELOPER INTERN"),
    ("SOFTWAREENGINEERING", "SOFTWARE ENGINEERING"),
    ("BACKENDWEBPROGRAMMING", "BACKEND WEB PROGRAMMING"),
    ("APPLICATIONDEVELOPMENT", "APPLICATION DEVELOPMENT"),
    ("USEREXPERIENCE", "USER EXPERIENCE"),
)
EDUCATION_RE = re.compile(r"\b(?:bachelor|master|phd|diploma|sarjana|s1|s2|s3|universit(?:y|as)|institute|institut|college)\b", re.I)
CERTIFICATION_RE = re.compile(r"\b(?:aws|gcp|azure|cka|ckad|pmp|scrum|google|microsoft|oracle|certified|certification|sertifikasi)\b", re.I)
LANGUAGE_RE = re.compile(r"\b(?:english|indonesian|bahasa indonesia|mandarin|japanese|korean|arabic|inggris|indonesia|jepang|korea|arab)\b", re.I)
SENIORITY_RE = re.compile(r"\b(?:intern|junior|jr\.?|mid(?:-level)?|senior|sr\.?|lead|principal|staff|manager|head)\b", re.I)
TEXT_LITERAL_RE = re.compile(rb"\((?:\\.|[^\\()])*\)\s*T[Jj]")
TEXT_HEX_RE = re.compile(rb"<([0-9A-Fa-f\s]+)>\s*T[Jj]")
TEXT_ARRAY_RE = re.compile(rb"\[(.*?)\]\s*TJ", re.S)
ARRAY_TOKEN_RE = re.compile(rb"\((?:\\.|[^\\()])*\)|<([0-9A-Fa-f\s]+)>|-?\d+(?:\.\d+)?")
TO_UNICODE_CHAR_RE = re.compile(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
TO_UNICODE_RANGE_RE = re.compile(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
MAX_DECOMPRESSED_STREAM_BYTES = 1_000_000
AMBIGUOUS_AUTO_SKILL_ALIASES = frozenset({"ai", "api", "go", "js", "ml", "py", "ts"})


@dataclass(frozen=True)
class ParsedPdfEvidence:
    text: str
    page_count: int
    section_names: tuple[str, ...]
    has_contact_signal: bool
    has_date_signal: bool
    formatting_risks: tuple[str, ...]
    has_quantified_impact: bool
    estimated_experience_years: float | None
    word_count: int
    role_titles: tuple[str, ...]
    company_names: tuple[str, ...]
    has_education_signal: bool
    has_certification_signal: bool
    language_signals: tuple[str, ...]
    seniority_hints: tuple[str, ...]
    parse_quality: str
    detected_issues: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return bool(self.text.strip()) and self.parse_quality in {"good", "partial"}


def _decode_pdf_bytes(value: bytes) -> str:
    if not value:
        return ""
    if value.startswith((b"\xfe\xff", b"\xff\xfe")):
        encoding = "utf-16-be" if value.startswith(b"\xfe\xff") else "utf-16-le"
        return value[2:].decode(encoding, errors="ignore")
    if len(value) >= 2 and len(value) % 2 == 0 and value[0::2].count(0) >= len(value) // 3:
        decoded = value.decode("utf-16-be", errors="ignore")
        if decoded.strip():
            return decoded
    return value.decode("utf-8", errors="ignore")


def _decode_pdf_literal_bytes(value: bytes) -> bytes:
    inner = value[1:-1]
    output = bytearray()
    index = 0
    while index < len(inner):
        char = inner[index : index + 1]
        if char != b"\\":
            output.extend(char)
            index += 1
            continue
        index += 1
        if index >= len(inner):
            break
        escaped = inner[index : index + 1]
        if escaped in {b"\n", b"\r"}:
            index += 1
            continue
        if escaped in b"01234567":
            octal = bytes([escaped[0]])
            index += 1
            while index < len(inner) and len(octal) < 3 and inner[index : index + 1] in b"01234567":
                octal += inner[index : index + 1]
                index += 1
            output.append(int(octal, 8))
            continue
        replacements = {b"n": b"\n", b"r": b"\n", b"t": b"\t", b"b": b"\b", b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}
        output.extend(replacements.get(escaped, escaped))
        index += 1
    return bytes(output)


def _decode_pdf_literal(value: bytes) -> str:
    return _decode_pdf_bytes(_decode_pdf_literal_bytes(value))


def _decode_hex_bytes(hex_value: bytes) -> bytes:
    clean = re.sub(rb"\s+", b"", hex_value)
    if len(clean) % 2:
        clean += b"0"
    try:
        return bytes.fromhex(clean.decode("ascii"))
    except ValueError:
        return b""


def _decode_pdf_coded_bytes(raw: bytes, cmap: dict[bytes, str]) -> str:
    if not raw:
        return ""
    if cmap:
        mapped: list[str] = []
        index = 0
        keys_by_length = sorted({len(key) for key in cmap}, reverse=True)
        while index < len(raw):
            for key_length in keys_by_length:
                token = raw[index : index + key_length]
                if token in cmap:
                    mapped.append(cmap[token])
                    index += key_length
                    break
            else:
                mapped.append(_decode_pdf_bytes(raw[index : index + 1]))
                index += 1
        return "".join(mapped)
    return _decode_pdf_bytes(raw)


def _decode_pdf_hex_string(hex_value: bytes, cmap: dict[bytes, str]) -> str:
    return _decode_pdf_coded_bytes(_decode_hex_bytes(hex_value), cmap)


def _extract_text_literals_from_stream(stream: bytes, cmap: dict[bytes, str]) -> list[str]:
    parts: list[str] = []
    for match in TEXT_LITERAL_RE.finditer(stream):
        literal = match.group(0).rsplit(b")", 1)[0] + b")"
        parts.append(_decode_pdf_coded_bytes(_decode_pdf_literal_bytes(literal), cmap))
    for match in TEXT_HEX_RE.finditer(stream):
        parts.append(_decode_pdf_hex_string(match.group(1), cmap))
    for array_match in TEXT_ARRAY_RE.finditer(stream):
        array_parts: list[str] = []
        for token_match in ARRAY_TOKEN_RE.finditer(array_match.group(1)):
            token = token_match.group(0)
            if token.startswith(b"("):
                array_parts.append(_decode_pdf_coded_bytes(_decode_pdf_literal_bytes(token), cmap))
            elif token.startswith(b"<"):
                array_parts.append(_decode_pdf_hex_string(token[1:-1], cmap))
            else:
                try:
                    if float(token) <= -120 and array_parts and not array_parts[-1].endswith(" "):
                        array_parts.append(" ")
                except ValueError:
                    continue
        if array_parts:
            parts.append("".join(array_parts))
    return parts


def _pdf_streams(pdf_bytes: bytes) -> list[tuple[bytes, bytes]]:
    """Return (stream_dictionary_window, raw_stream_bytes) without broad PDF regex backtracking."""

    streams: list[tuple[bytes, bytes]] = []
    position = 0
    marker = b"stream"
    end_marker = b"endstream"
    while True:
        stream_start = pdf_bytes.find(marker, position)
        if stream_start < 0:
            break
        data_start = stream_start + len(marker)
        if pdf_bytes[data_start : data_start + 2] == b"\r\n":
            data_start += 2
        elif pdf_bytes[data_start : data_start + 1] in {b"\n", b"\r"}:
            data_start += 1
        data_end = pdf_bytes.find(end_marker, data_start)
        if data_end < 0:
            break
        dictionary_window = pdf_bytes[max(0, stream_start - 2048) : stream_start]
        streams.append((dictionary_window, pdf_bytes[data_start:data_end].strip(b"\r\n")))
        position = data_end + len(end_marker)
    return streams


def _decompress_flate_stream(stream: bytes) -> bytes | None:
    try:
        decompressor = zlib.decompressobj()
        data = decompressor.decompress(stream, MAX_DECOMPRESSED_STREAM_BYTES)
        if decompressor.unconsumed_tail:
            return None
        return data + decompressor.flush(MAX_DECOMPRESSED_STREAM_BYTES - len(data))
    except zlib.error:
        return None


def _decompressed_flate_streams(pdf_bytes: bytes) -> list[bytes]:
    streams: list[bytes] = []
    for dictionary_window, stream in _pdf_streams(pdf_bytes):
        if b"/FlateDecode" not in dictionary_window:
            continue
        decompressed = _decompress_flate_stream(stream)
        if decompressed is not None:
            streams.append(decompressed)
    return streams


def _candidate_text_streams(pdf_bytes: bytes) -> list[bytes]:
    raw_streams = [stream for _, stream in _pdf_streams(pdf_bytes)]
    streams = [*raw_streams, *_decompressed_flate_streams(pdf_bytes)]
    return streams or [pdf_bytes]


def _decode_unicode_hex(hex_value: bytes) -> str:
    raw = _decode_hex_bytes(hex_value)
    if not raw:
        return ""
    if len(raw) > 2 and len(raw) % 2 == 1 and raw.startswith(b"\x00"):
        raw = raw[1:]
    if len(raw) % 2 == 0:
        decoded = raw.decode("utf-16-be", errors="ignore").replace("\x00", "")
        if decoded.strip():
            return decoded
    return _decode_pdf_bytes(raw).replace("\x00", "")


def _extract_to_unicode_cmap(streams: Sequence[bytes]) -> dict[bytes, str]:
    cmap: dict[bytes, str] = {}
    for stream in streams:
        for block in re.findall(rb"beginbfchar(.*?)endbfchar", stream, re.S):
            for source_hex, target_hex in TO_UNICODE_CHAR_RE.findall(block):
                source = _decode_hex_bytes(source_hex)
                target = _decode_unicode_hex(target_hex)
                if source and target:
                    cmap[source] = target
        for block in re.findall(rb"beginbfrange(.*?)endbfrange", stream, re.S):
            for start_hex, end_hex, target_hex in TO_UNICODE_RANGE_RE.findall(block):
                start = int(start_hex, 16)
                end = int(end_hex, 16)
                target = int(target_hex, 16)
                width = max(1, len(start_hex) // 2)
                if end < start or end - start > 512:
                    continue
                for offset, codepoint in enumerate(range(start, end + 1)):
                    cmap[codepoint.to_bytes(width, "big")] = chr(target + offset)
    return cmap


def _glyph_encoded_pdf_text_variant(text: str) -> str:
    """Repair common subset-font glyph codes when a PDF lacks usable ToUnicode maps."""

    replacements = {
        "\x00": " ",
        "\x03": " ",
        "\x0f": ",",
        "\x10": "-",
        "\x11": ".",
        "\x12": "/",
        "#": "@",
        "[": "X",
        "\\": "Y",
        "]": "Z",
    }
    translated: list[str] = []
    for char in text:
        codepoint = ord(char)
        if char in replacements:
            translated.append(replacements[char])
        elif 0x13 <= codepoint <= 0x1C:
            translated.append(str(codepoint - 0x13))
        elif 0x24 <= codepoint <= 0x3D:
            translated.append(chr(codepoint + 29))
        elif "D" <= char <= "Z" or "d" <= char <= "z":
            translated.append(chr(codepoint - 3))
        else:
            translated.append(char)
    repaired = "".join(translated)
    repaired = re.sub(r"(?<=[A-Za-z0-9])\s*\n\s*(?=[A-Za-z0-9])", "", repaired)
    repaired = re.sub(r"[ \t]+", " ", repaired)
    return re.sub(r"\n{3,}", "\n\n", repaired).strip()


def _has_contact_signal(text: str) -> bool:
    return bool(CONTACT_RE.search(text) or SPACED_CONTACT_RE.search(text))


def _repair_compact_cv_text(text: str) -> str:
    repaired = text.replace("\x00", " ")
    for source, target in COMPACT_TEXT_REPLACEMENTS:
        repaired = re.sub(source, target, repaired, flags=re.I)
    repaired = re.sub(r"(?<=[A-Z])\\(?=\s|[A-Z])", "Y", repaired)
    repaired = re.sub(r"(?<=[A-Z])\[(?=\s|[A-Z])", "X", repaired)
    repaired = re.sub(r"(?<=[A-Z])\](?=\s|[A-Z])", "Z", repaired)
    repaired = re.sub(r"[ \t]+", " ", repaired)
    return re.sub(r"\n{3,}", "\n\n", repaired).strip()


def _text_signal_score(text: str) -> int:
    sections = _section_names(text)
    return (
        len(sections) * 20
        + (15 if _has_contact_signal(text) else 0)
        + (10 if DATE_RE.search(text) else 0)
        + min(20, len(re.findall(r"\b[A-Za-z]{3,}\b", text)) // 8)
    )


def _fragmented_text_variant(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    single_char_ratio = sum(1 for line in lines if len(line) == 1) / len(lines)
    if single_char_ratio < 0.45:
        return text
    repaired = re.sub(r"(?<=[A-Za-z0-9])\s*\n\s*(?=[A-Za-z0-9])", "", text)
    repaired = re.sub(r"[ \t]+", " ", repaired)
    return re.sub(r"\n{3,}", "\n\n", repaired).strip()


def _best_text_variant(text: str) -> str:
    fragmented = _fragmented_text_variant(text)
    candidates = (text, fragmented, _glyph_encoded_pdf_text_variant(text), _glyph_encoded_pdf_text_variant(fragmented))
    return max(candidates, key=_text_signal_score)


def _extract_text_literals(pdf_bytes: bytes) -> str:
    streams = _candidate_text_streams(pdf_bytes)
    cmap = _extract_to_unicode_cmap(streams)
    parts: list[str] = []
    for stream in streams:
        parts.extend(_extract_text_literals_from_stream(stream, cmap))
    text = "\n".join(part.strip() for part in parts if part.strip())
    return _repair_compact_cv_text(_best_text_variant(re.sub(r"[ \t]+", " ", text).strip()))


def _page_count(pdf_bytes: bytes) -> int:
    pages = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    if pages:
        return pages
    fallback = len(re.findall(rb"/Page\b", pdf_bytes))
    return max(1, fallback) if pdf_bytes.startswith(PDF_MAGIC) else 0


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", value.casefold())


def _section_names(text: str) -> tuple[str, ...]:
    compact_text = _compact_text(text)
    found = [
        name
        for name, pattern in SECTION_PATTERNS
        if pattern.search(text) or any(keyword in compact_text for keyword in SECTION_COMPACT_KEYWORDS.get(name, ()))
    ]
    return tuple(dict.fromkeys(found))


def _unique_limited(values: Sequence[str], limit: int = 6) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = re.sub(r"\s+", " ", value).strip(" .,-–—|\t\n\r")
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(clean[:80])
        if len(normalized) >= limit:
            break
    return tuple(normalized)


def _role_titles(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in ROLE_TITLE_RE.finditer(text):
        title = match.group(0)
        values.append(title)
        stripped = re.sub(r"^(?:senior|sr\.?|lead|principal|staff|junior|jr\.?|mid(?:-level)?)\s+", "", title, flags=re.I)
        if stripped != title:
            values.append(stripped)
    compact = _compact_text(text)
    for marker, label in COMPACT_ROLE_TITLE_MARKERS:
        if marker in compact:
            values.append(label)
    return _unique_limited(values)


def _company_names(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in COMPANY_RE.finditer(text):
        value = re.split(r"\b(?:19\d{2}|20\d{2}|improved|reduced|increased|decreased|optimized|saved|meningkatkan|mengurangi)\b", match.group(1), maxsplit=1, flags=re.I)[0]
        values.append(value)
    for match in DOMAIN_ORG_RE.finditer(text):
        name = match.group(1).strip("._-| ").casefold()
        suffix = match.group(2).lower()
        if name and name not in GENERIC_DOMAIN_EXCLUSIONS:
            values.append(f"{name.title()}.{suffix}")
    compact = _compact_text(text)
    if "amikomcomputerclub" in compact:
        values.append("AMIKOM Computer Club")
    return _unique_limited(values)


def _has_education_signal(text: str) -> bool:
    compact = _compact_text(text)
    compact_markers = ("university", "universitas", "institute", "institut", "bachelor", "sarjana", "diploma", "undergraduate")
    return bool(EDUCATION_RE.search(text) or any(marker in compact for marker in compact_markers))


def _language_signals(text: str) -> tuple[str, ...]:
    return _unique_limited(match.group(0).lower() for match in LANGUAGE_RE.finditer(text))


def _seniority_hints(text: str) -> tuple[str, ...]:
    return _unique_limited(match.group(0).lower().replace("sr.", "senior").replace("jr.", "junior") for match in SENIORITY_RE.finditer(text))


def parse_pdf_bytes(pdf_bytes: bytes, *, max_bytes: int, max_pages: int) -> ParsedPdfEvidence:
    """Parse PDF bytes into deterministic evidence without OCR/hallucination."""

    issues: list[str] = []
    risks: list[str] = []
    if not pdf_bytes:
        return ParsedPdfEvidence("", 0, (), False, False, (), False, None, 0, (), (), False, False, (), (), "failed", ("empty PDF upload",))
    if len(pdf_bytes) > max_bytes:
        return ParsedPdfEvidence("", 0, (), False, False, (), False, None, 0, (), (), False, False, (), (), "failed", ("PDF exceeds configured byte limit",))
    if not pdf_bytes.lstrip().startswith(PDF_MAGIC):
        return ParsedPdfEvidence("", 0, (), False, False, (), False, None, 0, (), (), False, False, (), (), "failed", ("file is not a PDF",))

    page_count = _page_count(pdf_bytes)
    if page_count > max_pages:
        return ParsedPdfEvidence("", page_count, (), False, False, (), False, None, 0, (), (), False, False, (), (), "failed", ("PDF exceeds configured page limit",))

    if re.search(rb"/XObject|/Image", pdf_bytes):
        risks.append("image content present")
    if re.search(rb"/Columns\s+\d", pdf_bytes):
        risks.append("multi-column layout marker present")

    text = _extract_text_literals(pdf_bytes)
    sections = _section_names(text)
    has_contact = _has_contact_signal(text)
    has_date = bool(DATE_RE.search(text))
    has_quantified_impact = bool(QUANTIFIED_IMPACT_RE.search(text))
    estimated_experience_years = estimate_experience_years_from_text(text)
    word_count = len(re.findall(r"\w+", text))
    role_titles = _role_titles(text)
    company_names = _company_names(text)
    has_education_signal = _has_education_signal(text)
    has_certification_signal = bool(CERTIFICATION_RE.search(text))
    language_signals = _language_signals(text)
    seniority_hints = _seniority_hints(text)
    if not text.strip():
        issues.append("no extractable PDF text; scanned or image-only CV suspected")
    if not sections:
        issues.append("standard CV sections not detected")
    if not has_contact:
        issues.append("contact signal not detected")
    if not has_date:
        issues.append("date or timeline signal not detected")
    if 0 < word_count < 40 and len(sections) < 2:
        issues.append("CV text is short for ATS context")
    elif word_count > 1200:
        issues.append("CV text may be too long for fast recruiter review")
    if word_count >= 80 and not has_quantified_impact:
        issues.append("quantified impact signal not detected")
    if word_count >= 40 and "experience" in sections and not role_titles:
        issues.append("role title evidence not detected")
    if word_count >= 40 and "experience" in sections and not company_names:
        issues.append("company evidence not detected")
    if "education" in sections and not has_education_signal:
        issues.append("education credential detail not detected")
    if risks:
        issues.extend(risks)

    if not text.strip():
        quality = "empty"
    elif issues:
        quality = "partial"
    else:
        quality = "good"
    return ParsedPdfEvidence(
        text,
        page_count,
        sections,
        has_contact,
        has_date,
        tuple(risks),
        has_quantified_impact,
        estimated_experience_years,
        word_count,
        role_titles,
        company_names,
        has_education_signal,
        has_certification_signal,
        language_signals,
        seniority_hints,
        quality,
        tuple(issues),
    )


def estimate_experience_years_from_text(text: str) -> float | None:
    """Best-effort deterministic years-of-experience signal from CV text."""

    explicit_values = [float(match.group(1).replace(",", ".")) for match in EXPLICIT_EXPERIENCE_RE.finditer(text)]
    if explicit_values:
        return min(40.0, max(explicit_values))

    years = sorted({int(match.group(1)) for match in YEAR_RE.finditer(text)})
    bounded = [year for year in years if 1970 <= year <= 2035]
    if len(bounded) >= 2:
        span = max(bounded) - min(bounded)
        if 0 < span <= 40:
            return float(span)
    return None


def _skill_pattern(skill: str) -> re.Pattern[str]:
    escaped = re.escape(skill).replace(r"\ ", r"[\s._/-]+")
    return re.compile(rf"(?<![a-z0-9+#]){escaped}(?![a-z0-9+#])", re.I)


def normalized_skills_from_text(text: str, candidate_skill_hints: Sequence[str] = ()) -> tuple[str, ...]:
    """Extract normalized skill evidence from candidate hints plus approved aliases."""

    explicit_sql_requested = any(normalize_skill_token(skill) == "sql" for skill in candidate_skill_hints)
    search_terms = set(candidate_skill_hints)
    search_terms.update(skill for skill in SKILL_ALIASES.keys() if skill not in AMBIGUOUS_AUTO_SKILL_ALIASES)
    matched: set[str] = set()
    for raw_skill in search_terms:
        normalized = normalize_skill_token(raw_skill)
        if not normalized:
            continue
        if _skill_pattern(str(raw_skill)).search(text) or _skill_pattern(normalized).search(text):
            matched.add(normalized)
            continue
        if explicit_sql_requested and normalized == "sql" and re.search(r"\b(?:mysql|postgresql|postgres|sql\s*server)\b", text, re.I):
            matched.add("sql")
    return tuple(sorted(matched))


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
    if evidence.word_count and evidence.word_count < 40 and len(evidence.section_names) < 2:
        score -= 10
    if evidence.word_count > 1200:
        score -= 10
    if evidence.word_count >= 80 and not evidence.has_quantified_impact:
        score -= 8
    if evidence.word_count >= 40 and "experience" in evidence.section_names and not evidence.role_titles:
        score -= 5
    if evidence.word_count >= 40 and "experience" in evidence.section_names and not evidence.company_names:
        score -= 5
    score -= min(20, len(evidence.formatting_risks) * 10)
    return max(0, min(100, int(score))), evidence.detected_issues, evidence.parse_quality != "good"
