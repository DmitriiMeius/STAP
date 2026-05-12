from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CleaningOptions:
    remove_page_numbers: bool = True
    remove_isbn_lines: bool = True
    remove_toc_lines: bool = True
    remove_probable_footnotes: bool = True
    remove_repeated_headers: bool = True
    repair_hyphenated_line_breaks: bool = True
    normalize_whitespace: bool = True


@dataclass(frozen=True)
class RemovalRecord:
    reason: str
    line_number: int
    text: str


@dataclass
class CleaningResult:
    clean_text: str
    removed: list[RemovalRecord] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)


PAGE_NUMBER_RE = re.compile(r"^\s*(?:[-–—]\s*)?\d{1,5}(?:\s*[-–—])?\s*$")
ISBN_RE = re.compile(r"\bISBN(?:-1[03])?\b|(?:978|979)[-\s]?\d", re.IGNORECASE)
TOC_RE = re.compile(
    r"^\s*(?:contents|table of contents|оглавление|содержание)\b|"
    r".+\.{4,}\s*\d{1,5}\s*$",
    re.IGNORECASE,
)
FOOTNOTE_RE = re.compile(r"^\s*(?:\d{1,3}|[*†‡])[\).]?\s+\S.{0,220}$")
HEADER_NOISE_RE = re.compile(
    r"^\s*(?:google books|digitized by|scan|электронная библиотека|"
    r"www\.|https?://|copyright|all rights reserved)\b",
    re.IGNORECASE,
)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_control_chars(text: str) -> str:
    return CONTROL_CHARS_RE.sub("", text)


def repair_hyphenated_breaks(text: str) -> tuple[str, int]:
    repaired = re.subn(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])", r"\1\2", text)
    return repaired


def find_repeated_short_lines(lines: list[str], *, min_count: int = 3) -> set[str]:
    counts: dict[str, int] = {}
    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip())
        if not normalized or len(normalized) > 90:
            continue
        if PAGE_NUMBER_RE.match(normalized):
            continue
        counts[normalized.lower()] = counts.get(normalized.lower(), 0) + 1
    return {line for line, count in counts.items() if count >= min_count}


def should_remove_line(
    line: str,
    line_number: int,
    repeated_lines: set[str],
    options: CleaningOptions,
) -> str | None:
    stripped = line.strip()
    normalized = re.sub(r"\s+", " ", stripped).lower()

    if not stripped:
        return None
    if HEADER_NOISE_RE.search(stripped):
        return "technical-noise"
    if options.remove_page_numbers and PAGE_NUMBER_RE.match(stripped):
        return "page-number"
    if options.remove_isbn_lines and ISBN_RE.search(stripped):
        return "isbn-or-publisher-id"
    if options.remove_toc_lines and TOC_RE.search(stripped):
        return "table-of-contents"
    if options.remove_repeated_headers and normalized in repeated_lines:
        return "repeated-header-footer"
    if options.remove_probable_footnotes and FOOTNOTE_RE.match(stripped) and line_number > 5:
        return "probable-footnote"
    return None


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_corpus_text(text: str, options: CleaningOptions | None = None) -> CleaningResult:
    options = options or CleaningOptions()
    transformations: list[str] = []

    text = normalize_newlines(text)
    text_without_controls = remove_control_chars(text)
    if text_without_controls != text:
        transformations.append("Removed control characters")
        text = text_without_controls

    if options.repair_hyphenated_line_breaks:
        text, repaired_count = repair_hyphenated_breaks(text)
        if repaired_count:
            transformations.append(f"Repaired hyphenated line breaks: {repaired_count}")

    lines = text.split("\n")
    repeated_lines = find_repeated_short_lines(lines) if options.remove_repeated_headers else set()
    clean_lines: list[str] = []
    removed: list[RemovalRecord] = []

    for index, line in enumerate(lines, start=1):
        reason = should_remove_line(line, index, repeated_lines, options)
        if reason:
            removed.append(RemovalRecord(reason=reason, line_number=index, text=line.strip()))
            continue
        clean_lines.append(line)

    clean_text = "\n".join(clean_lines)
    if options.normalize_whitespace:
        before = clean_text
        clean_text = collapse_whitespace(clean_text)
        if clean_text != before:
            transformations.append("Normalized whitespace")

    return CleaningResult(clean_text=clean_text, removed=removed, transformations=transformations)


def format_cleaning_report(result: CleaningResult) -> str:
    lines = [
        "Corpus Cleaning Report",
        "=" * 60,
        f"Removed lines: {len(result.removed)}",
        f"Transformations: {len(result.transformations)}",
        "",
        "Transformations:",
    ]
    if result.transformations:
        lines.extend(f"- {item}" for item in result.transformations)
    else:
        lines.append("- None")

    lines.extend(["", "Removed fragments:"])
    if result.removed:
        for item in result.removed[:500]:
            sample = item.text[:180]
            lines.append(f"- line {item.line_number}: {item.reason}: {sample}")
        if len(result.removed) > 500:
            lines.append(f"- ... {len(result.removed) - 500} more removed lines")
    else:
        lines.append("- None")

    return "\n".join(lines)
