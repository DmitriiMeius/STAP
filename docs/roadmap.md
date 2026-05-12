# STAP Roadmap

## Phase 1 - Corpus Cleaning Engine

Goal: prepare a clean, reproducible research corpus before any analysis.

- Remove technical noise: page numbers, repeated headers, scan marks, ISBN lines.
- Remove structural noise: table of contents, publisher blocks, indexes, bibliographies.
- Flag probable foreign text: translator notes, editor comments, footnotes.
- Repair OCR/PDF extraction artifacts.
- Export clean text and a cleaning report.

## Phase 2 - Basic Analyzer

Goal: make the first useful research reports.

- Frequency analysis.
- Keyword extraction.
- Repeated concept detection.
- Quotation/context lookup.
- Per-file and total corpus reports.

## Phase 3 - Structured Corpus

Goal: move from flat text to research data.

```json
{
  "author": "Author",
  "title": "Title",
  "chapter": "Chapter 1",
  "paragraph": 15,
  "clean_text": "...",
  "source": {
    "file": "book.txt",
    "line_start": 120,
    "line_end": 125
  }
}
```

## Phase 4 - Semantic Layer

Goal: add meaning-aware tools without losing evidence.

- Embeddings.
- Semantic search.
- Concept maps.
- Author comparison.
- Corpus-to-corpus comparison.

## Phase 5 - Scientific Mode

Goal: every conclusion must be traceable.

Each generated insight should include:

- claim;
- source;
- quotation;
- context;
- cleaning assumptions;
- reproducible method.
