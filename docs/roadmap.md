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
- Lexical diversity.
- Readability and complexity.
- N-grams and repeated formulas.
- Sentiment and stance markers.
- Concept dynamics across corpus segments.

Status: active in `analysis_layers.py`.

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

- Local TF-IDF paragraph matching.
- Embeddings.
- Semantic search index.
- Concept maps.
- Author comparison.
- Corpus-to-corpus comparison.

Status: deterministic TF-IDF paragraph matching is implemented. Embeddings and vector databases are planned for later versions.

## Phase 5 - Scientific Mode

Goal: every conclusion must be traceable.

Each generated insight should include:

- claim;
- source;
- quotation;
- context;
- cleaning assumptions;
- reproducible method.

## Current Engines

- Corpus Cleaning Engine: deterministic cleanup of common contamination.
- NLP Layer: tokenization, sentence extraction, concept counting, stop-word filtering.
- Semantic Analysis Engine: local TF-IDF paragraph matching against dominant concepts.
- Citation & Evidence Engine: sentence-level evidence records for each concept.
- Knowledge Graph Engine: concept co-occurrence graph exportable as Graphviz DOT.
- Lexical Diversity Analysis: vocabulary breadth and repetition.
- Readability and Complexity Analysis: sentence and word density.
- N-gram and Formula Analysis: repeated two-word and three-word patterns.
- Sentiment and Stance Marker Analysis: evaluative, critical, uncertain, and normative markers.
- Concept Dynamics Analysis: segment-by-segment vocabulary and stance changes.
