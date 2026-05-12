# STAP - Scientific Text Analysis Platform

STAP is a desktop research tool for building clean, reproducible text corpora and analyzing them with evidence-first methods.

The project starts from a practical workflow: a source text is prepared by FileMerger or another extraction tool, then STAP cleans the corpus and calculates transparent text metrics. The long-term goal is a scientific analysis platform where every conclusion can be traced back to the source text, quotation, and local context.

## Current Focus

Version `0.2` focuses on the foundation:

- load source text or documents;
- extract text into a unified corpus;
- remove common contamination such as page numbers, ISBN lines, repeated headers, table-of-contents lines, technical scan noise, and probable footnotes;
- repair hyphenated line breaks from OCR or PDF extraction;
- normalize whitespace;
- analyze the cleaned corpus;
- extract key concepts;
- collect citation evidence for the strongest concepts;
- find semantically relevant paragraphs with a local TF-IDF model;
- build a concept co-occurrence knowledge graph;
- export a report, clean text, analysis JSON, and Graphviz DOT graph.

## Why Cleaning Comes First

Scientific text analysis depends on corpus integrity. If a book contains translator notes, headers, page numbers, OCR artifacts, publisher metadata, or unrelated introductions, frequency analysis and semantic analysis become contaminated.

STAP treats cleaning as a first-class step:

```text
source text
  -> extraction
  -> corpus cleaning
  -> clean corpus
  -> analysis
  -> evidence report
```

## Features

- Reads `TXT`, `RTF`, `PDF`, `DOCX`, `XLS`, `XLSX`, `FB2`, `FB2.ZIP`, `EPUB`, `HTML`, `HTM`, `MD`, and `CSV`.
- Creates a cleaned text corpus before analysis.
- Counts characters, words, unique words, sentences, paragraphs, average word length, and average sentence length.
- Shows top frequent words after stop-word filtering.
- Reports removed fragments and applied transformations.
- Semantic Analysis Engine: extracts concepts and relevant paragraphs using deterministic local NLP.
- Citation & Evidence Engine: links key concepts to exact sentence-level evidence.
- Knowledge Graph Engine: builds concept co-occurrence edges that can be exported as Graphviz DOT.
- Exports analysis reports, clean corpus text, analysis JSON, and knowledge graph DOT files.

## Run

```bat
pip install -r requirements.txt
python main.py
```

On Windows you can also run:

```bat
run.bat
```

## Project Structure

- `main.py` - Tkinter desktop UI and the clean/analyze workflow.
- `corpus_cleaner.py` - corpus cleaning rules and cleaning report generation.
- `analysis_layers.py` - semantic, citation/evidence, NLP, and graph analysis layers.
- `text_core.py` - text extraction and basic metrics.
- `docs/AI_talks.txt` - source notes that shaped the project direction.
- `docs/roadmap.md` - staged development roadmap.
- `requirements.txt` - Python dependencies.

## Roadmap

1. Corpus Cleaning Engine - implemented as the first layer.
2. Basic frequency and citation analysis - implemented with deterministic local rules.
3. Semantic Analysis Engine - initial TF-IDF implementation is active.
4. Knowledge Graph Engine - initial concept co-occurrence graph is active.
5. Corpus JSON export with paragraph/chapter structure.
6. Embeddings, semantic search index, and author comparison.
7. Evidence-first research reports.

## Build Windows EXE

```bat
python -m PyInstaller --noconfirm STAP.spec
```

The generated application will be placed under:

```text
dist\STAP\STAP.exe
```

## Repository

```text
https://github.com/DmitriiMeius/STAP.git
```
