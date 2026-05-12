# STAP - Scientific Text Analysis Platform

STAP is a desktop research tool for building clean, reproducible text corpora and analyzing them with evidence-first methods.

The project starts from a practical workflow: a source text is prepared by FileMerger or another extraction tool, then STAP cleans the corpus and calculates transparent text metrics. The long-term goal is a scientific analysis platform where every conclusion can be traced back to the source text, quotation, and local context.

## Current Focus

Version `0.1` focuses on the foundation:

- load source text or documents;
- extract text into a unified corpus;
- remove common contamination such as page numbers, ISBN lines, repeated headers, table-of-contents lines, technical scan noise, and probable footnotes;
- repair hyphenated line breaks from OCR or PDF extraction;
- normalize whitespace;
- analyze the cleaned corpus;
- export a report and clean text.

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
- Exports analysis reports and clean corpus text.

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
- `text_core.py` - text extraction and basic metrics.
- `docs/AI_talks.txt` - source notes that shaped the project direction.
- `docs/roadmap.md` - staged development roadmap.
- `requirements.txt` - Python dependencies.

## Roadmap

1. Corpus Cleaning Engine
2. Basic frequency and citation analysis
3. Corpus JSON export with paragraph/chapter structure
4. Semantic search and embeddings
5. Concept tracking and author comparison
6. Evidence-first research reports

## Repository

```text
https://github.com/DmitriiMeius/STAP.git
```
