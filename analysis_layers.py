from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from text_core import DEFAULT_STOP_WORDS, FileAnalysis, SENTENCE_RE, tokenize_words


@dataclass(frozen=True)
class SentenceEvidence:
    source: str
    sentence_index: int
    text: str
    score: float
    matched_terms: list[str]


@dataclass(frozen=True)
class ConceptRecord:
    term: str
    count: int
    sources: list[str]
    evidence: list[SentenceEvidence]


@dataclass(frozen=True)
class SemanticMatch:
    source: str
    paragraph_index: int
    score: float
    text: str
    top_terms: list[str]


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: int


@dataclass(frozen=True)
class KnowledgeGraph:
    nodes: list[tuple[str, int]]
    edges: list[GraphEdge]


@dataclass(frozen=True)
class AnalysisBundle:
    concepts: list[ConceptRecord]
    semantic_matches: list[SemanticMatch]
    graph: KnowledgeGraph


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.findall(text) if part.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def terms_for_text(text: str, *, stop_words: set[str] | None = None) -> list[str]:
    stop_words = stop_words or DEFAULT_STOP_WORDS
    return [word for word in tokenize_words(text) if word not in stop_words and len(word) > 2]


def sentence_score(sentence: str, concept_terms: set[str]) -> tuple[float, list[str]]:
    words = terms_for_text(sentence)
    if not words:
        return 0.0, []
    counts = Counter(words)
    matched = sorted(term for term in concept_terms if term in counts)
    if not matched:
        return 0.0, []
    score = sum(counts[term] for term in matched) / math.sqrt(len(words))
    return score, matched


def extract_concepts(
    analyses: list[FileAnalysis],
    *,
    top_terms: int = 30,
    evidence_per_term: int = 3,
) -> list[ConceptRecord]:
    corpus_counter: Counter[str] = Counter()
    source_terms: dict[str, Counter[str]] = {}
    source_sentences: dict[str, list[str]] = {}

    for analysis in analyses:
        source = analysis.path.name
        terms = terms_for_text(analysis.text)
        source_terms[source] = Counter(terms)
        source_sentences[source] = split_sentences(analysis.text)
        corpus_counter.update(terms)

    concepts: list[ConceptRecord] = []
    for term, count in corpus_counter.most_common(top_terms):
        evidence: list[SentenceEvidence] = []
        sources = sorted(source for source, counter in source_terms.items() if counter.get(term, 0) > 0)
        for source, sentences in source_sentences.items():
            for index, sentence in enumerate(sentences, start=1):
                score, matched = sentence_score(sentence, {term})
                if score > 0:
                    evidence.append(
                        SentenceEvidence(
                            source=source,
                            sentence_index=index,
                            text=sentence,
                            score=score,
                            matched_terms=matched,
                        )
                    )
        evidence.sort(key=lambda item: item.score, reverse=True)
        concepts.append(
            ConceptRecord(
                term=term,
                count=count,
                sources=sources,
                evidence=evidence[:evidence_per_term],
            )
        )
    return concepts


def build_idf(paragraph_terms: list[list[str]]) -> dict[str, float]:
    total = len(paragraph_terms) or 1
    document_frequency: Counter[str] = Counter()
    for terms in paragraph_terms:
        document_frequency.update(set(terms))
    return {
        term: math.log((1 + total) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }


def vectorize(terms: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(terms)
    total = sum(counts.values()) or 1
    return {term: (count / total) * idf.get(term, 1.0) for term, count in counts.items()}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    common = set(left).intersection(right)
    numerator = sum(left[term] * right[term] for term in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def semantic_matches_for_concepts(
    analyses: list[FileAnalysis],
    concepts: list[ConceptRecord],
    *,
    limit: int = 12,
) -> list[SemanticMatch]:
    concept_terms = [concept.term for concept in concepts[:12]]
    if not concept_terms:
        return []

    paragraphs: list[tuple[str, int, str, list[str]]] = []
    for analysis in analyses:
        for index, paragraph in enumerate(split_paragraphs(analysis.text), start=1):
            terms = terms_for_text(paragraph)
            if terms:
                paragraphs.append((analysis.path.name, index, paragraph, terms))

    paragraph_terms = [terms for _, _, _, terms in paragraphs]
    idf = build_idf(paragraph_terms + [concept_terms])
    query_vector = vectorize(concept_terms, idf)

    matches: list[SemanticMatch] = []
    for source, index, paragraph, terms in paragraphs:
        paragraph_vector = vectorize(terms, idf)
        score = cosine_similarity(query_vector, paragraph_vector)
        if score <= 0:
            continue
        top_terms = [term for term, _ in Counter(terms).most_common(8) if term in idf]
        matches.append(
            SemanticMatch(
                source=source,
                paragraph_index=index,
                score=score,
                text=paragraph[:1200],
                top_terms=top_terms,
            )
        )

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:limit]


def build_knowledge_graph(
    analyses: list[FileAnalysis],
    concepts: list[ConceptRecord],
    *,
    concept_limit: int = 20,
) -> KnowledgeGraph:
    concept_counts = {concept.term: concept.count for concept in concepts[:concept_limit]}
    concept_terms = set(concept_counts)
    edge_weights: defaultdict[tuple[str, str], int] = defaultdict(int)

    for analysis in analyses:
        for sentence in split_sentences(analysis.text):
            present = sorted(set(terms_for_text(sentence)).intersection(concept_terms))
            for left_index, left in enumerate(present):
                for right in present[left_index + 1 :]:
                    edge_weights[(left, right)] += 1

    edges = [
        GraphEdge(source=left, target=right, weight=weight)
        for (left, right), weight in edge_weights.items()
        if weight > 0
    ]
    edges.sort(key=lambda item: item.weight, reverse=True)
    return KnowledgeGraph(
        nodes=sorted(concept_counts.items(), key=lambda item: item[1], reverse=True),
        edges=edges[:80],
    )


def build_analysis_bundle(analyses: list[FileAnalysis]) -> AnalysisBundle:
    concepts = extract_concepts(analyses)
    semantic_matches = semantic_matches_for_concepts(analyses, concepts)
    graph = build_knowledge_graph(analyses, concepts)
    return AnalysisBundle(concepts=concepts, semantic_matches=semantic_matches, graph=graph)


def format_evidence_report(bundle: AnalysisBundle) -> str:
    lines = [
        "Semantic Analysis Engine",
        "=" * 60,
        "Key concepts:",
    ]
    for concept in bundle.concepts[:20]:
        source_list = ", ".join(concept.sources[:4])
        if len(concept.sources) > 4:
            source_list += ", ..."
        lines.append(f"- {concept.term}: {concept.count} [{source_list}]")

    lines.extend(["", "Citation & Evidence Engine", "=" * 60])
    for concept in bundle.concepts[:12]:
        lines.append(f"{concept.term} ({concept.count})")
        for item in concept.evidence:
            lines.append(
                f"- {item.source}, sentence {item.sentence_index}, score {item.score:.2f}: {item.text}"
            )
        lines.append("")

    lines.extend(["Semantic paragraph matches", "=" * 60])
    for item in bundle.semantic_matches:
        terms = ", ".join(item.top_terms[:6])
        lines.append(f"- {item.source}, paragraph {item.paragraph_index}, score {item.score:.3f}, terms: {terms}")
        lines.append(f"  {item.text}")

    lines.extend(["", "Knowledge Graph Engine", "=" * 60, "Top co-occurrences:"])
    if bundle.graph.edges:
        for edge in bundle.graph.edges[:30]:
            lines.append(f"- {edge.source} -> {edge.target}: {edge.weight}")
    else:
        lines.append("- No concept co-occurrences found.")

    return "\n".join(lines)


def export_analysis_json(bundle: AnalysisBundle, path: Path) -> None:
    data = asdict(bundle)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_graph_dot(graph: KnowledgeGraph, path: Path) -> None:
    lines = ["graph STAPKnowledgeGraph {"]
    for node, count in graph.nodes:
        label = f"{node} ({count})"
        lines.append(f'  "{node}" [label="{label}"];')
    for edge in graph.edges:
        lines.append(f'  "{edge.source}" -- "{edge.target}" [label="{edge.weight}", weight={edge.weight}];')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
