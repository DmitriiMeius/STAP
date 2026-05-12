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
class ReadabilityProfile:
    average_sentence_words: float
    average_word_length: float
    long_sentence_count: int
    long_sentence_share: float
    interpretation: str


@dataclass(frozen=True)
class LexicalDiversityProfile:
    total_terms: int
    unique_terms: int
    type_token_ratio: float
    hapax_legomena: int
    hapax_share: float


@dataclass(frozen=True)
class NgramRecord:
    phrase: str
    count: int


@dataclass(frozen=True)
class SentimentProfile:
    positive_markers: int
    negative_markers: int
    uncertainty_markers: int
    authority_markers: int
    dominant_tone: str


@dataclass(frozen=True)
class DynamicsSegment:
    segment: int
    start_percent: int
    end_percent: int
    top_terms: list[tuple[str, int]]
    sentiment: SentimentProfile


@dataclass(frozen=True)
class AnalysisBundle:
    concepts: list[ConceptRecord]
    semantic_matches: list[SemanticMatch]
    graph: KnowledgeGraph
    readability: ReadabilityProfile
    lexical_diversity: LexicalDiversityProfile
    bigrams: list[NgramRecord]
    trigrams: list[NgramRecord]
    sentiment: SentimentProfile
    dynamics: list[DynamicsSegment]


POSITIVE_MARKERS = {
    "good", "true", "truth", "clear", "strong", "valid", "right", "benefit", "stable",
    "добро", "истина", "истинный", "ясно", "сильный", "верный", "польза", "благо",
}
NEGATIVE_MARKERS = {
    "bad", "false", "error", "wrong", "weak", "danger", "risk", "problem", "crisis",
    "зло", "ложь", "ошибка", "неверный", "слабый", "опасность", "риск", "проблема", "кризис",
}
UNCERTAINTY_MARKERS = {
    "maybe", "perhaps", "possibly", "uncertain", "doubt", "seems", "вероятно", "возможно",
    "может", "сомнение", "сомневаться", "кажется", "неясно",
}
AUTHORITY_MARKERS = {
    "must", "should", "law", "order", "duty", "proof", "therefore", "обязан", "должен",
    "закон", "порядок", "долг", "доказательство", "следовательно", "необходимо",
}


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.findall(text) if part.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def terms_for_text(text: str, *, stop_words: set[str] | None = None) -> list[str]:
    stop_words = stop_words or DEFAULT_STOP_WORDS
    return [word for word in tokenize_words(text) if word not in stop_words and len(word) > 2]


def all_clean_text(analyses: list[FileAnalysis]) -> str:
    return "\n\n".join(analysis.text for analysis in analyses)


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


def build_readability_profile(analyses: list[FileAnalysis]) -> ReadabilityProfile:
    text = all_clean_text(analyses)
    sentences = split_sentences(text)
    words = tokenize_words(text)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    average_sentence_words = (sum(sentence_lengths) / len(sentence_lengths)) if sentence_lengths else 0.0
    average_word_length = (sum(len(word) for word in words) / len(words)) if words else 0.0
    long_sentence_count = sum(1 for length in sentence_lengths if length >= 30)
    long_sentence_share = (long_sentence_count / len(sentence_lengths)) if sentence_lengths else 0.0
    if average_sentence_words >= 30:
        interpretation = "Dense academic prose: long sentence chains require close reading."
    elif average_sentence_words >= 18:
        interpretation = "Moderately complex prose: suitable for concept and argument tracking."
    else:
        interpretation = "Compact prose: ideas are likely split into shorter units."
    return ReadabilityProfile(
        average_sentence_words=average_sentence_words,
        average_word_length=average_word_length,
        long_sentence_count=long_sentence_count,
        long_sentence_share=long_sentence_share,
        interpretation=interpretation,
    )


def build_lexical_diversity_profile(analyses: list[FileAnalysis]) -> LexicalDiversityProfile:
    terms = terms_for_text(all_clean_text(analyses))
    counts = Counter(terms)
    total_terms = len(terms)
    unique_terms = len(counts)
    hapax = sum(1 for count in counts.values() if count == 1)
    return LexicalDiversityProfile(
        total_terms=total_terms,
        unique_terms=unique_terms,
        type_token_ratio=(unique_terms / total_terms) if total_terms else 0.0,
        hapax_legomena=hapax,
        hapax_share=(hapax / unique_terms) if unique_terms else 0.0,
    )


def build_ngrams(analyses: list[FileAnalysis], n: int, *, limit: int = 25) -> list[NgramRecord]:
    terms = terms_for_text(all_clean_text(analyses))
    grams = (" ".join(terms[index : index + n]) for index in range(max(0, len(terms) - n + 1)))
    return [NgramRecord(phrase=phrase, count=count) for phrase, count in Counter(grams).most_common(limit)]


def build_sentiment_profile(text: str) -> SentimentProfile:
    terms = terms_for_text(text, stop_words=set())
    counts = Counter(terms)
    positive = sum(counts[term] for term in POSITIVE_MARKERS)
    negative = sum(counts[term] for term in NEGATIVE_MARKERS)
    uncertainty = sum(counts[term] for term in UNCERTAINTY_MARKERS)
    authority = sum(counts[term] for term in AUTHORITY_MARKERS)
    tone_scores = {
        "positive/evaluative": positive,
        "negative/critical": negative,
        "uncertain/questioning": uncertainty,
        "normative/authoritative": authority,
    }
    dominant = max(tone_scores.items(), key=lambda item: item[1])[0] if any(tone_scores.values()) else "neutral/undetected"
    return SentimentProfile(
        positive_markers=positive,
        negative_markers=negative,
        uncertainty_markers=uncertainty,
        authority_markers=authority,
        dominant_tone=dominant,
    )


def build_dynamics(analyses: list[FileAnalysis], *, segments: int = 5) -> list[DynamicsSegment]:
    text = all_clean_text(analyses)
    if not text.strip():
        return []
    length = len(text)
    result: list[DynamicsSegment] = []
    for index in range(segments):
        start = int(length * index / segments)
        end = int(length * (index + 1) / segments)
        segment_text = text[start:end]
        top_terms = Counter(terms_for_text(segment_text)).most_common(8)
        result.append(
            DynamicsSegment(
                segment=index + 1,
                start_percent=round(index * 100 / segments),
                end_percent=round((index + 1) * 100 / segments),
                top_terms=top_terms,
                sentiment=build_sentiment_profile(segment_text),
            )
        )
    return result


def build_analysis_bundle(analyses: list[FileAnalysis]) -> AnalysisBundle:
    concepts = extract_concepts(analyses)
    semantic_matches = semantic_matches_for_concepts(analyses, concepts)
    graph = build_knowledge_graph(analyses, concepts)
    return AnalysisBundle(
        concepts=concepts,
        semantic_matches=semantic_matches,
        graph=graph,
        readability=build_readability_profile(analyses),
        lexical_diversity=build_lexical_diversity_profile(analyses),
        bigrams=build_ngrams(analyses, 2),
        trigrams=build_ngrams(analyses, 3),
        sentiment=build_sentiment_profile(all_clean_text(analyses)),
        dynamics=build_dynamics(analyses),
    )


def format_evidence_report(bundle: AnalysisBundle) -> str:
    lines = [
        "Analysis Map",
        "=" * 60,
        "What is included and why:",
        "- Corpus Cleaning: removes contamination before interpretation.",
        "- Descriptive Metrics: shows size, density, and basic structure of the clean corpus.",
        "- Lexical Diversity: estimates vocabulary breadth and repetition.",
        "- Readability/Complexity: detects dense sentences that affect interpretation.",
        "- N-gram Analysis: finds repeated multi-word formulas, not just isolated words.",
        "- Sentiment/Stance Markers: detects evaluative, critical, uncertain, and normative language.",
        "- Concept Dynamics: shows how dominant terms change from beginning to end.",
        "- Semantic Analysis: finds concept-heavy paragraphs using local TF-IDF similarity.",
        "- Citation & Evidence: ties concepts to exact sentences.",
        "- Knowledge Graph: maps terms that repeatedly appear together.",
        "",
        "Lexical Diversity Analysis",
        "=" * 60,
        "Why: a high diversity score suggests broad vocabulary; a low score suggests repetition or formulaic language.",
        f"Total analysis terms: {bundle.lexical_diversity.total_terms}",
        f"Unique analysis terms: {bundle.lexical_diversity.unique_terms}",
        f"Type-token ratio: {bundle.lexical_diversity.type_token_ratio:.3f}",
        f"Hapax legomena: {bundle.lexical_diversity.hapax_legomena} ({bundle.lexical_diversity.hapax_share:.1%} of unique terms)",
        "",
        "Readability and Complexity Analysis",
        "=" * 60,
        "Why: sentence length and word length help identify dense argumentation and difficult passages.",
        f"Average sentence length: {bundle.readability.average_sentence_words:.2f} words",
        f"Average word length: {bundle.readability.average_word_length:.2f} characters",
        f"Long sentences: {bundle.readability.long_sentence_count} ({bundle.readability.long_sentence_share:.1%})",
        f"Interpretation: {bundle.readability.interpretation}",
        "",
        "N-gram and Formula Analysis",
        "=" * 60,
        "Why: repeated two-word and three-word phrases often reveal stable formulas, slogans, terms, or conceptual pairs.",
        "Top bigrams:",
    ]
    lines.extend(f"- {item.phrase}: {item.count}" for item in bundle.bigrams[:15])
    lines.append("Top trigrams:")
    lines.extend(f"- {item.phrase}: {item.count}" for item in bundle.trigrams[:15])

    lines.extend([
        "",
        "Sentiment and Stance Marker Analysis",
        "=" * 60,
        "Why: this does not replace interpretation; it flags where evaluative, critical, uncertain, or normative language is concentrated.",
        f"Positive/evaluative markers: {bundle.sentiment.positive_markers}",
        f"Negative/critical markers: {bundle.sentiment.negative_markers}",
        f"Uncertainty markers: {bundle.sentiment.uncertainty_markers}",
        f"Authority/normative markers: {bundle.sentiment.authority_markers}",
        f"Dominant detected tone: {bundle.sentiment.dominant_tone}",
        "",
        "Concept Dynamics Analysis",
        "=" * 60,
        "Why: dividing the corpus into segments shows whether the author's vocabulary and stance change over the text.",
    ])
    for segment in bundle.dynamics:
        terms = ", ".join(f"{term}:{count}" for term, count in segment.top_terms)
        lines.append(
            f"- Segment {segment.segment} ({segment.start_percent}-{segment.end_percent}%): "
            f"{terms}; tone={segment.sentiment.dominant_tone}"
        )

    lines.extend([
        "",
        "Semantic Analysis Engine",
        "=" * 60,
        "Why: identifies dominant concepts across the clean corpus and prepares the evidence layer.",
        "Key concepts:",
    ])
    for concept in bundle.concepts[:20]:
        source_list = ", ".join(concept.sources[:4])
        if len(concept.sources) > 4:
            source_list += ", ..."
        lines.append(f"- {concept.term}: {concept.count} [{source_list}]")

    lines.extend([
        "",
        "Citation & Evidence Engine",
        "=" * 60,
        "Why: each concept is linked to exact sentences so conclusions can be checked against the source.",
    ])
    for concept in bundle.concepts[:12]:
        lines.append(f"{concept.term} ({concept.count})")
        for item in concept.evidence:
            lines.append(
                f"- {item.source}, sentence {item.sentence_index}, score {item.score:.2f}: {item.text}"
            )
        lines.append("")

    lines.extend([
        "Semantic Paragraph Matching",
        "=" * 60,
        "Why: finds paragraphs closest to the dominant concept set using deterministic TF-IDF, without hallucinated summaries.",
    ])
    for item in bundle.semantic_matches:
        terms = ", ".join(item.top_terms[:6])
        lines.append(f"- {item.source}, paragraph {item.paragraph_index}, score {item.score:.3f}, terms: {terms}")
        lines.append(f"  {item.text}")

    lines.extend([
        "",
        "Knowledge Graph Engine",
        "=" * 60,
        "Why: co-occurrence edges show which concepts appear together in the same sentence and may form stable relationships.",
        "Top co-occurrences:",
    ])
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
