"""Lexical and semantic retrieval over documents with traceable sources."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


# Keep whitespace-delimited words intact. Segment CJK blocks into characters
# and neighboring pairs so unsegmented text can match a phrase inside a passage.
_CJK = (
    "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7a3\uf900-\ufaff"
    "\U00020000-\U000323af"
)
_TOKEN_PATTERN = re.compile(fr"[{_CJK}]+|[^\W_{_CJK}]+")
_CJK_START = re.compile(fr"[{_CJK}]")


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float


def _terms(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    for token in _TOKEN_PATTERN.findall(text):
        if not _CJK_START.match(token):
            yield token
            continue
        yield from token
        for index in range(len(token) - 1):
            yield token[index:index + 2]


def _validate_scores(scores, expected_count, name):
    if len(scores) != expected_count or not all(
        type(score) in (int, float) and math.isfinite(score) for score in scores
    ):
        raise ValueError(f"{name} must return one finite number for each input.")


def _unit_vectors(vectors, expected_count):
    if len(vectors) != expected_count:
        raise ValueError("The embedding provider returned the wrong number of vectors.")
    dimensions = len(vectors[0])
    if not dimensions or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Embedding vectors must have matching, nonzero dimensions.")
    normalized = []
    for vector in vectors:
        _validate_scores(vector, dimensions, "The embedding provider")
        norm = math.hypot(*vector)
        if not norm or not math.isfinite(norm):
            raise ValueError("Embedding vectors must have a finite, nonzero norm.")
        normalized.append(tuple(value / norm for value in vector))
    return normalized


class RetrievalIndex:
    """Rank with BM25, RRF of BM25 and cosine similarity, or an added reranker.

    embed(texts) returns one vector per text. Document vectors are computed
    lazily and reused for the life of this index. rerank(query, documents)
    returns one score per document in input order, where higher is better.
    """

    def __init__(self, documents, *, embed=None):
        self.documents = tuple(documents)
        if len({doc.id for doc in self.documents}) != len(self.documents):
            raise ValueError("Document IDs must be unique.")
        self._terms = [Counter(_terms(doc.text)) for doc in self.documents]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._mean_length = sum(self._lengths) / max(len(self.documents), 1)
        self._document_frequency = Counter(term for terms in self._terms for term in terms)
        self._embed = embed
        self._vectors = None

    def _bm25(self, query):
        query_terms = tuple(dict.fromkeys(_terms(query)))
        if not self._mean_length:
            return []
        k1, b = 1.2, 0.75
        idfs = {
            term: math.log1p((len(self.documents) - count + 0.5) / (count + 0.5))
            for term in query_terms
            if (count := self._document_frequency[term])
        }
        hits = []
        for doc, terms, length in zip(self.documents, self._terms, self._lengths):
            length_factor = k1 * (1 - b + b * length / self._mean_length)
            score = sum(
                idf * (terms[term] * (k1 + 1)) / (terms[term] + length_factor)
                for term, idf in idfs.items()
                if terms[term]
            )
            if score:
                hits.append(SearchHit(doc, score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def _semantic(self, query):
        if self._vectors is None:
            texts = [doc.text for doc in self.documents]
            self._vectors = _unit_vectors(self._embed(texts), len(texts))
        query_vector, = _unit_vectors(self._embed([query]), 1)
        if len(query_vector) != len(self._vectors[0]):
            raise ValueError("Query and document embeddings must have matching dimensions.")
        hits = (
            SearchHit(doc, math.fsum(a * b for a, b in zip(query_vector, vector)))
            for doc, vector in zip(self.documents, self._vectors)
        )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def rank(self, query, *, method="bm25", limit=5, rerank=None):
        if method not in ("bm25", "hybrid", "reranked"):
            raise ValueError(f"Unknown retrieval method: {method}")
        if limit < 0:
            raise ValueError("Retrieval limit cannot be negative.")
        if method != "bm25" and self._embed is None:
            raise ValueError("Hybrid retrieval requires a configured embedding provider.")
        if method == "reranked" and rerank is None:
            raise ValueError("Reranked retrieval requires a configured reranker.")
        if limit == 0 or not self.documents or not query.strip():
            return []

        lexical = self._bm25(query)
        if method == "bm25":
            return lexical[:limit]

        # Reciprocal rank fusion depends on rank rather than incomparable raw scores.
        scores = Counter()
        for ranking in (lexical, self._semantic(query)):
            for position, hit in enumerate(ranking, 1):
                scores[hit.document.id] += 1 / (60 + position)
        fused = sorted(
            (SearchHit(doc, scores[doc.id]) for doc in self.documents),
            key=lambda hit: hit.score,
            reverse=True,
        )
        if method == "hybrid":
            return fused[:limit]

        candidates = [hit.document for hit in fused[:max(limit, 20)]]
        reranked_scores = rerank(query, candidates)
        _validate_scores(reranked_scores, len(candidates), "The reranker")
        return sorted(
            (SearchHit(doc, score) for doc, score in zip(candidates, reranked_scores)),
            key=lambda hit: hit.score,
            reverse=True,
        )[:limit]


def build_context(hits, *, max_chars=4000):
    """Format ranked passages with provenance within a character budget."""
    sections = []
    remaining = max_chars
    for hit in hits:
        doc = hit.document
        source = " ".join(doc.source.split())
        header = f"[{doc.id}]"
        if source:
            header += f" {source}"
        header += "\n"
        text = doc.text.strip()
        space_for_text = remaining - len(header) - (2 if sections else 0)
        if space_for_text < 1:
            break
        if len(text) > space_for_text:
            sections.append(header + text[:space_for_text - 1].rstrip() + "…")
            break
        sections.append(header + text)
        remaining = space_for_text - len(text)
    return "\n\n".join(sections)
