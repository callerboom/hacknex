"""
knowledge.py — In-memory BM25 knowledge retrieval over data/knowledge/*.md
Person 3 — HackNEX 2026 HNX26EPS08

Zero vector DB, zero heavy embeddings. Pure Python BM25 over Markdown files.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "tell", "explain", "describe", "know", "mean", "help", "show", "find", "give", "work",
}

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, remove common stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in _STOPWORDS]
    return tokens if tokens else [t for t in text.split() if len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Lightweight BM25 Okapi implementation.
    k1=1.5, b=0.75 (standard defaults).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[str] = []          # raw text of each chunk
        self._tokens: List[List[str]] = []  # tokenized chunks
        self._df: Counter = Counter()        # document freq per term
        self._idf: dict[str, float] = {}    # precomputed IDF
        self._avgdl: float = 0.0
        self._n: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add_document(self, text: str) -> None:
        tokens = _tokenize(text)
        self._docs.append(text)
        self._tokens.append(tokens)
        for term in set(tokens):
            self._df[term] += 1
        self._n += 1

    def build(self) -> None:
        """Precompute IDF and average document length."""
        total_len = sum(len(t) for t in self._tokens)
        self._avgdl = total_len / self._n if self._n else 1.0
        for term, df in self._df.items():
            self._idf[term] = math.log(
                (self._n - df + 0.5) / (df + 0.5) + 1.0
            )
        logger.info(
            "BM25 index built: %d docs, %d unique terms, avgdl=%.1f",
            self._n,
            len(self._idf),
            self._avgdl,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def scores(self, query: str) -> List[Tuple[int, float]]:
        """Return list of (doc_index, bm25_score) sorted descending."""
        q_tokens = _tokenize(query)
        results: List[Tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._tokens):
            tf = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for term in q_tokens:
                idf = self._idf.get(term, 0.0)
                freq = tf.get(term, 0)
                norm = (
                    freq * (self.k1 + 1)
                    / (freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                )
                score += idf * norm
            results.append((idx, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_doc(self, idx: int) -> str:
        return self._docs[idx]


# ---------------------------------------------------------------------------
# Knowledge Retriever (public API)
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """
    Public interface for Person 3.
    Loads all *.md files under data/knowledge/, chunks them, builds BM25.
    """

    CHUNK_SENTENCES = 4   # sentences per chunk
    MAX_SNIPPET_CHARS = 600

    def __init__(self, knowledge_dir: str | Path | None = None):
        if knowledge_dir is None:
            # Resolve relative to this file's repo root (hacknex-edge/)
            # src/edge/knowledge.py -> parents[0]=edge, [1]=src, [2]=hacknex-edge
            here = Path(__file__).resolve()
            knowledge_dir = here.parents[2] / "data" / "knowledge"
        self._dir = Path(knowledge_dir)
        self._index = BM25Index()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        md_files = sorted(self._dir.glob("*.md"))
        if not md_files:
            logger.warning("No *.md files found in %s", self._dir)
            self._index.add_document("No offline knowledge loaded.")
            self._index.build()
            self._loaded = True
            return

        for path in md_files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._chunk(text)
            for chunk in chunks:
                self._index.add_document(chunk)
            logger.info("Loaded %d chunks from %s", len(chunks), path.name)

        self._index.build()
        self._loaded = True

    def _chunk(self, text: str) -> List[str]:
        """Split markdown text into semantic section / paragraph chunks."""
        raw_sections = re.split(r"\n{2,}|(?=^##+\s)", text, flags=re.MULTILINE)
        chunks = []
        for sec in raw_sections:
            sec = sec.strip()
            if not sec or len(sec) < 30:
                continue
            if len(sec) > 1200:
                parts = [p.strip() for p in sec.split("\n\n") if len(p.strip()) >= 30]
                chunks.extend(parts if parts else [sec[:1200]])
            else:
                chunks.append(sec)
        return chunks or [text[:500]]

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 1) -> str:
        """
        Returns the top matching text snippet to inject into LLM system prompt.
        Compatible with contracts.py expectations.
        """
        if not self._loaded:
            self.load()

        ranked = self._index.scores(query)
        snippets = []
        for idx, score in ranked[:top_k]:
            if score >= 1.5:
                doc = self._index.get_doc(idx)
                snippets.append(doc[: self.MAX_SNIPPET_CHARS])

        result = "\n---\n".join(snippets) if snippets else ""
        logger.debug("BM25 search '%s' -> score=%.3f", query, ranked[0][1] if ranked else 0)
        return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kr = KnowledgeRetriever()
    kr.load()
    q = "How to treat a minor burn?"
    snippet = kr.search(q)
    print(f"Query: {q}")
    print(f"Top snippet:\n{snippet}")
