"""
Offline BM25 Knowledge Retriever (HNX26EPS08).
Pure Python in-memory BM25 over data/knowledge/*.md files.
Zero embeddings, zero vector DB, zero network. CPU and RAM efficient.
"""

import re
import math
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "do",
    "be", "as", "by", "from", "are", "was", "were", "has", "have",
    "had", "will", "would", "can", "could", "should", "may", "might",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "what", "how", "when", "where", "who", "which", "if", "then",
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [w for w in text.split() if w and w not in _STOP_WORDS and len(w) > 1]
    return tokens


# ---------------------------------------------------------------------------
# DOCUMENT LOADER
# ---------------------------------------------------------------------------

def _load_docs(knowledge_dir: Path) -> list[dict]:
    """
    Load all *.md files from knowledge_dir.
    Each file is split into sections by heading (## or ###).
    Returns list of {"id": str, "title": str, "text": str, "tokens": list[str]}.
    """
    docs = []
    md_files = sorted(knowledge_dir.glob("*.md"))

    for md_file in md_files:
        source = md_file.stem
        raw = md_file.read_text(encoding="utf-8", errors="replace")
        # Split on markdown headings
        sections = re.split(r"\n(?=#{1,4}\s)", raw)
        for section in sections:
            lines = section.strip().splitlines()
            if not lines:
                continue
            # Extract heading as title
            title_line = lines[0].strip()
            title = re.sub(r"^#{1,4}\s*", "", title_line)
            text = "\n".join(lines).strip()
            if len(text) < 30:
                continue
            docs.append({
                "id": f"{source}::{title}",
                "title": title,
                "text": text,
                "source": source,
                "tokens": _tokenize(text),
            })

    return docs


# ---------------------------------------------------------------------------
# BM25 RANKER
# ---------------------------------------------------------------------------

class BM25:
    """
    Standard BM25 implementation.
    k1=1.5 and b=0.75 are well-established defaults (Robertson & Zaragoza 2009).
    """

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.avgdl = sum(len(d["tokens"]) for d in docs) / max(self.N, 1)

        # Build inverted index: term -> {doc_idx: term_freq}
        self._index: dict[str, dict[int, int]] = {}
        for idx, doc in enumerate(docs):
            freq: dict[str, int] = {}
            for token in doc["tokens"]:
                freq[token] = freq.get(token, 0) + 1
            for term, tf in freq.items():
                if term not in self._index:
                    self._index[term] = {}
                self._index[term][idx] = tf

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self.docs[doc_idx]
        dl = len(doc["tokens"])
        score = 0.0
        freq: dict[str, int] = {}
        for t in doc["tokens"]:
            freq[t] = freq.get(t, 0) + 1

        for term in query_tokens:
            if term not in self._index:
                continue
            df = len(self._index[term])
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf = freq.get(term, 0)
            norm_tf = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
            score += idf * norm_tf

        return score

    def search(self, query: str, top_k: int = 1) -> list[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored = [
            (i, self.score(q_tokens, i))
            for i in range(self.N)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.docs[i] for i, s in scored[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# KNOWLEDGE BASE (Singleton)
# ---------------------------------------------------------------------------

_DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"


class KnowledgeBase:
    """
    Offline BM25 knowledge retriever for the EdgeVoice assistant.
    Call search() to get the most relevant passage for LLM context injection.
    """

    def __init__(self, knowledge_dir: Optional[Path] = None):
        self._dir = Path(knowledge_dir) if knowledge_dir else _DEFAULT_KNOWLEDGE_DIR
        self._docs = _load_docs(self._dir)
        self._bm25 = BM25(self._docs)
        print(
            f"[KnowledgeBase] Loaded {len(self._docs)} sections "
            f"from {len(list(self._dir.glob('*.md')))} files in '{self._dir}'"
        )

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def search(self, query: str, top_k: int = 1, max_chars: int = 600) -> str:
        """
        Returns the top-k retrieved passage(s) as a compact string
        ready for LLM context injection.

        Args:
            query:     User's natural language query.
            top_k:     Number of top sections to retrieve.
            max_chars: Truncate each section to this many characters.

        Returns:
            Formatted string with source and content, or "" if no match found.
        """
        results = self._bm25.search(query, top_k=top_k)
        if not results:
            return ""

        parts = []
        for r in results:
            text = r["text"][:max_chars].strip()
            parts.append(f"[Source: {r['source']} / {r['title']}]\n{text}")

        return "\n\n".join(parts)

    def has_relevant_content(self, query: str, min_score_threshold: float = 0.5) -> bool:
        """Returns True if there's a meaningful BM25 match for the query."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return False
        scored = [(i, self._bm25.score(q_tokens, i)) for i in range(self._bm25.N)]
        top_score = max((s for _, s in scored), default=0.0)
        return top_score >= min_score_threshold


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    kb = KnowledgeBase()
    print(f"\nTotal sections indexed: {kb.doc_count}\n")

    queries = [
        "how to treat a burn",
        "CPR for adult",
        "fever management",
        "choking baby infant",
        "dehydration symptoms",
        "random unrelated query about space travel",
    ]

    print("=== BM25 KNOWLEDGE RETRIEVAL TEST ===\n")
    for q in queries:
        relevant = kb.has_relevant_content(q)
        result = kb.search(q, top_k=1, max_chars=300)
        print(f"Query: '{q}'")
        print(f"Relevant: {relevant}")
        if result:
            print(f"Retrieved:\n{result[:250]}...")
        else:
            print("Retrieved: [No match — will not inject context]")
        print()
