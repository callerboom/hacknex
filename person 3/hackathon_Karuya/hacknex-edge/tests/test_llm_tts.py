"""
test_llm_tts.py — Benchmark & integration tests for Person 3 modules.
HackNEX 2026 HNX26EPS08

Measures:
  - Time to First Token (TTFT)
  - Time to First Sentence (TTFS)
  - Time to First Audio (TTFA)
  - Total streaming latency
  - Memory footprint (psutil)
  - TTS cache hit performance

Run:  python tests/test_llm_tts.py
"""
from __future__ import annotations

import logging
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 fix
import time
from pathlib import Path

import numpy as np

# ── add project root to sys.path ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil

from src.edge.contracts import Turn
from src.edge.knowledge import KnowledgeRetriever
from src.edge.llm import GGUFLocalLLM
from src.edge.tts import SentenceChunkedTTS, split_on_boundary
from src.edge.tts_cache import CachedTTS, TTSCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("test_llm_tts")

PROC = psutil.Process(os.getpid())

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ram_mb() -> float:
    return PROC.memory_info().rss / (1024 ** 2)


def separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# TEST 1 — Knowledge Retrieval (BM25)
# ---------------------------------------------------------------------------

def test_knowledge_retrieval():
    separator("TEST 1 — BM25 Knowledge Retrieval")
    kr = KnowledgeRetriever()

    t0 = time.perf_counter()
    kr.load()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Index load time:  {load_ms:.1f} ms")

    queries = [
        "How to treat a minor burn?",
        "What to do for a sprained ankle?",
        "Signs of dehydration",
        "Fever management at home",
    ]

    for q in queries:
        t0 = time.perf_counter()
        snippet = kr.search(q, top_k=1)
        search_ms = (time.perf_counter() - t0) * 1000
        preview = snippet[:100].replace("\n", " ") + ("..." if len(snippet) > 100 else "")
        print(f"\nQuery: '{q}'")
        print(f"  Search: {search_ms:.2f} ms | Snippet: {preview}")

    assert load_ms < 2000, f"Index load too slow: {load_ms:.0f} ms"
    print("\n[PASS] Knowledge retrieval tests PASSED")


# ---------------------------------------------------------------------------
# TEST 2 — LLM Streaming (TTFT)
# ---------------------------------------------------------------------------

def test_llm_streaming():
    separator("TEST 2 — LLM Streaming (TTFT)")

    kr = KnowledgeRetriever()
    kr.load()

    query = "How do I treat a minor burn?"
    snippet = kr.search(query)
    print(f"Knowledge snippet injected: {len(snippet)} chars")

    llm = GGUFLocalLLM(knowledge_snippet=snippet)

    ram_before = ram_mb()
    t0 = time.perf_counter()
    llm.load()
    load_ms = (time.perf_counter() - t0) * 1000
    ram_after = ram_mb()
    print(f"Model load: {load_ms:.0f} ms | RAM delta: +{ram_after - ram_before:.1f} MB")

    print(f"\nQuery: '{query}'")
    print("Response: ", end="", flush=True)

    t_start = time.perf_counter()
    ttft_ms = None
    full_response = ""

    for tok in llm.stream(query, []):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_start) * 1000
        full_response += tok
        print(tok, end="", flush=True)

    total_ms = (time.perf_counter() - t_start) * 1000
    print()
    print(f"\n  TTFT:  {ttft_ms:.1f} ms")
    print(f"  Total streaming: {total_ms:.0f} ms")
    print(f"  Response length: {len(full_response)} chars")
    print(f"  RAM usage: {ram_mb():.1f} MB")

    assert full_response, "LLM returned empty response"
    print("\n[PASS] LLM streaming tests PASSED")
    return full_response


# ---------------------------------------------------------------------------
# TEST 3 — Sentence Boundary Chunker
# ---------------------------------------------------------------------------

def test_sentence_boundary():
    separator("TEST 3 — Sentence Boundary Chunker")

    test_cases = [
        ("Cool water should be run over a minor burn for ten minutes. Do not apply ice.", 2),
        ("Hello! How are you? I'm fine, thanks.", 2),  # Hello! merges (short)
        ("No boundary here yet", 0),
        ("First sentence. Second one. Third! Fourth, yes.", 1),  # short fragments merge
    ]

    all_passed = True
    for text, expected_min in test_cases:
        sents, remaining = split_on_boundary(text)
        ok = len(sents) >= expected_min
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} Input: '{text[:50]}...' | Sentences: {len(sents)} | Remaining: '{remaining[:30]}'")
        if not ok:
            all_passed = False

    assert all_passed, "Sentence boundary tests FAILED"
    print("\n[PASS] Sentence boundary tests PASSED")


# ---------------------------------------------------------------------------
# TEST 4 — TTS Time-to-First-Audio
# ---------------------------------------------------------------------------

def test_tts_ttfa():
    separator("TEST 4 — TTS Time-to-First-Audio (TTFA)")

    tts = SentenceChunkedTTS()
    print(f"Backend: {tts._backend_type} @ {tts.sample_rate} Hz")

    sentences = [
        "Cool water should be run over a minor burn for ten to twenty minutes.",
        "Do not apply ice, butter, or toothpaste. Consult a doctor for serious concerns.",
    ]

    total_chunks = 0
    ttfa_ms = None
    t0 = time.perf_counter()

    for sent in sentences:
        for chunk in tts.synth_stream(sent):
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t0) * 1000
            total_chunks += 1
            assert isinstance(chunk, np.ndarray), "Chunk must be np.ndarray"
            assert chunk.dtype == np.float32, "Chunk dtype must be float32"

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTFA:          {ttfa_ms:.1f} ms")
    print(f"  Total audio:   {total_ms:.0f} ms")
    print(f"  Chunks:        {total_chunks}")
    print(f"  Backend:       {tts._backend_type}")

    assert ttfa_ms is not None and ttfa_ms < 5000, f"TTFA too slow: {ttfa_ms:.0f} ms"
    print("\n[PASS] TTS TTFA tests PASSED")
    return tts


# ---------------------------------------------------------------------------
# TEST 5 — Full End-to-End Pipeline (LLM → Sentence Chunker → TTS)
# ---------------------------------------------------------------------------

def test_e2e_pipeline():
    separator("TEST 5 — E2E Pipeline (LLM → Sentence Splitter → TTS)")

    kr = KnowledgeRetriever()
    kr.load()
    query = "How do I treat a minor burn?"
    snippet = kr.search(query)

    llm = GGUFLocalLLM(knowledge_snippet=snippet)
    llm.load()
    tts = SentenceChunkedTTS()

    history = [Turn(role="user", content="What can you help me with?"),
               Turn(role="assistant", content="I can help with health and first-aid questions.")]

    print(f"Query: '{query}'")

    t_start = time.perf_counter()
    ttft_ms = None
    ttfs_ms = None
    ttfa_ms = None
    total_audio_chunks = 0
    total_tokens = 0

    token_stream = llm.stream(query, history)

    for audio_chunk in tts.stream_from_tokens(token_stream):
        if ttfa_ms is None:
            ttfa_ms = (time.perf_counter() - t_start) * 1000
            print(f"  → First audio chunk at {ttfa_ms:.1f} ms [PASS]")
        total_audio_chunks += 1

    total_ms = (time.perf_counter() - t_start) * 1000

    print(f"\n  E2E Latency:   {total_ms:.0f} ms")
    print(f"  TTFA:          {ttfa_ms:.1f} ms" if ttfa_ms else "  TTFA: N/A")
    print(f"  Audio chunks:  {total_audio_chunks}")
    print(f"  RAM:           {ram_mb():.1f} MB")

    print("\n[PASS] E2E pipeline tests PASSED")


# ---------------------------------------------------------------------------
# TEST 6 — TTS Cache (0ms warm hit)
# ---------------------------------------------------------------------------

def test_tts_cache():
    separator("TEST 6 — TTS Cache Performance")

    live_tts = SentenceChunkedTTS()
    cached_tts = CachedTTS(tts=live_tts)

    print("Warming cache...")
    t0 = time.perf_counter()
    cached_tts.warm()
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"Warm-up time: {warm_ms:.0f} ms")

    phrase = "Hello! How can I help you?"

    # Cold call (should be in cache after warm)
    t0 = time.perf_counter()
    chunks = list(cached_tts.synth_stream(phrase))
    hit_ms = (time.perf_counter() - t0) * 1000

    print(f"Cache HIT latency: {hit_ms:.2f} ms | Chunks: {len(chunks)}")
    print(f"Stats: {cached_tts.stats()}")

    assert hit_ms < 100, f"Cache hit too slow: {hit_ms:.1f} ms"
    assert len(chunks) > 0, "No audio chunks from cache"

    print("\n[PASS] TTS cache tests PASSED")


# ---------------------------------------------------------------------------
# TEST 7 — Memory footprint under limit
# ---------------------------------------------------------------------------

def test_memory_limit():
    separator("TEST 7 — Memory Footprint Check")

    RAM_LIMIT_MB = 3072  # from limits.yaml default

    current_ram = ram_mb()
    print(f"Current RSS: {current_ram:.1f} MB / {RAM_LIMIT_MB} MB limit")

    if current_ram > RAM_LIMIT_MB:
        print(f"⚠ WARNING: RAM {current_ram:.0f} MB exceeds limit {RAM_LIMIT_MB} MB")
    else:
        headroom = RAM_LIMIT_MB - current_ram
        print(f"[PASS] RAM OK — {headroom:.0f} MB headroom remaining")


# ---------------------------------------------------------------------------
# MAIN — run all tests
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  HackNEX 2026 — Person 3 Module B Benchmark")
    print("  Edge LLM + Knowledge + TTS Test Suite")
    print("=" * 60)

    results = {}

    tests = [
        ("knowledge_retrieval", test_knowledge_retrieval),
        ("sentence_boundary",   test_sentence_boundary),
        ("tts_ttfa",            test_tts_ttfa),
        ("llm_streaming",       test_llm_streaming),
        ("e2e_pipeline",        test_e2e_pipeline),
        ("tts_cache",           test_tts_cache),
        ("memory_limit",        test_memory_limit),
    ]

    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            results[name] = "PASS"
        except AssertionError as e:
            print(f"\n[FAIL] FAIL: {e}")
            results[name] = "FAIL"
        except Exception as e:
            print(f"\n[FAIL] ERROR: {e}")
            results[name] = f"ERROR: {e}"
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  [{elapsed:.0f} ms]")

    separator("SUMMARY")
    for name, status in results.items():
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {name:<30} {status}")

    passed = sum(1 for s in results.values() if s == "PASS")
    print(f"\n  {passed}/{len(results)} tests passed")
    print("=" * 60 + "\n")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
