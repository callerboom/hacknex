"""
TTS Cache: Pre-synthesized audio for common replies (HNX26EPS08).
Keyed by normalized text. Provides 0ms synthesis delay on cache hits.
Reports hit/miss rate to dashboard.
"""

import hashlib
import threading
import time
from pathlib import Path
from typing import Iterator, Optional
import numpy as np

from src.edge.audio_io import save_wav, load_wav

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tts_cache"

# Common phrases to pre-synthesize at startup
COMMON_PHRASES = [
    "Hello! I'm Jarvis Edge, your offline AI assistant. How can I help you?",
    "I didn't catch that. Could you please say that again?",
    "I'm sorry, I don't know the answer to that.",
    "Let me think about that.",
    "Here's what I found.",
    "Not professional medical advice — please consult a doctor.",
    "Opening Notepad.",
    "Opening Calculator.",
    "Done.",
    "Battery is charging.",
    "Audio muted.",
    "Volume increased.",
    "Volume decreased.",
]


def _normalize(text: str) -> str:
    """Normalize text for consistent cache keying."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _cache_key(text: str) -> str:
    return hashlib.md5(_normalize(text).encode()).hexdigest()[:16]


class TTSCache:
    """
    Disk + in-memory LRU cache for pre-synthesized TTS audio.
    On cache hit: streams audio from RAM/disk with 0ms synthesis delay.
    On cache miss: synthesizes with TTS engine and stores result.
    """

    def __init__(
        self,
        tts_engine=None,
        cache_dir: Optional[Path] = None,
        max_memory_items: int = 64,
        sample_rate: int = 22050,
    ):
        self._tts = tts_engine
        self._cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._max_mem = max_memory_items

        # In-memory cache: key -> np.ndarray
        self._mem: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

        # Stats
        self._hits = 0
        self._misses = 0

    def _mem_get(self, key: str) -> Optional[np.ndarray]:
        return self._mem.get(key)

    def _mem_put(self, key: str, audio: np.ndarray) -> None:
        if len(self._mem) >= self._max_mem:
            # Evict first item (simple FIFO)
            oldest = next(iter(self._mem))
            del self._mem[oldest]
        self._mem[key] = audio

    def _disk_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.wav"

    def _disk_get(self, key: str) -> Optional[np.ndarray]:
        p = self._disk_path(key)
        if p.exists():
            try:
                audio, _ = load_wav(p)
                return audio
            except Exception:
                return None
        return None

    def _disk_put(self, key: str, audio: np.ndarray) -> None:
        try:
            save_wav(self._disk_path(key), audio, self._sample_rate)
        except Exception as e:
            print(f"[TTSCache] Disk write error: {e}")

    def get_or_synth(self, text: str) -> Iterator[np.ndarray]:
        """
        Return audio chunks for text.
        Tries memory -> disk -> synthesize (in that order).
        """
        key = _cache_key(text)

        with self._lock:
            # 1. Memory hit
            audio = self._mem_get(key)
            if audio is not None:
                self._hits += 1
                yield from self._iter_chunks(audio)
                return

        # 2. Disk hit
        audio = self._disk_get(key)
        if audio is not None:
            with self._lock:
                self._hits += 1
                self._mem_put(key, audio)
            yield from self._iter_chunks(audio)
            return

        # 3. Cache miss — synthesize
        with self._lock:
            self._misses += 1

        if self._tts is None:
            return

        chunks = []
        for chunk in self._tts.synth_stream(text):
            chunks.append(chunk)
            yield chunk

        if chunks:
            audio = np.concatenate(chunks)
            with self._lock:
                self._mem_put(key, audio)
            self._disk_put(key, audio)

    def _iter_chunks(self, audio: np.ndarray, chunk_size: int = 2048) -> Iterator[np.ndarray]:
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]

    def prebuild(self, phrases: Optional[list[str]] = None) -> None:
        """
        Pre-synthesize a list of common phrases and store to disk+memory.
        Call this at startup (or from scripts/download_models.py).
        """
        if self._tts is None:
            print("[TTSCache] No TTS engine — skipping prebuild.")
            return

        phrases = phrases or COMMON_PHRASES
        print(f"[TTSCache] Pre-building cache for {len(phrases)} phrases ...")
        built = 0
        for text in phrases:
            key = _cache_key(text)
            if self._disk_path(key).exists():
                continue
            try:
                chunks = list(self._tts.synth_stream(text))
                if chunks:
                    audio = np.concatenate(chunks)
                    self._disk_put(key, audio)
                    with self._lock:
                        self._mem_put(key, audio)
                    built += 1
            except Exception as e:
                print(f"[TTSCache] prebuild error for '{text[:40]}': {e}")
        print(f"[TTSCache] Pre-built {built} new phrases. Cache ready.")

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
            "memory_items": len(self._mem),
            "disk_items": len(list(self._cache_dir.glob("*.wav"))),
        }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("[TTSCache] Testing cache with no TTS engine (disk only) ...")
    cache = TTSCache(tts_engine=None, sample_rate=22050)

    # Test with a phrase that doesn't exist
    result = list(cache.get_or_synth("Hello world"))
    print(f"  Miss (no engine): {len(result)} chunks returned")
    print(f"  Stats: {cache.stats}")
    print("[TTSCache] tts_cache.py OK")
