"""tts_cache.py — Pre-synthesized audio cache for common phrases (HNX26EPS08).
Integrated for HackNEX 2026.
On cache hit: returns audio with 0ms synthesis delay.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Dict, Union
import numpy as np

from src.edge.audio_io import load_wav, save_wav

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tts_cache"

COMMON_PHRASES = [
    "Hello! How can I help you?",
    "Hello! How can I assist you today?",
    "I didn't catch that. Could you please repeat?",
    "I didn't catch that. Could you repeat?",
    "Let me look that up for you.",
    "Please wait a moment.",
    "Opening Notepad.",
    "Opening Calculator.",
    "Audio muted.",
    "Increasing system volume.",
    "Decreasing system volume.",
    "Not professional medical advice — please consult a doctor.",
]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _cache_key(text: str) -> str:
    return hashlib.md5(_normalize(text).encode("utf-8")).hexdigest()[:16]


class TTSCache:
    """Two-level cache: RAM dictionary + disk WAV files."""

    def __init__(
        self,
        tts_engine=None,
        cache_dir: Optional[Union[str, Path]] = None,
        sample_rate: int = 22050,
        tts: Optional[Any] = None,
    ):
        self._tts = tts_engine or tts
        self._dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

        self._mem: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        # Load existing disk files into memory
        self._load_existing()

    def _load_existing(self):
        if not self._dir.exists():
            return
        for f in self._dir.glob("*.wav"):
            try:
                audio = load_wav(f, target_sr=self.sample_rate)
                if len(audio) > 0:
                    self._mem[f.stem] = audio
            except Exception:
                pass

    def prebuild(self):
        """Pre-synthesizes common phrases if engine is available."""
        if not self._tts:
            return
        for phrase in COMMON_PHRASES:
            key = _cache_key(phrase)
            if key not in self._mem:
                try:
                    chunks = list(self._tts.synth_stream(phrase))
                    if chunks:
                        audio = np.concatenate(chunks)
                        self._mem[key] = audio
                        save_wav(self._dir / f"{key}.wav", audio, sample_rate=self.sample_rate)
                except Exception:
                    pass

    def warm(self):
        """Alias for prebuild()."""
        self.prebuild()

    def get_or_synth(self, text: str) -> Iterator[np.ndarray]:
        """Fetch cached audio or synthesize on demand."""
        key = _cache_key(text)
        with self._lock:
            if key in self._mem:
                self._hits += 1
                audio = self._mem[key]
                chunk_size = int(self.sample_rate * 0.020)
                for i in range(0, len(audio), chunk_size):
                    yield audio[i : i + chunk_size]
                return

            self._misses += 1

        if self._tts:
            synth_chunks = []
            for chunk in self._tts.synth_stream(text):
                synth_chunks.append(chunk)
                yield chunk

            if synth_chunks:
                full_audio = np.concatenate(synth_chunks)
                with self._lock:
                    self._mem[key] = full_audio
                    save_wav(self._dir / f"{key}.wav", full_audio, sample_rate=self.sample_rate)

    def synth_stream(self, text: str) -> Iterator[np.ndarray]:
        """Alias for get_or_synth()."""
        return self.get_or_synth(text)

    def stats(self) -> dict:
        total = self._hits + self._misses
        rate = (self._hits / total * 100) if total else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(rate, 1),
            "cached_phrases": len(self._mem),
        }


class CachedTTS(TTSCache):
    """Wrapper class for Person 3 test suite compatibility."""
    def __init__(self, tts=None, cache=None):
        engine = tts or (cache._tts if cache else None)
        sample_rate = getattr(engine, "sample_rate", 22050)
        super().__init__(tts_engine=engine, sample_rate=sample_rate)
