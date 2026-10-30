"""
tts_cache.py — Pre-synthesized audio cache for common phrases.
Person 3 — HackNEX 2026 HNX26EPS08

On cache HIT  → stream audio with 0 ms synthesis delay.
On cache MISS → synthesize via tts.py and store for future calls.

Storage: disk-backed WAV files under <repo>/cache/tts_cache/
Runtime: also kept in RAM (numpy arrays) for instant replay.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phrases to pre-synthesize at startup
# ---------------------------------------------------------------------------

PRESYNTH_PHRASES: list[str] = [
    "Hello! How can I help you?",
    "I didn't catch that. Could you please repeat?",
    "Let me look that up for you.",
    "I'm sorry, I don't have information on that.",
    "Please wait a moment.",
    "Your request is being processed.",
    "Goodbye! Have a great day.",
    "I'm Edra, your offline assistant.",
    "That's a great question!",
    "I'm not sure about that. Consult a medical professional.",
]


def _text_key(text: str) -> str:
    """Create a stable cache key from the text."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:16]


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to WAV bytes."""
    pcm_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _wav_bytes_to_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load WAV bytes into float32 numpy array."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf) as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sample_rate


# ---------------------------------------------------------------------------
# TTSCache
# ---------------------------------------------------------------------------

class TTSCache:
    """
    Two-level cache: RAM dict + disk WAV files.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        sample_rate: int = 22050,
        chunk_duration_ms: float = 10.0,
    ):
        if cache_dir is None:
            here = Path(__file__).resolve()
            cache_dir = here.parents[3] / "cache" / "tts_cache"
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = sample_rate
        self._chunk_size = int(sample_rate * chunk_duration_ms / 1000)

        # RAM cache: key -> np.ndarray
        self._ram: dict[str, np.ndarray] = {}

        # Statistics
        self._hits = 0
        self._misses = 0

        # Load existing disk cache into RAM
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _disk_path(self, key: str) -> Path:
        return self._dir / f"{key}.wav"

    def _load_from_disk(self) -> None:
        loaded = 0
        for wav_path in self._dir.glob("*.wav"):
            key = wav_path.stem
            try:
                audio, sr = _wav_bytes_to_audio(wav_path.read_bytes())
                self._ram[key] = audio
                loaded += 1
            except Exception as e:
                logger.warning("Failed to load cached WAV %s: %s", wav_path, e)
        if loaded:
            logger.info("TTSCache: loaded %d cached audio files from disk.", loaded)

    def _save_to_disk(self, key: str, audio: np.ndarray) -> None:
        try:
            wav_bytes = _audio_to_wav_bytes(audio, self.sample_rate)
            self._disk_path(key).write_bytes(wav_bytes)
        except Exception as e:
            logger.warning("Failed to save TTS cache to disk: %s", e)

    # ------------------------------------------------------------------
    # Core get/set
    # ------------------------------------------------------------------

    def get(self, text: str) -> np.ndarray | None:
        key = _text_key(text)
        audio = self._ram.get(key)
        if audio is not None:
            self._hits += 1
            return audio

        # Try disk (in case it was added by another process)
        disk = self._disk_path(key)
        if disk.exists():
            try:
                audio, _ = _wav_bytes_to_audio(disk.read_bytes())
                self._ram[key] = audio
                self._hits += 1
                return audio
            except Exception:
                pass

        self._misses += 1
        return None

    def put(self, text: str, audio: np.ndarray) -> None:
        key = _text_key(text)
        self._ram[key] = audio
        self._save_to_disk(key, audio)

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def stream(self, text: str) -> Iterator[np.ndarray] | None:
        """
        Returns an iterator of audio chunks if cache hit, else None.
        Caller should fall through to live TTS on None.
        """
        audio = self.get(text)
        if audio is None:
            return None
        return self._chunked(audio)

    def _chunked(self, audio: np.ndarray) -> Iterator[np.ndarray]:
        for i in range(0, len(audio), self._chunk_size):
            yield audio[i : i + self._chunk_size]

    # ------------------------------------------------------------------
    # Pre-synthesis at startup
    # ------------------------------------------------------------------

    def warm(self, tts) -> None:
        """
        Pre-synthesize all PRESYNTH_PHRASES using the provided TTS object.
        Call once at startup (in a background thread if needed).
        """
        newly_synth = 0
        t0 = time.perf_counter()
        for phrase in PRESYNTH_PHRASES:
            if self.get(phrase) is not None:
                continue  # already cached
            try:
                chunks = list(tts.synth_stream(phrase))
                if chunks:
                    audio = np.concatenate(chunks)
                    self.put(phrase, audio)
                    newly_synth += 1
                    logger.debug("Pre-synthesized: '%s'", phrase[:40])
            except Exception as e:
                logger.warning("Failed to pre-synth '%s': %s", phrase[:30], e)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "TTS cache warmed: %d new, %d total - %.0f ms",
            newly_synth,
            len(self._ram),
            elapsed,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "cached_phrases": len(self._ram),
            "cache_dir": str(self._dir),
        }


# ---------------------------------------------------------------------------
# CachedTTS — Combines TTSCache + live SentenceChunkedTTS
# ---------------------------------------------------------------------------

class CachedTTS:
    """
    Drop-in TTS wrapper that checks cache first, falls back to live synthesis.
    Satisfies the TTS protocol from contracts.py.
    """

    def __init__(self, tts=None, cache: TTSCache | None = None):
        # Lazy import to avoid circular imports
        if tts is None:
            from src.edge.tts import SentenceChunkedTTS
            tts = SentenceChunkedTTS()
        self._tts = tts
        self._cache = cache or TTSCache(sample_rate=tts.sample_rate)
        self.sample_rate = tts.sample_rate

    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]:
        """Check cache first; fall through to live TTS."""
        cached = self._cache.stream(sentence)
        if cached is not None:
            logger.debug("TTS cache HIT: '%s'", sentence[:40])
            yield from cached
        else:
            logger.debug("TTS cache MISS: '%s'", sentence[:40])
            chunks = list(self._tts.synth_stream(sentence))
            if chunks:
                audio = np.concatenate(chunks)
                self._cache.put(sentence, audio)
                yield from self._cache._chunked(audio)

    def warm(self) -> None:
        """Pre-synthesize common phrases."""
        self._cache.warm(self._tts)

    def stats(self) -> dict:
        return self._cache.stats()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.edge.tts import SentenceChunkedTTS

    live_tts = SentenceChunkedTTS()
    cache = TTSCache(sample_rate=live_tts.sample_rate)

    print("Warming cache with common phrases...")
    cache.warm(live_tts)
    print(f"Cache stats: {cache.stats()}")

    # Test cache hit
    phrase = "Hello! How can I help you?"
    t0 = time.perf_counter()
    it = cache.stream(phrase)
    if it:
        chunks = list(it)
        print(f"Cache HIT - {len(chunks)} chunks in {(time.perf_counter()-t0)*1000:.2f} ms")
    else:
        print("Cache MISS (unexpected)")

    # Test miss + store
    novel = "This is a novel phrase not in the warm-up list."
    t0 = time.perf_counter()
    it = cache.stream(novel)
    print(f"Novel phrase cache: {'HIT' if it else 'MISS (will synthesize live)'}")
    print(f"Final stats: {cache.stats()}")
