"""
tts.py — Sentence-chunked streaming Text-to-Speech (TTS).
Person 3 — HackNEX 2026 HNX26EPS08

Implements TTS protocol from contracts.py:
    sample_rate: int
    synth_stream(sentence: str) -> Iterator[np.ndarray]

Pipeline:
  1. Receives streaming tokens from LLM.
  2. Buffers until sentence boundary (. ? ! ,)
  3. Immediately synthesizes each sentence chunk.
  4. Yields float32 np.ndarray audio blocks (<150ms to first audio).

Backends (in order of preference):
  A. sherpa_onnx.OfflineTts  (CPU-optimized VITS/Kokoro)
  B. piper_tts               (Piper ONNX)
  C. NumPy sine-wave stub    (fallback for testing without models)
"""
from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Iterator, List

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence Boundary Splitter
# ---------------------------------------------------------------------------

MIN_CHARS = 15   # Minimum chars to flush as a sentence

# Regex: match end-of-sentence punctuation followed by space OR end-of-string
_SENT_RE = re.compile(r"([^.!?]*[.!?]+)\s*")


def split_on_boundary(buffer: str) -> tuple[list[str], str]:
    """
    Split buffer at sentence boundaries (.  !  ?) returning completed
    sentences and any trailing fragment.
    Returns (sentences_ready, remaining_buffer).
    """
    sentences: list[str] = []
    pos = 0
    for m in _SENT_RE.finditer(buffer):
        sent = m.group(1).strip()
        if sent and len(sent) >= MIN_CHARS:
            sentences.append(sent)
        elif sent:
            # Too short — merge into next chunk
            if sentences:
                sentences[-1] += " " + sent
            else:
                sentences.append(sent)
        pos = m.end()
    remaining = buffer[pos:].strip()
    return sentences, remaining


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _load_models_cfg() -> dict:
    here = Path(__file__).resolve()
    cfg_path = here.parents[2] / "config" / "models.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _try_sherpa_onnx(cfg: dict) -> "object | None":
    """Try to load sherpa_onnx.OfflineTts."""
    try:
        import sherpa_onnx  # type: ignore

        tts_cfg = cfg.get("tts", {}).get("sherpa_onnx", {})
        model_dir = tts_cfg.get("model_dir", "")
        if not model_dir or not Path(model_dir).exists():
            logger.warning("sherpa_onnx TTS model_dir not found: %s", model_dir)
            return None

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(Path(model_dir) / "model.onnx"),
                    lexicon=str(Path(model_dir) / "lexicon.txt"),
                    tokens=str(Path(model_dir) / "tokens.txt"),
                    data_dir="",
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                ),
                num_threads=2,
                provider="cpu",
            ),
            rule_fsts="",
            max_num_sentences=1,
        )
        tts = sherpa_onnx.OfflineTts(tts_config)
        logger.info("sherpa_onnx TTS backend loaded.")
        return tts
    except Exception as e:
        logger.debug("sherpa_onnx TTS load failed: %s", e)
        return None


def _try_piper(cfg: dict):
    """Try to load piper-tts backend."""
    try:
        from piper import PiperVoice  # type: ignore

        tts_cfg = cfg.get("tts", {}).get("piper", {})
        model_path = tts_cfg.get("model", "")
        config_path = tts_cfg.get("config", model_path + ".json" if model_path else "")

        if not model_path or not Path(model_path).exists():
            logger.warning("Piper model not found: %s", model_path)
            return None

        voice = PiperVoice.load(model_path, config_path=config_path or None, use_cuda=False)
        logger.info("Piper TTS backend loaded.")
        return voice
    except Exception as e:
        logger.debug("Piper TTS load failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Stub backend (sine wave — for testing)
# ---------------------------------------------------------------------------

class _SineWaveStub:
    """Generates a short sine-wave tone per sentence (testing only)."""

    sample_rate = 22050

    def synthesize(self, text: str, sid: int = 0):
        logger.warning("Using SINE WAVE STUB TTS - no real TTS model loaded.")
        duration = min(0.5 + len(text) * 0.02, 3.0)
        t = np.linspace(0, duration, int(self.sample_rate * duration), dtype=np.float32)
        audio = 0.15 * np.sin(2 * math.pi * 440 * t)
        # Return object with .samples attribute (mimics sherpa_onnx)
        class _R:
            samples = audio
        return _R()


# ---------------------------------------------------------------------------
# TTS Main Class
# ---------------------------------------------------------------------------

class SentenceChunkedTTS:
    """
    Streaming sentence-chunked TTS.
    Satisfies the TTS protocol from contracts.py.
    """

    def __init__(self, sample_rate: int | None = None):
        cfg = _load_models_cfg()

        # Try backends
        self._backend_type = "none"
        self._sherpa = _try_sherpa_onnx(cfg)
        self._piper = None
        self._stub = None

        if self._sherpa is not None:
            self._backend_type = "sherpa_onnx"
            self.sample_rate = 22050
        else:
            self._piper = _try_piper(cfg)
            if self._piper is not None:
                self._backend_type = "piper"
                self.sample_rate = 22050
            else:
                self._stub = _SineWaveStub()
                self._backend_type = "stub"
                self.sample_rate = _SineWaveStub.sample_rate

        if sample_rate is not None:
            self.sample_rate = sample_rate

        logger.info("TTS backend: %s @ %d Hz", self._backend_type, self.sample_rate)

    # ------------------------------------------------------------------
    # Core synth_stream — Protocol implementation
    # ------------------------------------------------------------------

    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]:
        """
        Synthesize a single sentence and yield audio chunk(s).
        Designed to be called once per sentence boundary.
        """
        sentence = sentence.strip()
        if not sentence:
            return

        t0 = time.perf_counter()
        audio = self._synthesize(sentence)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("TTS synth %.0f chars in %.1f ms", len(sentence), elapsed)

        # Yield in ~10ms chunks for smooth playback
        chunk_size = int(self.sample_rate * 0.010)
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]

    def _synthesize(self, text: str) -> np.ndarray:
        """Run the active backend and return float32 audio array."""
        if self._backend_type == "sherpa_onnx":
            result = self._sherpa.generate(text=text, sid=0, speed=1.0)
            return np.array(result.samples, dtype=np.float32)
        elif self._backend_type == "piper":
            chunks = list(self._piper.synthesize(text))
            if chunks:
                arrays = [c.audio_float_array for c in chunks if hasattr(c, "audio_float_array")]
                return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)
        else:
            result = self._stub.synthesize(text)
            return result.samples

    # ------------------------------------------------------------------
    # Token stream → TTS pipeline
    # ------------------------------------------------------------------

    def stream_from_tokens(self, token_iter: Iterator[str]) -> Iterator[np.ndarray]:
        """
        High-level helper:
        Takes raw LLM token stream, does sentence boundary detection,
        and yields audio chunks as each sentence completes.

        Usage in pipeline.py:
            for audio_chunk in tts.stream_from_tokens(llm.stream(text, history)):
                speaker.play(audio_chunk)
        """
        buffer = ""
        for token in token_iter:
            buffer += token
            sentences, buffer = split_on_boundary(buffer)
            for sent in sentences:
                yield from self.synth_stream(sent)

        # Flush any remaining text
        buffer = buffer.strip()
        if buffer and len(buffer) >= 3:
            yield from self.synth_stream(buffer)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tts = SentenceChunkedTTS()
    print(f"Backend: {tts._backend_type}, Sample Rate: {tts.sample_rate} Hz")

    # Test direct synthesis
    test_sentences = [
        "Cool water should be run over a minor burn for ten to twenty minutes.",
        "Do not apply ice, butter, or toothpaste. Consult a doctor for serious burns.",
    ]
    total_chunks = 0
    t0 = time.perf_counter()
    for sent in test_sentences:
        for chunk in tts.synth_stream(sent):
            total_chunks += 1
            if total_chunks == 1:
                ttfa = (time.perf_counter() - t0) * 1000
                print(f"Time-to-first-audio: {ttfa:.1f} ms")
    print(f"Total audio chunks: {total_chunks}")
    print(f"Total time: {(time.perf_counter()-t0)*1000:.0f} ms")
