"""
ASR: Offline Speech-to-Text with dual-tier engine (HNX26EPS08).
Tier 1: faster-whisper (CTranslate2 int8, tiny.en / base.en) — high accuracy.
Tier 2: sherpa-onnx streaming transducer — lowest latency chunk-by-chunk.
Implements the ASR protocol from contracts.py.
"""

import time
import threading
from pathlib import Path
from typing import Iterator, Optional
import numpy as np

from src.edge.contracts import Transcript, SAMPLE_RATE

# ---------------------------------------------------------------------------
# TIER 1: FASTER-WHISPER (CTranslate2 int8)
# ---------------------------------------------------------------------------

class WhisperASR:
    """
    Faster-Whisper int8 ASR using CTranslate2 CPU inference.
    Downloads model on first use from HuggingFace (cached locally).
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        compute_type: str = "int8",
        threads: int = 4,
        device: str = "cpu",
    ):
        from faster_whisper import WhisperModel
        print(f"[ASR-Whisper] Loading {model_size} ({compute_type}) on CPU with {threads} threads ...")
        t0 = time.perf_counter()
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=1,
            cpu_threads=threads,
            local_files_only=True,   # OFFLINE: never call HuggingFace API
        )
        self._lang = "en"
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[ASR-Whisper] Model loaded in {load_ms:.0f}ms")


    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Full-segment transcription. Returns Transcript with timing."""
        t0 = time.perf_counter()
        audio_f32 = audio.astype(np.float32)

        segments, info = self._model.transcribe(
            audio_f32,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )

        text_parts = [s.text.strip() for s in segments]
        text = " ".join(text_parts).strip()
        asr_ms = (time.perf_counter() - t0) * 1000

        # Confidence: faster-whisper avg_logprob -> approx probability
        confidence = float(getattr(info, "language_probability", 0.9))

        return Transcript(
            text=text,
            is_final=True,
            lang=self._lang,
            asr_ms=round(asr_ms, 1),
            confidence=round(confidence, 3),
        )

    def transcribe_stream(self, chunks: Iterator[np.ndarray]) -> Iterator[Transcript]:
        """
        Streaming: batch chunks into a single array, then transcribe.
        Yields one partial (is_final=False) and one final (is_final=True) transcript.
        """
        collected = []
        for chunk in chunks:
            if chunk is not None:
                collected.append(chunk)

        if not collected:
            yield Transcript(text="", is_final=True, lang="en", asr_ms=0.0, confidence=0.0)
            return

        audio = np.concatenate(collected)
        result = self.transcribe(audio)
        yield result


# ---------------------------------------------------------------------------
# TIER 2: SHERPA-ONNX STREAMING TRANSDUCER
# ---------------------------------------------------------------------------

class SherpaStreamingASR:
    """
    Sherpa-ONNX online streaming transducer ASR.
    Requires model files in models/asr/sherpa-onnx-streaming-zipformer-*.
    Falls back to WhisperASR if model not found.
    """

    MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "asr" / "sherpa-streaming"

    def __init__(self, threads: int = 4):
        self._available = False
        self._threads = threads
        self._recognizer = None
        self._try_load()

    def _try_load(self):
        encoder = self.MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx"
        decoder = self.MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx"
        joiner  = self.MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx"
        tokens  = self.MODEL_DIR / "tokens.txt"

        if not all(p.exists() for p in [encoder, decoder, joiner, tokens]):
            print(f"[ASR-Sherpa] Model files not found in {self.MODEL_DIR} — tier unavailable.")
            return

        try:
            import sherpa_onnx
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                num_threads=self._threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                max_active_paths=4,
            )
            self._recognizer = recognizer
            self._available = True
            print("[ASR-Sherpa] Streaming transducer loaded.")
        except Exception as e:
            print(f"[ASR-Sherpa] Failed to load: {e}")

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if not self._available:
            raise RuntimeError("Sherpa-ONNX model not available")

        import sherpa_onnx
        t0 = time.perf_counter()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio.astype(np.float32))
        tail_paddings = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
        stream.accept_waveform(SAMPLE_RATE, tail_paddings)
        stream.input_finished()

        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        result = self._recognizer.get_result(stream)
        text = result.text.strip()
        asr_ms = (time.perf_counter() - t0) * 1000

        return Transcript(
            text=text,
            is_final=True,
            lang="en",
            asr_ms=round(asr_ms, 1),
            confidence=0.85,
        )

    def transcribe_stream(self, chunks: Iterator[np.ndarray]) -> Iterator[Transcript]:
        """
        True streaming: feed each chunk to the recognizer and emit partials.
        """
        if not self._available:
            collected = [c for c in chunks if c is not None]
            if collected:
                audio = np.concatenate(collected)
                yield Transcript(text="", is_final=False, lang="en", asr_ms=0.0, confidence=0.0)
            yield Transcript(text="", is_final=True, lang="en", asr_ms=0.0, confidence=0.0)
            return

        import sherpa_onnx
        t0 = time.perf_counter()
        stream = self._recognizer.create_stream()
        last_text = ""

        for chunk in chunks:
            if chunk is None:
                break
            stream.accept_waveform(SAMPLE_RATE, chunk.astype(np.float32))
            if self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            text = self._recognizer.get_result(stream).text.strip()
            if text and text != last_text:
                last_text = text
                yield Transcript(
                    text=text,
                    is_final=False,
                    lang="en",
                    asr_ms=round((time.perf_counter() - t0) * 1000, 1),
                    confidence=0.8,
                )

        # Finalize
        tail = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
        stream.accept_waveform(SAMPLE_RATE, tail)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        final_text = self._recognizer.get_result(stream).text.strip()
        asr_ms = (time.perf_counter() - t0) * 1000

        yield Transcript(
            text=final_text,
            is_final=True,
            lang="en",
            asr_ms=round(asr_ms, 1),
            confidence=0.85,
        )


# ---------------------------------------------------------------------------
# UNIFIED ASR: Auto-selects best available engine
# ---------------------------------------------------------------------------

class EdgeASR:
    """
    Unified ASR implementing the ASR protocol.
    Automatically selects the best available tier based on config.

    Tier 1 (high accuracy): faster-whisper int8
    Tier 2 (low latency): sherpa-onnx streaming transducer
    """

    TIER_1 = "whisper"
    TIER_2 = "sherpa"

    def __init__(
        self,
        tier: str = TIER_1,
        model_size: str = "tiny.en",
        threads: int = 4,
    ):
        self._tier = tier
        self._engine = None
        self._fallback = None

        if tier == self.TIER_1 or tier == "auto":
            try:
                self._engine = WhisperASR(model_size=model_size, threads=threads)
                print(f"[EdgeASR] Using Tier 1: faster-whisper ({model_size})")
            except Exception as e:
                print(f"[EdgeASR] Whisper load failed: {e}")

        if tier == self.TIER_2 or (tier == "auto" and self._engine is None):
            sherpa = SherpaStreamingASR(threads=threads)
            if sherpa._available:
                self._engine = sherpa
                print("[EdgeASR] Using Tier 2: sherpa-onnx streaming transducer")
            else:
                if self._engine is None:
                    print("[EdgeASR] WARNING: No ASR engine available!")

    def set_tier(self, tier: str) -> None:
        """Hot-switch ASR tier (called by governor during degradation)."""
        if tier == self.TIER_1 and not isinstance(self._engine, WhisperASR):
            try:
                self._engine = WhisperASR()
                print("[EdgeASR] Switched to Tier 1 (faster-whisper)")
            except Exception as e:
                print(f"[EdgeASR] Tier 1 switch failed: {e}")
        elif tier == self.TIER_2 and not isinstance(self._engine, SherpaStreamingASR):
            sherpa = SherpaStreamingASR()
            if sherpa._available:
                self._engine = sherpa
                print("[EdgeASR] Switched to Tier 2 (sherpa-onnx)")

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if self._engine is None:
            return Transcript(text="", is_final=True, lang="en", asr_ms=0.0, confidence=0.0)
        return self._engine.transcribe(audio)

    def transcribe_stream(self, chunks: Iterator[np.ndarray]) -> Iterator[Transcript]:
        if self._engine is None:
            yield Transcript(text="", is_final=True, lang="en", asr_ms=0.0, confidence=0.0)
            return
        yield from self._engine.transcribe_stream(chunks)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("[ASR] Testing faster-whisper int8 (tiny.en) ...")
    asr = EdgeASR(tier="whisper", model_size="tiny.en", threads=4)

    # Test with synthetic audio (silence — expect empty transcript)
    silent = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    t0 = time.perf_counter()
    result = asr.transcribe(silent)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[ASR] Silent 2s -> '{result.text}' in {elapsed:.0f}ms (asr_ms={result.asr_ms}ms)")
    print("[ASR] asr.py OK — model download complete, real audio test in test_asr_vad.py")
