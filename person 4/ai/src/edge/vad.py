"""
Silero VAD: Voice Activity Detection with pre-roll buffer and barge-in (HNX26EPS08).
Uses Silero VAD v4 ONNX model via onnxruntime (CPU only).
Downloads model automatically on first run, caches locally.
"""

import time
import threading
import collections
import urllib.request
from pathlib import Path
from typing import Iterator, Optional, Callable
import numpy as np
import onnxruntime as ort

from src.edge.contracts import SpeechSegment, SAMPLE_RATE

# ---------------------------------------------------------------------------
# MODEL DOWNLOAD & CACHE
# ---------------------------------------------------------------------------

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
MODEL_CACHE = Path(__file__).resolve().parent.parent.parent / "models" / "vad" / "silero_vad.onnx"

CHUNK_SAMPLES = 512          # 32ms at 16kHz (Silero requirement)
SAMPLE_RATE_VAD = 16000


def _ensure_model() -> Path:
    if MODEL_CACHE.exists():
        return MODEL_CACHE
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[VAD] Downloading Silero VAD ONNX to {MODEL_CACHE} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
    print(f"[VAD] Download complete ({MODEL_CACHE.stat().st_size // 1024} KB)")
    return MODEL_CACHE


# ---------------------------------------------------------------------------
# SILERO VAD ONNX WRAPPER
# ---------------------------------------------------------------------------

class SileroVADONNX:
    """Thin wrapper around the Silero VAD ONNX model."""

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._reset_state()

    def _reset_state(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array([SAMPLE_RATE_VAD], dtype=np.int64)

    def reset(self):
        self._reset_state()

    def __call__(self, chunk: np.ndarray) -> float:
        """
        Run VAD on a 512-sample (32ms) float32 chunk.
        Returns speech probability in [0, 1].
        """
        if len(chunk) != CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, max(0, CHUNK_SAMPLES - len(chunk))))[:CHUNK_SAMPLES]

        x = chunk[np.newaxis, :].astype(np.float32)
        outs = self._session.run(
            None,
            {"input": x, "sr": self._sr, "h": self._h, "c": self._c},
        )
        prob = float(outs[0].squeeze())
        self._h = outs[1]
        self._c = outs[2]
        return prob


# ---------------------------------------------------------------------------
# VAD ENGINE WITH PRE-ROLL AND HANGOVER
# ---------------------------------------------------------------------------

class SileroVAD:
    """
    Full VAD engine implementing the VAD protocol from contracts.py.

    Features:
    - Pre-roll circular buffer (~250ms) so word onsets are never clipped.
    - Hangover duration (~500ms) to handle natural speaking pauses.
    - Light noise gate to filter very low-amplitude noise.
    - Barge-in detection: fires a callback if speech detected during playback.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        hangover_chunks: int = 16,     # ~512ms hangover at 32ms/chunk
        preroll_chunks: int = 8,       # ~256ms pre-roll
        noise_gate: float = 0.005,     # min RMS to consider chunk non-silent
        model_path: Optional[Path] = None,
        on_barge_in: Optional[Callable] = None,
    ):
        self.threshold = threshold
        self.hangover_chunks = hangover_chunks
        self.preroll_chunks = preroll_chunks
        self.noise_gate = noise_gate
        self.on_barge_in = on_barge_in

        model = model_path or _ensure_model()
        self._vad = SileroVADONNX(model)

        # Pre-roll ring buffer
        self._preroll: collections.deque[np.ndarray] = collections.deque(maxlen=preroll_chunks)

        # State
        self._speaking = False
        self._hangover = 0
        self._speech_chunks: list[np.ndarray] = []
        self._t_speech_start: float = 0.0

        # Barge-in
        self._bot_speaking = threading.Event()

    def set_bot_speaking(self, speaking: bool) -> None:
        """Signal whether the bot is currently speaking (for barge-in detection)."""
        if speaking:
            self._bot_speaking.set()
        else:
            self._bot_speaking.clear()

    def segments(self, mic: Iterator[np.ndarray]) -> Iterator[SpeechSegment]:
        """
        VAD protocol: consume mic chunk stream, yield SpeechSegment objects.
        Each SpeechSegment contains the full audio array of one utterance,
        plus timestamps for speech_end and vad_decision.
        """
        self._vad.reset()
        self._preroll.clear()
        self._speaking = False
        self._hangover = 0
        self._speech_chunks = []

        for raw_chunk in mic:
            if raw_chunk is None:
                break

            chunk = raw_chunk.astype(np.float32)

            # Noise gate: skip if chunk is near-silent
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < self.noise_gate and not self._speaking:
                self._preroll.append(chunk)
                continue

            prob = self._vad(chunk)

            # --- Barge-in detection ---
            if self._bot_speaking.is_set() and prob > self.threshold:
                if self.on_barge_in:
                    self.on_barge_in()

            if prob > self.threshold:
                if not self._speaking:
                    # SPEECH START: prepend pre-roll buffer
                    self._speaking = True
                    self._t_speech_start = time.perf_counter()
                    self._speech_chunks = list(self._preroll) + [chunk]
                else:
                    self._speech_chunks.append(chunk)
                self._hangover = 0
            else:
                if self._speaking:
                    self._speech_chunks.append(chunk)
                    self._hangover += 1
                    if self._hangover >= self.hangover_chunks:
                        # SPEECH END
                        t_speech_end = time.perf_counter()
                        audio = np.concatenate(self._speech_chunks)
                        t_vad_decision = time.perf_counter()

                        yield SpeechSegment(
                            audio=audio,
                            t_speech_end=t_speech_end,
                            t_vad_decision=t_vad_decision,
                        )

                        # Reset
                        self._speaking = False
                        self._hangover = 0
                        self._speech_chunks = []
                        self._vad.reset()
                        self._preroll.clear()
                else:
                    self._preroll.append(chunk)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("[VAD] Ensuring Silero VAD ONNX model is present...")
    model_path = _ensure_model()
    vad_model = SileroVADONNX(model_path)
    # Test with a silent chunk
    silent = np.zeros(512, dtype=np.float32)
    prob = vad_model(silent)
    print(f"[VAD] Silent chunk probability: {prob:.4f} (expected ~0.0)")
    # Test with a loud chunk
    noise = np.random.randn(512).astype(np.float32) * 0.5
    prob2 = vad_model(noise)
    print(f"[VAD] Noise chunk probability: {prob2:.4f}")
    print("[VAD] vad.py OK")
