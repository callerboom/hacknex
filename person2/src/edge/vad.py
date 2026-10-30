"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Module A: High-Performance Voice Activity Detection (VAD) & Barge-In Guard
Author: Person 2 (Edge Audio Lead)

Features:
- Silero VAD v5 ONNX with 64-sample acoustic context (100% offline)
- AUDIO-TIME Endpoint Delay calculation (hangover + chunk duration in sample time)
- Pure VAD compute time metric (vad_compute_ms)
- Multi-layer Echo & Self-Trigger Protection:
  * Raised threshold multiplier during bot speaker output
  * 160ms+ continuous speech requirement before firing barge-in
  * 100ms mute/guard window after bot starts speaking
- Circular Pre-roll buffer (256ms) & transient click rejection
"""

from __future__ import annotations
import collections
import os
import threading
import time
from typing import Callable, Iterator, Optional
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ort = None
    ONNX_AVAILABLE = False

from src.edge.contracts import SpeechSegment
from src.edge.config import CONFIG


class SileroVADEngine:
    """
    Offline Silero VAD v5 ONNX model wrapper.
    Accepts 512-sample (32ms) audio frames at 16000 Hz with 64-sample acoustic context.
    Caches the ONNX InferenceSession instance to avoid redundant reload and memory thrashing.
    """
    _session_cache: dict[str, ort.InferenceSession] = {}
    _cache_lock = threading.Lock()

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or CONFIG.vad.model_path
        self.session: Optional[ort.InferenceSession] = None
        self.sample_rate = CONFIG.audio.sample_rate
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._sr_tensor = np.array(self.sample_rate, dtype=np.int64)
        self._init_session()

    def _init_session(self):
        if not ONNX_AVAILABLE:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"\n[CRITICAL ERROR] Silero VAD model missing at '{self.model_path}'.\n"
                "Please run 'python scripts/download_models.py' to download model weights."
            )

        with self._cache_lock:
            if self.model_path not in self._session_cache:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self._session_cache[self.model_path] = ort.InferenceSession(
                    self.model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
            self.session = self._session_cache[self.model_path]

    def reset_state(self):
        """Resets recurrent state and context for fresh utterance."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def predict(self, audio_chunk: np.ndarray) -> float:
        """Calculates speech probability for a 512-sample float32 chunk."""
        if self.session is None:
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            return 1.0 if rms > 0.02 else 0.0

        if len(audio_chunk) != 512:
            if len(audio_chunk) < 512:
                audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)), mode="constant")
            else:
                audio_chunk = audio_chunk[:512]

        chunk_2d = audio_chunk.reshape(1, 512).astype(np.float32)
        x = np.concatenate([self._context, chunk_2d], axis=1)
        self._context = x[:, -64:]

        inputs = {
            "input": x,
            "state": self._state,
            "sr": self._sr_tensor
        }

        outputs = self.session.run(None, inputs)
        prob = float(outputs[0].flat[0])
        self._state = outputs[1]
        return prob


class EdgeVAD:
    """
    Production Edge Voice Activity Detector with:
    - Audio-time endpoint delay calculation (invariant to playback speed)
    - Multi-stage echo/self-trigger guard
    - Circular pre-roll ring buffer
    - Real-time barge-in detector
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = CONFIG.audio.sample_rate,
        chunk_size: int = CONFIG.audio.chunk_size,
        start_threshold: float = CONFIG.vad.start_threshold,
        end_threshold: float = CONFIG.vad.end_threshold,
        preroll_chunks: Optional[int] = None,
        hangover_chunks: Optional[int] = None,
        min_speech_chunks: Optional[int] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
        is_bot_speaking_fn: Optional[Callable[[], bool]] = None,
        get_playback_start_fn: Optional[Callable[[], float]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_speech_ended: Optional[Callable[[], None]] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.preroll_chunks = preroll_chunks or CONFIG.preroll_chunks
        self.hangover_chunks = hangover_chunks or CONFIG.hangover_chunks
        self.min_speech_chunks = min_speech_chunks or CONFIG.min_speech_chunks
        self.on_barge_in = on_barge_in
        self.is_bot_speaking_fn = is_bot_speaking_fn
        self.get_playback_start_fn = get_playback_start_fn
        self.on_speech_started = on_speech_started
        self.on_speech_ended = on_speech_ended

        # Pre-roll ring buffer
        self.pre_roll_ring: collections.deque[np.ndarray] = collections.deque(maxlen=self.preroll_chunks)

        # Dynamic noise floor tracker
        self.noise_floor_rms = 0.005
        self.noise_alpha = CONFIG.vad.noise_floor_alpha

        # Consecutive speech frames counter for barge-in stability
        self._barge_in_consecutive_frames = 0
        self._barge_in_fired_for_turn = False

        # VAD Engine
        self.engine = SileroVADEngine(model_path)

    def set_barge_in_callback(self, callback: Callable[[], None]) -> None:
        self.on_barge_in = callback

    def set_is_bot_speaking_fn(self, fn: Callable[[], bool]) -> None:
        self.is_bot_speaking_fn = fn

    def set_playback_start_fn(self, fn: Callable[[], float]) -> None:
        self.get_playback_start_fn = fn

    def set_speech_started_callback(self, callback: Callable[[], None]) -> None:
        self.on_speech_started = callback

    def set_speech_ended_callback(self, callback: Callable[[], None]) -> None:
        self.on_speech_ended = callback

    def segments(self, mic_stream: Iterator[np.ndarray]) -> Iterator[SpeechSegment]:
        """
        Consumes streaming 512-sample audio chunks and yields SpeechSegment objects.
        Accurately reports audio-time endpoint delay and pure VAD compute time.
        """
        in_speech = False
        accumulated_chunks: list[np.ndarray] = []
        silence_counter = 0
        speech_chunks_count = 0

        t_speech_start = 0.0
        t_speech_end = 0.0
        total_vad_compute_ms = 0.0

        for chunk in mic_stream:
            t_now = time.perf_counter()

            # -------------------------------------------------------------
            # 1. OPTIMIZED VAD PREDICTION
            # -------------------------------------------------------------
            t_vad_start = time.perf_counter()
            chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))

            if not in_speech:
                self.noise_floor_rms = (self.noise_alpha * self.noise_floor_rms) + ((1.0 - self.noise_alpha) * chunk_rms)
                self.noise_floor_rms = max(1e-6, min(self.noise_floor_rms, 0.05))

            # Only skip neural inference on pure digital silence (< 1e-6)
            if chunk_rms < 1e-6 and not in_speech:
                prob = 0.0
            else:
                prob = self.engine.predict(chunk)

            total_vad_compute_ms += (time.perf_counter() - t_vad_start) * 1000.0

            # -------------------------------------------------------------
            # 2. ECHO / SELF-TRIGGER PROTECTION & BARGE-IN
            # -------------------------------------------------------------
            is_speaking = bool(self.is_bot_speaking_fn and self.is_bot_speaking_fn())

            # Adjust threshold if bot is outputting audio
            active_start_thresh = self.start_threshold
            if is_speaking:
                active_start_thresh = min(0.95, self.start_threshold * CONFIG.barge_in.speech_threshold_multiplier)

            if is_speaking:
                # Check guard window right after bot started speaking
                in_guard_window = False
                if self.get_playback_start_fn:
                    start_t = self.get_playback_start_fn()
                    if (t_now - start_t) * 1000.0 < CONFIG.barge_in.guard_window_ms:
                        in_guard_window = True

                if prob >= active_start_thresh and not in_guard_window:
                    self._barge_in_consecutive_frames += 1
                    # Require >= 160ms continuous speech to avoid acoustic leak/mic pop false triggers
                    if self._barge_in_consecutive_frames >= CONFIG.barge_in_min_chunks:
                        if not self._barge_in_fired_for_turn and self.on_barge_in:
                            self.on_barge_in()
                            self._barge_in_fired_for_turn = True
                else:
                    self._barge_in_consecutive_frames = 0
            else:
                self._barge_in_consecutive_frames = 0
                self._barge_in_fired_for_turn = False

            # -------------------------------------------------------------
            # 3. HYSTERESIS ENDPOINTING STATE MACHINE
            # -------------------------------------------------------------
            if not in_speech:
                self.pre_roll_ring.append(chunk)

                if prob >= active_start_thresh:
                    in_speech = True
                    silence_counter = 0
                    speech_chunks_count = 1
                    t_speech_start = t_now

                    accumulated_chunks = list(self.pre_roll_ring)
                    self.pre_roll_ring.clear()
                    if self.on_speech_started:
                        try:
                            self.on_speech_started()
                        except Exception:
                            pass
            else:
                accumulated_chunks.append(chunk)

                if prob >= self.end_threshold:
                    silence_counter = 0
                    speech_chunks_count += 1
                    t_speech_end = t_now
                else:
                    silence_counter += 1

                # Check if hangover duration expired
                if silence_counter >= self.hangover_chunks:
                    t_vad_decision = t_now

                    if speech_chunks_count >= self.min_speech_chunks:
                        if self.on_speech_ended:
                            try:
                                self.on_speech_ended()
                            except Exception:
                                pass
                        audio_full = np.concatenate(accumulated_chunks)
                        # Audio-time hangover calculation: hangover_chunks * chunk_duration_ms
                        chunk_ms = (self.chunk_size / float(self.sample_rate)) * 1000.0
                        audio_hangover_ms = float(self.hangover_chunks * chunk_ms)

                        segment = SpeechSegment(
                            audio=audio_full,
                            t_speech_start=t_speech_start,
                            t_speech_end=t_speech_end if t_speech_end > 0.0 else t_now,
                            t_vad_decision=t_vad_decision,
                            sample_rate=self.sample_rate,
                            audio_hangover_ms=audio_hangover_ms,
                            vad_compute_ms=total_vad_compute_ms
                        )
                        yield segment

                    # Reset state machine
                    in_speech = False
                    accumulated_chunks = []
                    silence_counter = 0
                    speech_chunks_count = 0
                    t_speech_start = 0.0
                    t_speech_end = 0.0
                    total_vad_compute_ms = 0.0
                    self.engine.reset_state()
                    self.pre_roll_ring.clear()

        # Handle stream exhaustion while in active speech
        if in_speech and speech_chunks_count >= self.min_speech_chunks and accumulated_chunks:
            t_now = time.perf_counter()
            audio_full = np.concatenate(accumulated_chunks)
            chunk_ms = (self.chunk_size / float(self.sample_rate)) * 1000.0
            audio_hangover_ms = float(silence_counter * chunk_ms)
            segment = SpeechSegment(
                audio=audio_full,
                t_speech_start=t_speech_start,
                t_speech_end=t_speech_end if t_speech_end > 0.0 else t_now,
                t_vad_decision=t_now,
                sample_rate=self.sample_rate,
                audio_hangover_ms=audio_hangover_ms,
                vad_compute_ms=total_vad_compute_ms
            )
            yield segment
