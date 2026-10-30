"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Module A: Audio Input / Output & Sound Device Streamer
Author: Person 2 (Edge Audio Lead)

Features:
- Native sample rate detection & polyphase resampler (44.1k/48k -> 16k mono float32)
- Fixed 512-sample chunk output stream at exactly 16000 Hz
- Low-latency AudioPlayer with stream.abort() for instant hardware buffer drop (<5ms cutoff)
- Echo guard timing hooks for self-trigger protection
- Headless and synthetic audio fallback mode for CI & offline testing
"""

from __future__ import annotations
import math
import queue
import threading
import time
import wave
from typing import Iterator, Optional, Tuple
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    from scipy.signal import resample_poly
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from src.edge.config import CONFIG


# ---------------------------------------------------------------------------
# RESAMPLER: STANDALONE & HIGH-FIDELITY
# ---------------------------------------------------------------------------

def resample(audio: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    """
    Resamples a 1D float32 audio array from sr_in to sr_out.
    Uses polyphase filtering via scipy.signal.resample_poly if available,
    with an accurate Fourier/linear interpolation fallback.
    Preserves audio length and frequency characteristics.
    """
    if audio is None or len(audio) == 0:
        return np.array([], dtype=np.float32)

    audio_f32 = np.asarray(audio, dtype=np.float32)
    if sr_in == sr_out:
        return audio_f32

    if SCIPY_AVAILABLE:
        common_gcd = math.gcd(int(sr_in), int(sr_out))
        up = int(sr_out) // common_gcd
        down = int(sr_in) // common_gcd
        resampled = resample_poly(audio_f32, up, down).astype(np.float32)
        return resampled

    # Pure NumPy fallback via linear interpolation
    target_length = int(round(len(audio_f32) * float(sr_out) / float(sr_in)))
    orig_indices = np.linspace(0.0, len(audio_f32) - 1, num=len(audio_f32), dtype=np.float32)
    new_indices = np.linspace(0.0, len(audio_f32) - 1, num=target_length, dtype=np.float32)
    return np.interp(new_indices, orig_indices, audio_f32).astype(np.float32)


# ---------------------------------------------------------------------------
# LOW-LATENCY AUDIO PLAYER WITH STREAM.ABORT()
# ---------------------------------------------------------------------------

class AudioPlayer:
    """
    Low-latency audio playback controller with stream.abort() for instant cutoff.
    Includes guard window timing for echo / self-trigger protection.
    """
    def __init__(self, default_sr: int = 16000):
        self.default_sr = default_sr
        self.stop_event = threading.Event()
        self._is_playing_event = threading.Event()
        self._lock = threading.Lock()
        self._active_stream = None
        self.last_playback_start_time = 0.0
        self.last_abort_latency_ms = 0.0
        self.device_output_latency_ms = 0.0

    @property
    def is_playing(self) -> bool:
        """Returns True if the speaker is currently outputting sound."""
        return self._is_playing_event.is_set()

    def stop(self) -> float:
        """
        Instantly halts active playback using stream.abort() to immediately
        discard hardware buffers.
        Returns the detect-to-abort call latency in milliseconds.
        """
        t0 = time.perf_counter()
        self.stop_event.set()
        with self._lock:
            if self._active_stream is not None:
                try:
                    # abort() immediately discards pending driver buffers
                    self._active_stream.abort()
                except Exception:
                    pass
        self._is_playing_event.clear()
        abort_duration_ms = (time.perf_counter() - t0) * 1000.0
        self.last_abort_latency_ms = abort_duration_ms
        return abort_duration_ms

    def play_array(self, audio: np.ndarray, sample_rate: Optional[int] = None, blocking: bool = True) -> bool:
        """
        Plays a float32 numpy array.
        Returns True if played to completion, False if aborted by barge-in.
        """
        if audio is None or len(audio) == 0:
            return True

        sr = sample_rate or self.default_sr
        audio_f32 = np.asarray(audio, dtype=np.float32)

        max_amp = np.max(np.abs(audio_f32))
        if max_amp > 1.0:
            audio_f32 = audio_f32 / max_amp

        self.stop_event.clear()
        self._is_playing_event.set()
        self.last_playback_start_time = time.perf_counter()

        if not SOUNDDEVICE_AVAILABLE:
            # Emulated playback in headless environment
            duration = len(audio_f32) / float(sr)
            t_end = time.perf_counter() + duration
            while time.perf_counter() < t_end:
                if self.stop_event.is_set():
                    self._is_playing_event.clear()
                    return False
                time.sleep(0.01)
            self._is_playing_event.clear()
            return True

        interrupted = False
        try:
            chunk_size = CONFIG.audio.playback_chunk_size
            idx = 0
            total_samples = len(audio_f32)

            with sd.OutputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
                latency="low"
            ) as stream:
                with self._lock:
                    self._active_stream = stream
                    if hasattr(stream, "latency") and stream.latency:
                        self.device_output_latency_ms = stream.latency * 1000.0

                while idx < total_samples:
                    if self.stop_event.is_set():
                        interrupted = True
                        break

                    chunk = audio_f32[idx:min(idx + chunk_size, total_samples)]
                    stream.write(chunk)
                    idx += len(chunk)

        except Exception:
            # Device open failure fallback
            duration = len(audio_f32) / float(sr)
            t_end = time.perf_counter() + duration
            while time.perf_counter() < t_end:
                if self.stop_event.is_set():
                    interrupted = True
                    break
                time.sleep(0.01)
        finally:
            with self._lock:
                self._active_stream = None
            self._is_playing_event.clear()

        return not interrupted

    def play_chunks(self, chunks_iter: Iterator[np.ndarray], sample_rate: Optional[int] = None) -> bool:
        """Plays a continuous stream of chunks with barge-in support."""
        sr = sample_rate or self.default_sr
        self.stop_event.clear()
        self._is_playing_event.set()
        self.last_playback_start_time = time.perf_counter()

        interrupted = False
        for chunk in chunks_iter:
            if self.stop_event.is_set():
                interrupted = True
                break
            if not self.play_array(chunk, sample_rate=sr, blocking=True):
                interrupted = True
                break

        self._is_playing_event.clear()
        return not interrupted


# ---------------------------------------------------------------------------
# RESAMPLING MICROPHONE STREAM WITH NATIVE SAMPLE RATE AUTO-DETECTION
# ---------------------------------------------------------------------------

class MicrophoneStream:
    """
    Continuous real-time microphone capture stream.
    Detects hardware native sample rate (44.1k/48k), resamples to 16kHz mono float32,
    and yields exactly 512-sample chunks.
    """
    def __init__(self, target_sr: int = 16000, chunk_size: int = 512, queue_maxsize: int = 100):
        self.target_sr = target_sr
        self.chunk_size = chunk_size
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_maxsize)
        self.is_running = False
        self._stream = None
        self._thread = None
        self.native_sr = target_sr
        self._resample_buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()

    def _detect_native_sr(self) -> int:
        if not SOUNDDEVICE_AVAILABLE:
            return self.target_sr
        try:
            dev_info = sd.query_devices(kind="input")
            sr = int(dev_info.get("default_samplerate", self.target_sr))
            return sr if sr > 0 else self.target_sr
        except Exception:
            return self.target_sr

    def _audio_callback(self, indata, frames, time_info, status):
        """Native sounddevice callback."""
        raw_audio = indata[:, 0].copy().astype(np.float32)

        # Fast path if native rate is already target rate (16kHz)
        if self.native_sr == self.target_sr and len(raw_audio) == self.chunk_size:
            try:
                self.audio_queue.put_nowait(raw_audio)
                return
            except queue.Full:
                try:
                    _ = self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(raw_audio)
                    return
                except (queue.Empty, queue.Full):
                    pass

        # Resample from native rate to 16000 if needed
        if self.native_sr != self.target_sr:
            audio_16k = resample(raw_audio, self.native_sr, self.target_sr)
        else:
            audio_16k = raw_audio

        with self._lock:
            self._resample_buffer = np.concatenate([self._resample_buffer, audio_16k])

            # Slice into exact 512-sample chunks
            while len(self._resample_buffer) >= self.chunk_size:
                chunk = self._resample_buffer[:self.chunk_size]
                self._resample_buffer = self._resample_buffer[self.chunk_size:]
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    try:
                        _ = self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(chunk)
                    except (queue.Empty, queue.Full):
                        pass

    def start(self) -> None:
        """Starts real microphone audio capture."""
        if self.is_running:
            return

        self.is_running = True
        self.native_sr = self._detect_native_sr()

        if SOUNDDEVICE_AVAILABLE:
            try:
                # Try opening directly at target sample rate (16000Hz mono, 512 block)
                self._stream = sd.InputStream(
                    samplerate=self.target_sr,
                    blocksize=self.chunk_size,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback
                )
                self._stream.start()
                self.native_sr = self.target_sr
                return
            except Exception as e_16k:
                try:
                    # Fallback to hardware native rate + polyphase resampler
                    native_block = int(round(self.chunk_size * float(self.native_sr) / float(self.target_sr)))
                    self._stream = sd.InputStream(
                        samplerate=self.native_sr,
                        blocksize=native_block,
                        channels=1,
                        dtype="float32",
                        callback=self._audio_callback
                    )
                    self._stream.start()
                    return
                except Exception as e_native:
                    print(f"[WARNING] Microphone initialization failed: {e_16k} / {e_native}", file=sys.stderr)

        # Headless fallback generator thread
        self._thread = threading.Thread(target=self._headless_generator, daemon=True)
        self._thread.start()

    def _headless_generator(self):
        """Generates ambient silence chunks if physical mic is unavailable."""
        frame_sec = self.chunk_size / float(self.target_sr)
        while self.is_running:
            chunk = np.random.normal(0, 0.0005, self.chunk_size).astype(np.float32)
            try:
                self.audio_queue.put(chunk, timeout=0.1)
            except queue.Full:
                pass
            time.sleep(frame_sec)

    def stop(self) -> None:
        """Stops microphone capture."""
        self.is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def stream(self) -> Iterator[np.ndarray]:
        """Yields audio chunks continuously."""
        if not self.is_running:
            self.start()

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.2)
                yield chunk
            except queue.Empty:
                continue


# ---------------------------------------------------------------------------
# AUDIO UTILITIES: WAV I/O & SYNTHESIS
# ---------------------------------------------------------------------------

def load_wav(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Loads any WAV file and returns 1D float32 array at target_sr."""
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio = (audio_int16 / 32768.0).astype(np.float32)
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    else:
        audio = (np.frombuffer(raw_bytes, dtype=np.int8) / 128.0).astype(np.float32)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if src_sr != target_sr:
        audio = resample(audio, src_sr, target_sr)

    return audio


def save_wav(file_path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Saves a 1D float32 numpy array into a 16-bit PCM WAV file."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def generate_speech_like_audio(duration_sec: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Synthesizes harmonic tones for pipeline audio testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    carrier = 0.4 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 800 * t) + 0.2 * np.sin(2 * np.pi * 2200 * t)
    envelope = (np.sin(2 * np.pi * 3.5 * t) ** 2) * np.exp(-t * 0.2)
    return (carrier * envelope * 0.8).astype(np.float32)
