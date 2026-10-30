"""
Audio I/O: Real microphone capture and speaker playback (HNX26EPS08).
16 kHz mono float32. Low-latency streaming via sounddevice.
Supports clean shutdown and barge-in interruption.
"""

import threading
import time
import queue
import wave
import struct
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
CHUNK_SAMPLES = 512          # ~32ms per chunk at 16kHz
PLAYBACK_CHUNK = 2048        # playback buffer size


class MicStream:
    """
    Real microphone capture stream.
    Yields np.ndarray chunks of float32 audio at 16kHz.
    Thread-safe; supports clean stop.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        chunk_samples: int = CHUNK_SAMPLES,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.device = device
        self.chunk_samples = chunk_samples
        self.sample_rate = sample_rate
        self._q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata: np.ndarray, frames: int, t, status) -> None:
        if status:
            pass  # Could log status here
        if not self._stop_event.is_set():
            chunk = indata[:, 0].copy()  # mono
            try:
                self._q.put_nowait(chunk)
            except queue.Full:
                pass

    def __enter__(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._callback,
            latency="low",
        )
        self._stream.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        self._stop_event.set()
        self._q.put(None)  # sentinel
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def chunks(self) -> Iterator[np.ndarray]:
        """Yields audio chunks indefinitely until stop() is called."""
        while not self._stop_event.is_set():
            chunk = self._q.get()
            if chunk is None:
                break
            yield chunk


class AudioPlayer:
    """
    Speaker playback with barge-in interrupt support.
    Accepts np.ndarray float32 chunks at any sample rate.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._stop_event = threading.Event()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def interrupt(self):
        """Signal barge-in: stop playback immediately."""
        self._stop_event.set()

    def play_chunks(self, chunk_iter: Iterator[np.ndarray]) -> None:
        """
        Play audio chunks from an iterator.
        Stops immediately if interrupt() is called.
        """
        self._stop_event.clear()
        self._playing = True
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                device=self.device,
                blocksize=PLAYBACK_CHUNK,
                latency="low",
            ) as stream:
                for chunk in chunk_iter:
                    if self._stop_event.is_set():
                        break
                    if chunk is None or len(chunk) == 0:
                        continue
                    audio = np.asarray(chunk, dtype=np.float32)
                    if audio.ndim == 2:
                        audio = audio[:, 0]
                    stream.write(audio)
        except Exception as e:
            print(f"[AudioPlayer] Playback error: {e}")
        finally:
            self._playing = False

    def play_array(self, audio: np.ndarray) -> None:
        """Play a complete np.ndarray. Blocks until done or interrupted."""
        chunk_size = PLAYBACK_CHUNK
        def gen():
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]
        self.play_chunks(gen())


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as float32 mono numpy array."""
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_ch > 1:
        data = data[::n_ch]  # take first channel

    # Resample to 16kHz if needed (simple linear interpolation)
    if sr != SAMPLE_RATE:
        factor = SAMPLE_RATE / sr
        n_new = int(len(data) * factor)
        data = np.interp(
            np.linspace(0, len(data) - 1, n_new),
            np.arange(len(data)),
            data
        ).astype(np.float32)

    return data, SAMPLE_RATE


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save float32 numpy array as 16-bit WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def list_devices() -> list[dict]:
    """Return available audio input/output devices."""
    devices = []
    for i, d in enumerate(sd.query_devices()):
        devices.append({
            "index": i,
            "name": d["name"],
            "max_input_channels": d["max_input_channels"],
            "max_output_channels": d["max_output_channels"],
            "default_samplerate": d["default_samplerate"],
        })
    return devices


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("[AudioIO] Available audio devices:")
    for d in list_devices():
        tag = ""
        if d["max_input_channels"] > 0:
            tag += "[IN] "
        if d["max_output_channels"] > 0:
            tag += "[OUT]"
        print(f"  {d['index']:3d}: {tag} {d['name']}")
    print(f"\nDefault input device: {sd.query_devices(kind='input')['name']}")
    print(f"Default output device: {sd.query_devices(kind='output')['name']}")
    print(f"\nSample rate: {SAMPLE_RATE} Hz | Chunk: {CHUNK_SAMPLES} samples ({CHUNK_SAMPLES/SAMPLE_RATE*1000:.0f}ms)")
    print("[AudioIO] audio_io.py OK")
