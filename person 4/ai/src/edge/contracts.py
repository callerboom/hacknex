from dataclasses import dataclass
from typing import Iterator, Optional, Protocol
import numpy as np

SAMPLE_RATE = 16000  # mono float32 in [-1, 1] inside the pipeline

@dataclass
class Turn:
    role: str
    text: str

@dataclass
class Transcript:
    text: str
    is_final: bool
    lang: str
    asr_ms: float
    confidence: float

@dataclass
class SpeechSegment:
    audio: np.ndarray
    t_speech_end: float
    t_vad_decision: float  # perf_counter times

class VAD(Protocol):
    def segments(self, mic: Iterator[np.ndarray]) -> Iterator['SpeechSegment']: ...

class ASR(Protocol):
    def transcribe(self, audio: np.ndarray) -> Transcript: ...
    def transcribe_stream(self, chunks: Iterator[np.ndarray]) -> Iterator[Transcript]: ...

class LLM(Protocol):
    def stream(self, user_text: str, history: list[Turn]) -> Iterator[str]: ...

class TTS(Protocol):
    sample_rate: int
    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]: ...

class Metrics(Protocol):
    def mark(self, req_id: str, event: str, t: Optional[float] = None) -> None: ...

