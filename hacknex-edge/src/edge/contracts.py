"""contracts.py — Frozen shared data types & protocols.
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)

This file is the single frozen contract integrating Person 1, Person 2, Person 3, and Person 4.
All modules plug together through these interfaces.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, Callable, Any, runtime_checkable
import numpy as np

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# 1. CORE DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """Continuous audio segment containing validated user speech."""
    audio: np.ndarray
    t_speech_end: float = 0.0
    t_vad_decision: float = 0.0
    t_speech_start: float = 0.0
    sample_rate: int = SAMPLE_RATE
    audio_hangover_ms: float = 768.0
    vad_compute_ms: float = 0.0

    @property
    def duration_sec(self) -> float:
        return len(self.audio) / float(self.sample_rate) if self.sample_rate else 0.0

    @property
    def duration_ms(self) -> float:
        return self.duration_sec * 1000.0

    @property
    def endpointing_delay_ms(self) -> float:
        return float(self.audio_hangover_ms)

    @property
    def start_time(self) -> float:
        return self.t_speech_start

    @property
    def end_time(self) -> float:
        return self.t_speech_end


@dataclass
class Transcript:
    """Speech recognition result produced by Module A (ASR)."""
    text: str = ""
    is_final: bool = True
    lang: str = "en"
    asr_ms: float = 0.0
    confidence: float = 1.0
    words: list[dict[str, Any]] = field(default_factory=list)
    lang_prob: float = 1.0
    lang_candidates: dict[str, float] = field(default_factory=dict)
    translated: bool = False
    translation_text: str = ""
    segment: Optional[SpeechSegment] = None

    @property
    def asr_latency_ms(self) -> float:
        return self.asr_ms

    @asr_latency_ms.setter
    def asr_latency_ms(self, val: float) -> None:
        self.asr_ms = val


@dataclass
class Turn:
    """A single turn in the conversation history."""
    role: str
    content: str = ""
    text: str = ""

    def __post_init__(self):
        if not self.content and self.text:
            self.content = self.text
        elif not self.text and self.content:
            self.text = self.content


@dataclass
class FastRouteResult:
    """Result from Person 1's fast rule router."""
    handled: bool
    response: Optional[str] = None
    command_type: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class LatencyEvent:
    """A named timestamp marker for the waterfall dashboard."""
    name: str
    ts: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self, other: "LatencyEvent") -> float:
        return (self.ts - other.ts) * 1000.0


@dataclass
class TelemetryFrame:
    """Snapshot sent to the dashboard over WebSocket."""
    event: str
    vad_ms: float = 0.0
    asr_ms: float = 0.0
    router_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    tts_first_audio_ms: float = 0.0
    e2e_ms: float = 0.0
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    governor_tier: int = 0
    transcript: str = ""
    response_preview: str = ""


# ---------------------------------------------------------------------------
# 2. INTERFACE PROTOCOLS
# ---------------------------------------------------------------------------

@runtime_checkable
class AudioIOProtocol(Protocol):
    def mic_stream(self, chunk_size: int = 512) -> Iterator[np.ndarray]: ...
    def play(self, audio: np.ndarray, sample_rate: int = 16000) -> None: ...
    def stop_playback(self) -> None: ...


@runtime_checkable
class VADProtocol(Protocol):
    def segments(self, mic_stream: Iterator[np.ndarray]) -> Iterator[SpeechSegment]: ...
    def set_barge_in_callback(self, callback: Callable[[], None]) -> None: ...


@runtime_checkable
class ASRProtocol(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: Optional[str] = None) -> Transcript: ...
    def transcribe_stream(self, chunks: Iterator[np.ndarray]) -> Iterator[Transcript]: ...


@runtime_checkable
class RouterProtocol(Protocol):
    def route_intent(self, user_text: str) -> Optional[str]: ...


@runtime_checkable
class KnowledgeProtocol(Protocol):
    def search(self, query: str, top_k: int = 1) -> str: ...


@runtime_checkable
class LLMProtocol(Protocol):
    def stream(self, user_text: str, history: list[Turn]) -> Iterator[str]: ...


@runtime_checkable
class TTSProtocol(Protocol):
    @property
    def sample_rate(self) -> int: ...
    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]: ...


@runtime_checkable
class MetricsProtocol(Protocol):
    def mark(self, req_id: str, event: str, t: Optional[float] = None) -> None: ...


# Backward-compatibility aliases
VAD = VADProtocol
ASR = ASRProtocol
Router = RouterProtocol
LLM = LLMProtocol
TTS = TTSProtocol
Metrics = MetricsProtocol
