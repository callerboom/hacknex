"""
contracts.py — Frozen shared data types & protocols.
DO NOT ALTER without cross-module coordination.
HackNEX 2026 — HNX26EPS08
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """Raw audio captured between VAD start and end events."""
    audio: np.ndarray          # float32, mono, 16 kHz
    sample_rate: int = 16000
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0      # filled after VAD end

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class Transcript:
    """ASR result produced by Module A."""
    text: str
    is_final: bool = True
    confidence: float = 1.0
    asr_latency_ms: float = 0.0
    segment: SpeechSegment | None = None


@dataclass
class Turn:
    """One exchange in the conversation history."""
    role: str          # "user" | "assistant"
    content: str


@dataclass
class LatencyEvent:
    """A named timestamp marker for the waterfall dashboard."""
    name: str
    ts: float = field(default_factory=time.perf_counter)  # perf_counter seconds

    def elapsed_ms(self, other: "LatencyEvent") -> float:
        return (self.ts - other.ts) * 1000.0


@dataclass
class TelemetryFrame:
    """Snapshot sent to the dashboard over WebSocket."""
    event: str                          # e.g. "utterance_done"
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
# Module Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class VAD(Protocol):
    """Voice Activity Detector protocol (Module A)."""
    def is_speech(self, frame: np.ndarray) -> bool: ...
    def reset(self) -> None: ...


@runtime_checkable
class ASR(Protocol):
    """Automatic Speech Recognition protocol (Module A)."""
    def transcribe(self, segment: SpeechSegment) -> Transcript: ...


@runtime_checkable
class Router(Protocol):
    """Fast rule-based router protocol (Person 1)."""
    def route(self, user_text: str) -> str | None: ...


@runtime_checkable
class LLM(Protocol):
    """Local GGUF LLM protocol (Person 3 — Module B)."""
    def stream(self, user_text: str, history: list[Turn]) -> Iterator[str]: ...


@runtime_checkable
class TTS(Protocol):
    """Text-to-Speech protocol (Person 3 — Module B)."""
    sample_rate: int

    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]: ...
