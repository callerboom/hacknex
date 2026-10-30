"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Module Contracts & Frozen Shared Data Types

STRICT RULE: This file is the single frozen contract between Person 1, Person 2,
Person 3, and Person 4. All modules plug together ONLY through these contracts.

CONTRACT_CHANGELOG:
- Added VAD and ASR aliases pointing to VADProtocol and ASRProtocol for strict backward compatibility.
- Added t_speech_start to SpeechSegment to accurately track start-of-speech timestamp.
- Updated endpointing_delay_ms to reflect AUDIO-TIME delay (hangover + chunk duration) rather than wall clock.
- Added vad_compute_ms to SpeechSegment to record pure CPU compute time taken by VAD inference.
- Added optional words field to Transcript for word-level timestamps and alignments.
- Added lang_prob (float in [0.0, 1.0]) for language detection confidence.
- Added lang_candidates (dict[str, float]) for top language candidate probabilities.
- Added translated (bool) and translation_text (str) for Tier 3 language translate fallback.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, Callable, Any
import numpy as np


# ---------------------------------------------------------------------------
# 1. AUDIO & SPEECH CONTRACTS (PERSON 2: MODULE A)
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    """
    Continuous audio segment containing validated user speech.
    Produced by Person 2's VAD and consumed by ASR.
    """
    audio: np.ndarray                 # 1D float32 normalized in [-1.0, 1.0] at sample_rate
    t_speech_start: float             # time.perf_counter() when user speech began
    t_speech_end: float               # time.perf_counter() when user speech ended (pre-hangover)
    t_vad_decision: float             # time.perf_counter() when VAD fired endpoint decision
    sample_rate: int = 16000          # Standard 16kHz audio sample rate
    audio_hangover_ms: float = 768.0  # Audio-time hangover duration in milliseconds
    vad_compute_ms: float = 0.0       # Pure CPU execution time spent by VAD inference

    @property
    def duration_sec(self) -> float:
        """Duration of the audio in seconds."""
        return len(self.audio) / float(self.sample_rate)

    @property
    def endpointing_delay_ms(self) -> float:
        """
        AUDIO-TIME delay (hangover duration plus chunk duration) in ms.
        Computed from audio-time samples/duration, invariant to fast-forward vs real-time playback.
        """
        return float(self.audio_hangover_ms)


@dataclass
class Transcript:
    """
    Speech recognition result produced by Person 2's ASR and consumed by Person 1's Router.
    """
    text: str                         # Normalized clean transcribed text
    is_final: bool = True             # True if segment is complete, False for partial streaming
    lang: str = "en"                  # Detected or configured language ISO code ('ta', 'en', etc.)
    asr_ms: float = 0.0               # Inference time taken by ASR engine in milliseconds
    confidence: float = 1.0           # ASR confidence score [0.0, 1.0]
    words: list[dict[str, Any]] = field(default_factory=list)  # Word-level timestamps if available
    lang_prob: float = 1.0            # Language detection probability score [0.0, 1.0]
    lang_candidates: dict[str, float] = field(default_factory=dict) # Top language candidate scores
    translated: bool = False          # True if text was translated from a non-supported language
    translation_text: str = ""        # English translation if translated is True


# ---------------------------------------------------------------------------
# 2. CONVERSATION & RETRIEVAL CONTRACTS (PERSON 1 & PERSON 3)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single turn in the conversation history."""
    role: str                         # "user" or "assistant" or "system"
    content: str                      # Utterance text


@dataclass
class FastRouteResult:
    """Result from Person 1's fast rule router."""
    handled: bool
    response: Optional[str] = None
    command_type: Optional[str] = None # "greeting", "math", "time", "date", "jarvis_os", etc.
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# 3. INTERFACE PROTOCOLS FOR MODULAR DECOUPLING
# ---------------------------------------------------------------------------

class AudioIOProtocol(Protocol):
    """Protocol for real microphone capture and speaker playback."""
    def mic_stream(self, chunk_size: int = 512) -> Iterator[np.ndarray]:
        """Yields continuous float32 16kHz mono audio chunks."""
        ...

    def play(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """Plays audio through speaker. Blocks until done or cancelled."""
        ...

    def stop_playback(self) -> None:
        """Instantly halts any ongoing playback (for barge-in)."""
        ...


class VADProtocol(Protocol):
    """Protocol for Voice Activity Detection."""
    def segments(self, mic_stream: Iterator[np.ndarray]) -> Iterator[SpeechSegment]:
        """Processes mic stream and yields SpeechSegment objects on speech end."""
        ...

    def set_barge_in_callback(self, callback: Callable[[], None]) -> None:
        """Registers callback to trigger when user interrupts bot."""
        ...


class ASRProtocol(Protocol):
    """Protocol for Automatic Speech Recognition."""
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: Optional[str] = None) -> Transcript:
        """Transcribes complete audio array into Transcript."""
        ...

    def transcribe_stream(self, chunks: Iterator[np.ndarray]) -> Iterator[Transcript]:
        """Transcribes streaming audio chunks progressively."""
        ...


class RouterProtocol(Protocol):
    """Protocol for Person 1's Fast-Path Intent Router."""
    def route_intent(self, user_text: str) -> Optional[str]:
        """Returns instant string if handled, or None if forwarded to LLM."""
        ...


class KnowledgeProtocol(Protocol):
    """Protocol for Person 3's in-memory BM25 retrieval."""
    def search(self, query: str, top_k: int = 1) -> str:
        """Retrieves most relevant knowledge markdown snippet."""
        ...


class LLMProtocol(Protocol):
    """Protocol for Person 3's local streaming GGUF LLM."""
    def stream(self, user_text: str, history: list[Turn]) -> Iterator[str]:
        """Yields generated tokens one by one."""
        ...


class TTSProtocol(Protocol):
    """Protocol for Person 3's sentence-chunked TTS synthesis."""
    @property
    def sample_rate(self) -> int:
        ...

    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]:
        """Synthesizes sentence and yields audio chunks for playback."""
        ...


class MetricsProtocol(Protocol):
    """Protocol for Person 4's high-precision latency and resource tracking."""
    def mark(self, req_id: str, event: str, t: Optional[float] = None) -> None:
        """Records timestamp for a latency milestone."""
        ...


# ---------------------------------------------------------------------------
# 4. BACKWARD COMPATIBILITY ALIASES
# ---------------------------------------------------------------------------
VAD = VADProtocol
ASR = ASRProtocol
