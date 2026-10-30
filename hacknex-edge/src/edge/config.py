"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Module A: Centralized Edge Audio, Multilingual & Language Detection Configuration
Author: Edge Audio, ASR & Multilingual Lead

Central Single Source of Truth for all audio, VAD, ASR, multilingual, language detection, and latency settings.
No magic numbers across any files.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class AudioConfig:
    """Core audio I/O & sampling configuration."""
    sample_rate: int = 16000
    chunk_size: int = 512              # 512 samples @ 16kHz = 32.0 ms
    channels: int = 1                  # Mono
    dtype: str = "float32"
    playback_chunk_size: int = 512     # Small buffer for low-latency output & abort (<5ms)


@dataclass(frozen=True)
class VADConfig:
    """Silero VAD, Pre-roll, Hangover, and Energy filter configuration."""
    model_path: str = "models/vad/silero_vad.onnx"
    start_threshold: float = 0.35       # Relaxed onset threshold for natural human speech
    end_threshold: float = 0.20         # Prevents trailing syllables/vowels from triggering false silence
    preroll_ms: float = 384.0          # 12 chunks * 32ms = 384ms (ensures leading syllables are not clipped)
    hangover_ms: float = 768.0         # 24 chunks * 32ms = 768ms (natural conversational inter-word pause tolerance)
    min_speech_ms: float = 128.0       # 4 chunks * 32ms = 128ms
    energy_gate_multiplier: float = 1.20 # RMS multiplier over noise floor
    noise_floor_alpha: float = 0.98    # Exponential moving average factor


@dataclass(frozen=True)
class BargeInConfig:
    """Self-trigger & Echo protection configuration."""
    speech_threshold_multiplier: float = 1.40  # Raise threshold during bot speech
    barge_in_min_speech_ms: float = 160.0       # Must sustain speech for >= 160ms (5 chunks) to interrupt
    guard_window_ms: float = 100.0             # Mute/ignore window right after bot starts speaking
    recommend_headphones: bool = True          # Documented recommended competition setup


@dataclass(frozen=True)
class ASRConfig:
    """Offline ASR English configuration."""
    # Active model directory: ultra-fast English whisper_tiny_en
    whisper_model_dir: str = (
        "models/asr/whisper_tiny_en"
        if os.path.exists("models/asr/whisper_tiny_en/model.bin")
        else "models/asr/whisper_tiny_multilingual"
    )
    sherpa_model_dir: str = "models/asr/sherpa_streaming"
    cpu_threads: int = 4
    beam_size: int = 1                         # Greedy decoding for minimum latency
    compute_type: str = "int8"
    offline_enforced: bool = True

    # -------------------------------------------------------------
    # ENGLISH LANGUAGE ARCHITECTURE
    # -------------------------------------------------------------
    language_mode: str = "en"
    allowed_languages: tuple[str, ...] = ("en",)
    lang_confidence_margin: float = 0.15
    lang_min_prob: float = 0.35
    lang_min_audio_ms: float = 1000.0
    prior_boost_ta_en: float = 0.10
    sticky_session_lang: bool = False
    min_command_confidence: float = 0.40              # Minimum confidence to execute OS action

    # -------------------------------------------------------------
    # HALLUCINATION & SIGNAL THRESHOLDS
    # -------------------------------------------------------------
    no_speech_threshold: float = 0.75                  # Drop segments where no_speech_prob > 0.75 (was 0.60 — too strict)
    logprob_threshold: float = -1.50                   # Average logprob lower bound (was -1.00 — was dropping real speech)
    compression_ratio_threshold: float = 2.40          # Drop repetitive compression ratio > 2.40

    # -------------------------------------------------------------
    # ENGLISH TTS VOICE MAP
    # -------------------------------------------------------------
    asr_model_english: str = "models/asr/whisper_tiny_en"
    tts_voices: tuple[tuple[str, str], ...] = (
        ("en", "models/tts/en_piper"),
    )



@dataclass(frozen=True)
class EdgeListeningConfig:
    """Master configuration container for Module A."""
    audio: AudioConfig = AudioConfig()
    vad: VADConfig = VADConfig()
    barge_in: BargeInConfig = BargeInConfig()
    asr: ASRConfig = ASRConfig()

    @property
    def preroll_chunks(self) -> int:
        return max(1, int(round(self.vad.preroll_ms / (self.audio.chunk_size / self.audio.sample_rate * 1000.0))))

    @property
    def hangover_chunks(self) -> int:
        return max(1, int(round(self.vad.hangover_ms / (self.audio.chunk_size / self.audio.sample_rate * 1000.0))))

    @property
    def min_speech_chunks(self) -> int:
        return max(1, int(round(self.vad.min_speech_ms / (self.audio.chunk_size / self.audio.sample_rate * 1000.0))))

    @property
    def barge_in_min_chunks(self) -> int:
        return max(1, int(round(self.barge_in.barge_in_min_speech_ms / (self.audio.chunk_size / self.audio.sample_rate * 1000.0))))


# Global singleton configuration
CONFIG = EdgeListeningConfig()
