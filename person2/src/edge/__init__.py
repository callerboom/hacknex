"""
HackNEX 2026 Edge Conversational Engine.
Module A: Audio, VAD & Multilingual ASR (Person 2)
Router & Actions are decoupled in src/edge/router/ (Person 3)
"""

from src.edge.contracts import SpeechSegment, Transcript, VAD, ASR, VADProtocol, ASRProtocol, AudioIOProtocol
from src.edge.audio_io import AudioPlayer, MicrophoneStream, load_wav, save_wav, resample
from src.edge.vad import EdgeVAD, SileroVADEngine
from src.edge.asr import OfflineASR, FasterWhisperEngine
from src.edge.config import CONFIG
from src.edge.text_utils import normalize_text, score_text

# Re-export router symbols for backward compatibility
from src.edge.router import COMMAND_ANALYZER, MultilingualCommandAnalyzer, AnalyzedCommand

__all__ = [
    "SpeechSegment",
    "Transcript",
    "VAD",
    "ASR",
    "VADProtocol",
    "ASRProtocol",
    "AudioIOProtocol",
    "AudioPlayer",
    "MicrophoneStream",
    "load_wav",
    "save_wav",
    "resample",
    "EdgeVAD",
    "SileroVADEngine",
    "OfflineASR",
    "FasterWhisperEngine",
    "COMMAND_ANALYZER",
    "MultilingualCommandAnalyzer",
    "AnalyzedCommand",
    "CONFIG",
    "normalize_text",
    "score_text"
]
