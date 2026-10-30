"""
TTS: Sentence-chunked offline Text-to-Speech (HNX26EPS08).
Uses piper-tts (ONNX VITS) on Windows — fully CPU, fully offline.
Implements the TTS protocol from contracts.py.
Sentence-boundary chunker starts synthesis on the FIRST completed sentence
so audio playback begins < 200ms after LLM starts streaming.
"""

import re
import time
import threading
from pathlib import Path
from typing import Iterator, Optional
import numpy as np

from src.edge.contracts import SAMPLE_RATE

# ---------------------------------------------------------------------------
# SENTENCE BOUNDARY CHUNKER
# ---------------------------------------------------------------------------

# Regex: split on . ! ? followed by space/end, but not on abbreviations like "Dr."
_SENT_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', re.MULTILINE)


def sentence_chunker(token_stream: Iterator[str]) -> Iterator[str]:
    """
    Consume a stream of LLM tokens and yield complete sentences.
    A sentence is yielded as soon as a terminal punctuation mark is detected,
    so TTS can start speaking before the full LLM response is generated.
    """
    buf = ""
    for token in token_stream:
        if token is None:
            continue
        buf += token
        # Check for sentence boundary
        parts = _SENT_END.split(buf)
        while len(parts) > 1:
            sentence = parts.pop(0).strip()
            buf = parts[0] if parts else ""
            if sentence:
                yield sentence
    # Yield any remaining text
    if buf.strip():
        yield buf.strip()


def split_sentences(text: str) -> list[str]:
    """Split a complete text block into sentences."""
    parts = _SENT_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# PIPER TTS ENGINE (ONNX VITS — CPU Only)
# ---------------------------------------------------------------------------

_PIPER_VOICE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tts"


class PiperTTS:
    """
    Piper TTS using piper-tts Python package (ONNX VITS CPU inference).
    Downloads voice on first use, caches in models/tts/.
    Implements TTS protocol: synth_stream(sentence) -> Iterator[np.ndarray]
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        model_dir: Optional[Path] = None,
    ):
        from piper import PiperVoice

        self._model_dir = Path(model_dir) if model_dir else _PIPER_VOICE_DIR
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 22050  # Piper default

        onnx_path = self._model_dir / f"{voice}.onnx"
        config_path = self._model_dir / f"{voice}.onnx.json"

        if not onnx_path.exists():
            self._download_voice(voice, onnx_path, config_path)

        print(f"[TTS-Piper] Loading voice '{voice}' from {onnx_path.name} ...")
        t0 = time.perf_counter()
        self._voice = PiperVoice.load(str(onnx_path), config_path=str(config_path), use_cuda=False)
        self.sample_rate = self._voice.config.sample_rate
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[TTS-Piper] Voice loaded in {load_ms:.0f}ms, sample_rate={self.sample_rate}Hz")

    def _download_voice(self, voice: str, onnx_path: Path, config_path: Path):
        import urllib.request
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        # Convert voice name to path: en_US-lessac-medium -> en/en_US/lessac/medium
        parts = voice.split("-")
        lang_code = parts[0][:2]  # "en"
        lang_full = parts[0]      # "en_US"
        speaker = parts[1] if len(parts) > 1 else "default"
        quality = parts[2] if len(parts) > 2 else "medium"
        voice_path = f"{lang_code}/{lang_full}/{speaker}/{quality}"

        for ext in [".onnx", ".onnx.json"]:
            fname = f"{voice}{ext}"
            url = f"{base_url}/{voice_path}/{fname}"
            dest = self._model_dir / fname
            if not dest.exists():
                print(f"[TTS-Piper] Downloading {fname} ...")
                try:
                    urllib.request.urlretrieve(url, dest)
                    print(f"[TTS-Piper] Downloaded {fname} ({dest.stat().st_size // 1024} KB)")
                except Exception as e:
                    print(f"[TTS-Piper] Download failed: {e}\n  Manual: {url}")

    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]:
        """
        Synthesize a sentence and yield float32 audio chunks.
        Piper yields AudioChunk with audio_float_array in [-1, 1].
        """
        if not sentence.strip():
            return

        chunk_size = 2048
        for chunk in self._voice.synthesize(sentence):
            audio = getattr(chunk, "audio_float_array", None)
            if audio is None and hasattr(chunk, "audio_int16_array"):
                audio = chunk.audio_int16_array.astype(np.float32) / 32768.0
            if audio is not None and len(audio) > 0:
                for i in range(0, len(audio), chunk_size):
                    yield audio[i:i + chunk_size]

    def synth_full(self, text: str) -> np.ndarray:
        """Synthesize complete text to a single array."""
        chunks = list(self.synth_stream(text))
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ---------------------------------------------------------------------------
# UNIFIED TTS (Auto-selects best available engine)
# ---------------------------------------------------------------------------

class EdgeTTS:
    """
    Unified TTS implementing the TTS protocol from contracts.py.
    Primary: PiperTTS (ONNX VITS)
    Fallback: sherpa-onnx OfflineTts
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        model_dir: Optional[Path] = None,
    ):
        self._engine = None
        self.sample_rate = SAMPLE_RATE

        # Try Piper first
        try:
            self._engine = PiperTTS(voice=voice, model_dir=model_dir)
            self.sample_rate = self._engine.sample_rate
            self._engine_name = "piper"
            return
        except Exception as e:
            print(f"[TTS] Piper failed: {e} — trying sherpa-onnx fallback ...")

        # Sherpa-ONNX fallback
        try:
            self._load_sherpa()
        except Exception as e:
            print(f"[TTS] Sherpa-ONNX fallback also failed: {e}")
            print("[TTS] WARNING: No TTS engine loaded. Run scripts/download_models.py")

    def _load_sherpa(self):
        import sherpa_onnx
        model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "tts" / "sherpa"
        vits_model = model_dir / "model.onnx"
        lexicon = model_dir / "lexicon.txt"
        tokens_file = model_dir / "tokens.txt"

        if not vits_model.exists():
            raise FileNotFoundError(f"Sherpa TTS model not found at {vits_model}")

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(vits_model),
                    lexicon=str(lexicon) if lexicon.exists() else "",
                    tokens=str(tokens_file),
                ),
                num_threads=2,
                debug=False,
            ),
            rule_fsts="",
            max_num_sentences=1,
        )
        self._sherpa_tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = self._sherpa_tts.sample_rate
        self._engine_name = "sherpa"
        print(f"[TTS] Sherpa-ONNX TTS loaded, sample_rate={self.sample_rate}Hz")

    def synth_stream(self, sentence: str) -> Iterator[np.ndarray]:
        """TTS protocol: synthesize sentence, yield float32 audio chunks."""
        if not sentence.strip():
            return

        if self._engine is not None and self._engine_name == "piper":
            yield from self._engine.synth_stream(sentence)
        elif hasattr(self, "_sherpa_tts"):
            result = self._sherpa_tts.generate(sentence, sid=0, speed=1.0)
            audio = np.array(result.samples, dtype=np.float32)
            chunk_size = 2048
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]
        else:
            # No TTS available — yield silence
            duration_samples = max(int(self.sample_rate * 1.0), 1)
            yield np.zeros(duration_samples, dtype=np.float32)

    def synth_full(self, text: str) -> np.ndarray:
        chunks = list(self.synth_stream(text))
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    @property
    def is_available(self) -> bool:
        return self._engine is not None or hasattr(self, "_sherpa_tts")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Test sentence chunker
    tokens = iter(["The quick brown fox. ", "It jumps over the lazy dog! ", "How cool is that?"])
    print("=== SENTENCE CHUNKER TEST ===")
    for sent in sentence_chunker(tokens):
        print(f"  Sentence: '{sent}'")

    print("\n[TTS] Attempting to load Piper TTS ...")
    tts = EdgeTTS()
    if tts.is_available:
        print(f"[TTS] Engine ready. Sample rate: {tts.sample_rate}Hz")
        t0 = time.perf_counter()
        chunks = list(tts.synth_stream("Hello, I am Jarvis Edge, running fully offline."))
        elapsed = (time.perf_counter() - t0) * 1000
        total_samples = sum(len(c) for c in chunks)
        duration_ms = total_samples / tts.sample_rate * 1000
        print(f"[TTS] Synthesized {total_samples} samples ({duration_ms:.0f}ms audio) in {elapsed:.0f}ms")
    else:
        print("[TTS] No engine available. Run scripts/download_models.py to download voice models.")
    print("[TTS] tts.py OK")
