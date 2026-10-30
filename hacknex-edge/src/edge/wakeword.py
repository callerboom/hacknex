"""
Wake-Word Detection Engine for EdgeVoice (HNX26EPS08).
Uses openWakeWord (ONNX runtime on CPU) on 16 kHz audio chunks.
Runs only when idle; zero latency added to speech-to-reply pipeline.
Falls back silently to always-listening VAD if models or package are missing.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("wakeword")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "wakeword"
DEFAULT_WAKEWORD_MODEL = DEFAULT_MODEL_DIR / "hey_jarvis_v0.1.onnx"
DEFAULT_MELSPEC_MODEL = DEFAULT_MODEL_DIR / "melspectrogram.onnx"
DEFAULT_EMBED_MODEL = DEFAULT_MODEL_DIR / "embedding_model.onnx"


class WakeWordDetector:
    """
    On-device wake-word detector using openWakeWord (ONNX, CPU-only).
    Listens for 'Hey Jarvis' on 16 kHz 16-bit mono PCM chunks.
    Manages active listening window (wake_window_s) that extends while user speaks.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        threshold: float = 0.5,
        wake_window_s: float = 5.0,
    ):
        self._threshold = threshold
        self._wake_window_s = wake_window_s
        self._active_until = 0.0
        self._model = None
        self._model_name = "hey_jarvis_v0.1"
        self._available = False
        self._sample_rate = 16000

        self._init_model(model_path)

    def _init_model(self, model_path: Optional[str | Path] = None) -> None:
        target_model = Path(model_path) if model_path else DEFAULT_WAKEWORD_MODEL
        melspec = DEFAULT_MELSPEC_MODEL
        embed = DEFAULT_EMBED_MODEL

        if not target_model.exists():
            logger.warning(
                f"[WakeWord] Model not found at {target_model} — falling back to always-listening VAD."
            )
            return

        try:
            import openwakeword
            from openwakeword.model import Model

            kwargs = {
                "wakeword_models": [str(target_model)],
                "inference_framework": "onnx",
            }
            if melspec.exists() and embed.exists():
                kwargs["melspec_model_path"] = str(melspec)
                kwargs["embedding_model_path"] = str(embed)

            self._model = Model(**kwargs)
            self._model_name = target_model.stem
            self._available = True
            logger.info(f"[WakeWord] Loaded model '{self._model_name}' (CPU ONNX, threshold={self._threshold})")
        except Exception as e:
            logger.warning(
                f"[WakeWord] Failed to initialize openwakeword: {e} — falling back to always-listening VAD."
            )
            self._model = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_active(self) -> bool:
        """True if the assistant is within the active listening window."""
        return time.time() < self._active_until

    def extend_window(self) -> None:
        """Extend the active window when speech continues."""
        self._active_until = time.time() + self._wake_window_s

    def reset_window(self) -> None:
        """Close the active window."""
        self._active_until = 0.0

    def process_chunk(self, chunk: np.ndarray) -> Tuple[bool, bool, float]:
        """
        Process a 16 kHz audio chunk (float32 or int16).
        Returns:
            (wake_fired, is_window_active, score)
        """
        if not self._available or self._model is None:
            # If detector is unavailable, pass-through as active
            return False, True, 0.0

        # Convert float32 [-1, 1] to int16 [-32768, 32767] expected by openwakeword
        if chunk.dtype == np.float32 or np.issubdtype(chunk.dtype, np.floating):
            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            pcm16 = chunk.astype(np.int16)

        try:
            prediction = self._model.predict(pcm16)
            score = 0.0
            if prediction:
                for k, v in prediction.items():
                    if k.startswith(self._model_name) or self._model_name in k:
                        score = float(v)
                        break
                    score = max(score, float(v))

            wake_fired = score >= self._threshold
            if wake_fired:
                self.extend_window()
                logger.info(f"[WakeWord] Wake-word '{self._model_name}' fired! (score={score:.3f})")

            return wake_fired, self.is_active(), score

        except Exception as e:
            logger.debug(f"[WakeWord] Prediction error: {e}")
            return False, self.is_active(), 0.0

    def reset(self) -> None:
        """Reset internal buffers."""
        if self._model:
            try:
                self._model.reset()
            except Exception:
                pass
        self._active_until = 0.0
