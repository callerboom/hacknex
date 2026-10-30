"""Voice Pipeline & Real-Time Action Execution Engine.

Connects:
Microphone / Audio Stream -> Person 2 STT -> clean user_text -> Person 1 route_intent(user_text) -> Execute Action -> Voice & Text Response.

Features:
- Sub-millisecond local fast-path execution (<1ms).
- Zero-latency application launching (Calculator, Notepad, Browser, Explorer, VS Code, Settings).
- Real-time hardware telemetry (Battery, RAM, CPU, Disk, WiFi, Internet).
- Offline speech synthesis (Windows SAPI / System.Speech).
- Clean fallback to conversational LLM when route_intent() returns None.
"""

import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from src.edge.router import route_intent


def speak_text_offline(text: str, non_blocking: bool = True) -> None:
    """Speak response text offline using native Windows System.Speech SAPI.

    Args:
        text: Response message string to synthesize.
        non_blocking: Whether to speak asynchronously in a background thread.
    """
    if not text or not text.strip():
        return

    # Clean text of markdown formatting before speaking
    clean_speech = text.replace("*", "").replace("#", "").replace("`", "").strip()

    def _speak_worker():
        try:
            if sys.platform == "win32":
                # Escape single and double quotes for PowerShell
                escaped = clean_speech.replace("'", "''").replace('"', '`"')
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = 1; $synth.Speak('{escaped}')",
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except Exception:
            pass

    if non_blocking:
        t = threading.Thread(target=_speak_worker, daemon=True)
        t.start()
    else:
        _speak_worker()


class VoicePipeline:
    """End-to-end voice and command pipeline for Jarvis-Edge."""

    def __init__(
        self,
        llm_fallback_fn: Optional[Callable[[str], str]] = None,
        enable_tts: bool = True,
    ):
        """Initialize voice pipeline with optional conversational LLM fallback.

        Args:
            llm_fallback_fn: Callable invoked when fast-router returns None.
            enable_tts: Whether to speak responses via offline TTS.
        """
        self.llm_fallback_fn = llm_fallback_fn or self._default_llm_fallback
        self.enable_tts = enable_tts

    @staticmethod
    def _default_llm_fallback(user_text: str) -> str:
        """Default fallback handler when no local fast-path matches."""
        return f"[LLM Fallback]: Query '{user_text}' forwarded to conversational LLM engine."

    def process_transcript(self, raw_transcript: str) -> Dict[str, Any]:
        """Process a recognized voice transcript from Person 2 STT.

        Flow:
            Person 2 STT output -> clean string -> route_intent -> execute -> voice/text response.

        Args:
            raw_transcript: The recognized text output from STT.

        Returns:
            Dictionary containing:
                - 'input_text': Cleaned user transcript
                - 'handled_by': 'fast_router' or 'llm_fallback'
                - 'response': Text response message
                - 'latency_ms': Routing execution latency in milliseconds
        """
        if raw_transcript is None:
            user_text = ""
        else:
            user_text = str(raw_transcript).strip()

        start_time = time.perf_counter()

        # Step 1: Fast rule-based router (Person 1)
        fast_response = route_intent(user_text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if fast_response is not None:
            handled_by = "fast_router"
            response = fast_response
        else:
            handled_by = "llm_fallback"
            response = self.llm_fallback_fn(user_text)

        # Step 2: Trigger speech synthesis (non-blocking)
        if self.enable_tts and response:
            speak_text_offline(response, non_blocking=True)

        return {
            "input_text": user_text,
            "handled_by": handled_by,
            "response": response,
            "latency_ms": round(elapsed_ms, 3),
        }

    def run_interactive_loop(self) -> None:
        """Run interactive terminal listener loop for voice and text commands."""
        print("=" * 70)
        print("  JARVIS-EDGE : OFFLINE VOICE & FAST-PATH PIPELINE ACTIVE")
        print("  Type any command in English, Tamil, Tanglish, Hindi, Telugu,")
        print("  Malayalam, or Kannada (or 'exit' / 'quit' to stop).")
        print("=" * 70)

        while True:
            try:
                user_input = input("\n[VOICE / TEXT INPUT] >>> ")
                if user_input.strip().lower() in ("exit", "quit", "q"):
                    print("Exiting Jarvis Voice Pipeline. Goodbye!")
                    break

                result = self.process_transcript(user_input)
                print(f"[{result['handled_by'].upper()}] ({result['latency_ms']} ms):")
                print(f"  {result['response']}")

            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break


def execute_voice_command(user_text: str, enable_tts: bool = False) -> Dict[str, Any]:
    """Helper entry point to execute a single voice transcript immediately.

    Args:
        user_text: Recognized voice transcript string.
        enable_tts: Whether to trigger offline voice synthesizer.

    Returns:
        Execution result dictionary with response and latency metrics.
    """
    pipeline = VoicePipeline(enable_tts=enable_tts)
    return pipeline.process_transcript(user_text)


if __name__ == "__main__":
    pipeline = VoicePipeline(enable_tts=True)
    pipeline.run_interactive_loop()
