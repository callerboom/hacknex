"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Problem 13: Multilingual Live Interactive Assistant & Demo UX
Author: Edge Audio, ASR & Multilingual Lead

Features:
- CLI Flags:
    --lang-mode {tier1_only, auto_all, locked:<xx>}
    --model {tiny, base, small}
    --translate
    --list-devices
    --headless
    --wav <path>
- Real-time listening with Silero VAD v5 + Faster-Whisper int8 on CPU
- Displays language tier (Tier 1: Verified, Tier 2: Verified, Tier 3: Unverified best-effort)
- Ambiguity reporting: shows top candidate languages if margin < 0.15
- Whitelisted action dispatcher with confidence gating (<0.40 rejected)
- 100% offline, CPU-only, UTF-8 safe Windows terminal output.
"""

import argparse
import os
import sys
import time
import numpy as np

# Ensure UTF-8 output on Windows terminals to prevent cp1252 charmap crashes
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.edge.audio_io import MicrophoneStream, AudioPlayer, load_wav
from src.edge.vad import EdgeVAD
from src.edge.asr import OfflineASR
from src.edge.router.command_analyzer import analyze_command
from src.edge.model_check import verify_all_models
from src.edge.config import CONFIG


def get_tier_info(lang_code: str) -> tuple[int, str]:
    """Returns tier info for English engine."""
    if lang_code == "en":
        return (1, "ENGLISH (Transcribe + Intent + Whitelisted Action)")
    else:
        return (2, "FALLBACK")


def test_microphone(duration_sec: float = 10.0):
    """Live interactive microphone tester showing VU level meter and Silero VAD probability."""
    print("=" * 78)
    print("  MICROPHONE DIAGNOSTIC & REAL-TIME VU LEVEL METER")
    print(f"  Listening to physical microphone for up to {duration_sec:.0f}s (or press Ctrl+C)...")
    print("  Speak into your laptop microphone to see the level bars jump!")
    print("=" * 78)

    mic = MicrophoneStream()
    mic.start()
    vad = EdgeVAD()

    t_end = time.perf_counter() + duration_sec
    chunk_count = 0
    speech_detected_count = 0
    max_rms = 0.0

    try:
        for chunk in mic.stream():
            chunk_count += 1
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            max_rms = max(max_rms, rms)
            prob = vad.engine.predict(chunk)

            # Build a live VU bar (scaled for laptop mic sensitivities)
            bar_len = min(25, int(rms * 2500))
            vu_bar = "#" * bar_len + "-" * (25 - bar_len)
            tag = " [🎙️ SPEECH DETECTED!]" if prob >= 0.35 else ""
            if prob >= 0.35:
                speech_detected_count += 1

            sys.stdout.write(f"\r  [{vu_bar}] Level: {rms:.5f} | VAD Prob: {prob:.2f}{tag:24}")
            sys.stdout.flush()

            if time.perf_counter() > t_end:
                break
    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")
    finally:
        mic.stop()
        print("\n" + "=" * 78)
        print("DIAGNOSTIC SUMMARY:")
        print(f"  * Audio Chunks Captured:  {chunk_count}")
        print(f"  * Peak RMS Volume:        {max_rms:.6f}")
        print(f"  * Speech Frames Detected: {speech_detected_count}")
        if speech_detected_count > 0:
            print("  ==> SUCCESS: Microphone and Silero VAD are detecting your voice perfectly!")
        else:
            print("  ==> NOTE: Peak level was quiet. Speak closer to the mic or raise Windows Mic Volume.")
        print("=" * 78)


def list_devices():
    """Lists host audio devices."""
    try:
        import sounddevice as sd
        print("\n=== AVAILABLE AUDIO DEVICES ===")
        print(sd.query_devices())
        print(f"\nDefault Input Device:  {sd.query_devices(kind='input')['name']}")
        print(f"Default Output Device: {sd.query_devices(kind='output')['name']}\n")
    except Exception as e:
        print(f"Failed to query sound devices: {e}")


def run_live_loop(
    lang_mode: str = "en",
    model_choice: str = "tiny_en",
    translate: bool = False,
    headless: bool = False,
    wav_file: str = "tests/assets/clip2_command.wav",
    hangover_ms: float = CONFIG.vad.hangover_ms,
    start_thresh: float = CONFIG.vad.start_threshold
):
    print("=" * 78)
    print("  HackNEX 2026: English On-Device Voice Engine")
    print("  Language: English (en) | 100% Offline | CPU-Only | Zero Cloud")
    print("=" * 78)

    # Resolve model directory: default to whisper_tiny_en
    model_dir = "models/asr/whisper_tiny_en"
    if model_choice in ("tiny_multi", "multilingual"):
        model_dir = "models/asr/whisper_tiny_multilingual"

    hangover_chunks = max(1, int(round(hangover_ms / 32.0)))

    print(f"\n[ACTIVE CONFIGURATION]")
    print(f"  * Whisper Model:       {os.path.basename(model_dir)} ({model_choice})")
    print(f"  * Language:            English (en)")
    print(f"  * Speech Hangover:     {hangover_ms:.0f} ms ({hangover_chunks} frames) [Natural human cadence]")
    print(f"  * Onset Sensitivity:   {start_thresh:.2f} [Sensitive conversational speech pickup]")
    print(f"  * Confidence Gate:     {CONFIG.asr.min_command_confidence:.2f}")


    # 1. Verify Model Integrity & Offline Proof
    print("\n[1/3] Verifying offline model files & SHA256 hashes...")
    try:
        verify_all_models("models/CHECKSUMS.sha256")
        print("      --> Checksums validated. Models are authentic and 100% offline.")
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        sys.exit(1)

    # 2. Initialize Engines
    print("\n[2/3] Initializing Silero VAD v5 and Faster-Whisper int8...")
    t0 = time.perf_counter()
    player = AudioPlayer()

    def on_started():
        print("\n[🎙️ SPEECH DETECTED] Listening to your complete sentence... (take your time, normal pauses allowed)", flush=True)

    def on_ended():
        print("[⏳ SENTENCE COMPLETE] Silence detected. Decoding speech with Whisper...", flush=True)

    vad = EdgeVAD(
        start_threshold=start_thresh,
        hangover_chunks=hangover_chunks,
        on_barge_in=player.stop,
        is_bot_speaking_fn=lambda: player.is_playing,
        on_speech_started=on_started,
        on_speech_ended=on_ended
    )
    asr = OfflineASR(whisper_model_dir=model_dir)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"      --> Initialized in {init_ms:.1f}ms.")

    # 3. Start Streaming Loop
    print("\n[3/3] Ready. Speak into the microphone (Ctrl+C to stop).")
    print("  Conversational Mode: Natural pauses (700-800ms) supported without cutting off.")
    print("  Try commands: 'check battery', 'open notepad', 'what time is it', 'open calculator', 'check memory'")

    if headless:
        print(f"\n[HEADLESS MODE] Feeding test audio: {wav_file}")
        audio = load_wav(wav_file, target_sr=16000)
        chunks = [audio[i:i+512] for i in range(0, len(audio), 512)]
        stream = iter(chunks)
    else:
        mic = MicrophoneStream()
        mic.start()
        print("      --> Microphone stream started successfully at 16000Hz mono.")
        print("\n[LISTENING] Ready! Say your command (e.g. 'check battery', 'open notepad', 'what time is it')...")
        stream = mic.stream()

    turn_count = 0
    task_mode = "translate" if translate else "transcribe"

    try:
        for segment in vad.segments(stream):
            turn_count += 1
            print(f"\n" + "-" * 78)
            print(f"[TURN {turn_count}] SPEECH DETECTED (Audio Duration: {segment.duration_sec:.2f}s)")
            print(f"  * Audio-Time Endpoint Delay: {segment.endpointing_delay_ms:.1f} ms (real hangover)")
            print(f"  * VAD Compute Time:          {segment.vad_compute_ms:.2f} ms")

            # ASR Decoding
            t_asr0 = time.perf_counter()
            transcript = asr.transcribe(
                segment.audio,
                language_mode=lang_mode,
                task=task_mode
            )
            wall_asr_ms = (time.perf_counter() - t_asr0) * 1000.0

            tier_num, tier_desc = get_tier_info(transcript.lang)

            print(f"\n  --- ASR DECODING RESULTS ---")
            print(f"  * Detected Language:         [{transcript.lang.upper()}] (prob: {transcript.lang_prob:.3f})")
            print(f"  * Language Tier:             Tier {tier_num} - {tier_desc}")
            print(f"  * ASR Inference Latency:     {transcript.asr_ms:.1f} ms")
            print(f"  * Confidence Score:          {transcript.confidence:.2f}")

            # Show candidates on ambiguity
            if transcript.lang_candidates and len(transcript.lang_candidates) > 1:
                top_cands = sorted(transcript.lang_candidates.items(), key=lambda x: x[1], reverse=True)[:3]
                cands_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_cands)
                print(f"  * Candidate Languages:       {cands_str}")

            print(f"  >>> TRANSCRIPT:              \"{transcript.text}\"")

            if transcript.translated and transcript.translation_text:
                print(f"  >>> TRANSLATION (EN):        \"{transcript.translation_text}\"")

            # Intent Analysis & Whitelisted Action Dispatch
            if transcript.text.strip():
                t_act0 = time.perf_counter()
                cmd = analyze_command(
                    user_text=transcript.text,
                    detected_lang=transcript.lang,
                    confidence=transcript.confidence,
                    is_translated=transcript.translated,
                    translation_text=transcript.translation_text,
                    execute_actions=True
                )
                act_ms = (time.perf_counter() - t_act0) * 1000.0

                print(f"\n  --- INTENT ROUTING & ACTION DISPATCH ---")
                print(f"  * Intent Classified:         [{cmd.intent}] (matched in {cmd.latency_ms:.2f}ms)")
                print(f"  * Action Executed:           {'YES (Triggered OS Binary)' if cmd.action_executed else 'NO (Query only / Unmapped)'}")
                print(f"  >>> VOICE RESPONSE:          \"{cmd.response_en}\"")

            if headless:
                break

    except KeyboardInterrupt:
        print("\n\nDemo gracefully stopped.")
    finally:
        if not headless and "mic" in locals():
            mic.stop()


def main():
    parser = argparse.ArgumentParser(description="HackNEX 2026 English EdgeVoice Assistant")
    parser.add_argument("--test-mic", action="store_true", help="Test microphone and display live VU level meter")
    parser.add_argument("--lang-mode", choices=["en"], default="en", help="Language mode (English only)")
    parser.add_argument("--model", choices=["tiny_en", "tiny_multi"], default="tiny_en", help="Whisper model tier")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices")
    parser.add_argument("--headless", action="store_true", help="Run with synthetic audio / WAV file")
    parser.add_argument("--wav", type=str, default="tests/assets/clip2_command.wav", help="WAV file for headless test")
    parser.add_argument("--hangover-ms", type=float, default=CONFIG.vad.hangover_ms, help="Silence hangover in ms (default: 768ms)")
    parser.add_argument("--start-thresh", type=float, default=CONFIG.vad.start_threshold, help="VAD start probability threshold (default: 0.35)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.test_mic:
        test_microphone()
        return

    run_live_loop(
        lang_mode=args.lang_mode,
        model_choice=args.model,
        headless=args.headless,
        wav_file=args.wav,
        hangover_ms=args.hangover_ms,
        start_thresh=args.start_thresh
    )


if __name__ == "__main__":
    main()
