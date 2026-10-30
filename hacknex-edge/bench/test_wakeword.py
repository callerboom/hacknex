"""
Wake-Word Benchmark & Validation Suite (HNX26EPS08).
Measures:
- Miss rate on real positive utterances containing 'Hey Jarvis'
- False accept rate per hour on silence and background acoustic noise
- CPU utilization (ms per second of audio) and peak RAM footprint
Outputs real measured metrics to results/wakeword.csv.
"""

import sys
import os
import time
from pathlib import Path
import psutil
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edge.wakeword import WakeWordDetector
from src.edge.tts import EdgeTTS

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = RESULTS_DIR / "wakeword.csv"


def resample_22k_to_16k(audio_22k: np.ndarray) -> np.ndarray:
    """Resample audio from 22050 Hz to 16000 Hz using linear interpolation."""
    if len(audio_22k) == 0:
        return np.zeros(0, dtype=np.float32)
    target_len = int(len(audio_22k) * 16000 / 22050)
    audio_16k = np.interp(
        np.linspace(0, len(audio_22k), target_len, endpoint=False),
        np.arange(len(audio_22k)),
        audio_22k,
    ).astype(np.float32)
    return audio_16k


def stream_audio_through_detector(detector: WakeWordDetector, audio_16k: np.ndarray, chunk_size: int = 1280):
    """Feed audio in 1280-sample (80ms) chunks to detector and return (any_fired, max_score)."""
    detector.reset()
    any_fired = False
    max_score = 0.0

    for i in range(0, len(audio_16k), chunk_size):
        chunk = audio_16k[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        fired, active, score = detector.process_chunk(chunk)
        if fired:
            any_fired = True
        max_score = max(max_score, score)

    return any_fired, max_score


def main():
    print("=" * 60)
    print("  EdgeVoice Wake-Word Evaluation (openWakeWord ONNX)")
    print("=" * 60)

    proc = psutil.Process()
    ram_before_mb = proc.memory_info().rss / (1024 ** 2)

    detector = WakeWordDetector(threshold=0.5, wake_window_s=5.0)
    if not detector.is_available:
        print("[ERROR] WakeWordDetector could not be loaded!")
        sys.exit(1)

    ram_after_mb = proc.memory_info().rss / (1024 ** 2)
    model_ram_mb = ram_after_mb - ram_before_mb
    print(f"Model loaded: {detector.model_name}")
    print(f"Detector RAM footprint: {model_ram_mb:.1f} MB (Process RSS: {ram_after_mb:.1f} MB)")

    # Initialize TTS for real spoken waveform generation
    print("\n[1/4] Generating spoken test phrases via local TTS engine ...")
    tts = EdgeTTS()

    positives = [
        "Hey Jarvis.",
        "Hey Jarvis, what time is it?",
        "Hey Jarvis, can you hear me?",
        "Hey Jarvis, what is today's date?",
        "Hey Jarvis, calculate 48 times 12.",
        "Hey Jarvis, tell me a quick fact.",
        "Hey Jarvis, good morning.",
        "Hey Jarvis, check system status.",
    ]

    negatives = [
        "What time is it right now?",
        "Can you hear me speaking?",
        "Today is a sunny and pleasant day.",
        "Calculate 48 times 12 please.",
        "Good morning to everyone in the room.",
        "System status report is clear.",
        "Tell me a story about outer space.",
        "Who is the current world champion?",
    ]

    # Evaluate Positives
    print("\n[2/4] Evaluating positive wake-word utterances ...")
    pos_detected = 0
    pos_scores = []
    for phrase in positives:
        wave_22k = tts.synth_full(phrase)
        wave_16k = resample_22k_to_16k(wave_22k)
        fired, max_score = stream_audio_through_detector(detector, wave_16k)
        pos_scores.append(max_score)
        if fired or max_score >= 0.5:
            pos_detected += 1
            status = "HIT"
        else:
            status = "MISS"
        print(f"  [{status}] Score: {max_score:.3f} | '{phrase}'")

    miss_count = len(positives) - pos_detected
    miss_rate_pct = (miss_count / len(positives)) * 100.0
    print(f"\nPositive Accuracy: {pos_detected}/{len(positives)} ({100.0 - miss_rate_pct:.1f}%) | Miss Rate: {miss_rate_pct:.1f}%")

    # Evaluate Negative Speech
    print("\n[3/4] Evaluating negative speech utterances (no wake-word) ...")
    neg_false_alarms = 0
    for phrase in negatives:
        wave_22k = tts.synth_full(phrase)
        wave_16k = resample_22k_to_16k(wave_22k)
        fired, max_score = stream_audio_through_detector(detector, wave_16k)
        if fired or max_score >= 0.5:
            neg_false_alarms += 1
            status = "FALSE-ALARM"
        else:
            status = "CORRECT-REJECT"
        print(f"  [{status}] Score: {max_score:.3f} | '{phrase}'")

    # Evaluate Silence & Noise stream
    print("\n[4/4] Evaluating silence & acoustic noise streaming + latency overhead ...")
    detector.reset()
    total_audio_sec = 60.0  # 60 seconds of streamed audio
    num_samples = int(total_audio_sec * 16000)

    # 30s silence + 30s light ambient noise
    noise = np.random.normal(0, 0.02, num_samples // 2).astype(np.float32)
    silence = np.zeros(num_samples // 2, dtype=np.float32)
    test_stream = np.concatenate([silence, noise])

    chunk_size = 1280
    stream_false_accepts = 0

    t_cpu_start = time.process_time()
    t_wall_start = time.perf_counter()

    for i in range(0, len(test_stream), chunk_size):
        chunk = test_stream[i : i + chunk_size]
        fired, active, score = detector.process_chunk(chunk)
        if fired:
            stream_false_accepts += 1

    t_cpu_end = time.process_time()
    t_wall_end = time.perf_counter()

    cpu_time_total_s = t_cpu_end - t_cpu_start
    cpu_ms_per_audio_sec = (cpu_time_total_s / total_audio_sec) * 1000.0
    wall_time_s = t_wall_end - t_wall_start
    rtf = wall_time_s / total_audio_sec

    # Scale false accepts to per-hour
    fa_per_hour = (stream_false_accepts / total_audio_sec) * 3600.0

    print(f"Processed {total_audio_sec:.1f}s audio in {wall_time_s:.2f}s (RTF: {rtf:.3f}x real-time)")
    print(f"CPU time per audio second: {cpu_ms_per_audio_sec:.2f} ms")
    print(f"False accepts on silence/noise: {stream_false_accepts} in {total_audio_sec:.0f}s ({fa_per_hour:.2f} / hr)")
    print(f"Negative speech false alarms: {neg_false_alarms}/{len(negatives)}")

    peak_ram_mb = proc.memory_info().rss / (1024 ** 2)

    # Write results to CSV
    header = "metric,value,unit,notes\n"
    rows = [
        f"model_name,{detector.model_name},string,openWakeWord ONNX CPU\n",
        f"threshold,{detector.threshold if hasattr(detector, 'threshold') else 0.5},float,detection score cutoff\n",
        f"wake_window_s,5.0,seconds,active conversation window\n",
        f"positives_tested,{len(positives)},count,synthesized real speech\n",
        f"positives_detected,{pos_detected},count,wake-word hits\n",
        f"miss_rate_pct,{miss_rate_pct:.1f},percent,false negatives\n",
        f"negatives_tested,{len(negatives)},count,speech without wake-word\n",
        f"negatives_rejected,{len(negatives) - neg_false_alarms},count,correct rejections\n",
        f"silence_noise_seconds,{total_audio_sec:.1f},seconds,tested stream\n",
        f"false_accepts_per_hour,{fa_per_hour:.2f},per_hour,silence + acoustic noise\n",
        f"cpu_ms_per_audio_sec,{cpu_ms_per_audio_sec:.2f},ms/s,processing cost on CPU\n",
        f"real_time_factor,{rtf:.4f},ratio,ratio to real-time (< 1.0 is faster than real-time)\n",
        f"detector_ram_mb,{model_ram_mb:.1f},MB,incremental RAM footprint\n",
        f"peak_process_ram_mb,{peak_ram_mb:.1f},MB,total RSS during run\n",
    ]

    with open(CSV_OUT, "w", encoding="utf-8") as f:
        f.write(header + "".join(rows))

    print(f"\n[OK] Results saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
