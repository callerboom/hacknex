"""
Model Downloader: Pulls all required models for HNX26EPS08 EdgeVoice.
Runs once before demo. Skips already-downloaded files.
"""

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 5)
        sys.stdout.write(f"\r  [{bar:<20}] {pct:.1f}%  ({downloaded//1024}KB / {total_size//1024}KB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()
    else:
        sys.stdout.write(f"\r  Downloaded {downloaded//1024}KB...")
        sys.stdout.flush()


def download(url: str, dest: Path, label: str):
    if dest.exists():
        print(f"  [SKIP] {label} already exists ({dest.stat().st_size//1024}KB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DOWNLOAD] {label}")
    print(f"    URL: {url}")
    print(f"    Dest: {dest}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print(f"  [OK] {label} ({dest.stat().st_size//1024}KB)")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        if dest.exists():
            dest.unlink()


DOWNLOADS = [
    # ============================================================
    # VAD: Silero VAD v4 ONNX
    # ============================================================
    {
        "label": "Silero VAD ONNX v4",
        "url": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "dest": MODELS_DIR / "vad" / "silero_vad.onnx",
    },

    # ============================================================
    # ASR: faster-whisper — handled by the library itself on first run.
    # We pre-download the tiny.en model manually for offline use.
    # ============================================================
    # faster-whisper downloads from HuggingFace automatically on first use.
    # For fully offline, use: huggingface-cli download Systran/faster-whisper-tiny.en

    # ============================================================
    # LLM: Qwen2.5-0.5B-Instruct Q5_K_M GGUF (~350MB)
    # Smallest capable model for daily-helper tasks.
    # ============================================================
    {
        "label": "Qwen2.5-0.5B-Instruct Q5_K_M GGUF (LLM)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q5_k_m.gguf",
        "dest": MODELS_DIR / "llm" / "qwen2.5-0.5b-instruct-q5_k_m.gguf",
    },

    # ============================================================
    # TTS: Piper voice — en_US-lessac-medium
    # ============================================================
    {
        "label": "Piper TTS en_US-lessac-medium ONNX",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "dest": MODELS_DIR / "tts" / "en_US-lessac-medium.onnx",
    },
    {
        "label": "Piper TTS en_US-lessac-medium config JSON",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    # ============================================================
    # Wake-word: openWakeWord (Hey Jarvis) ONNX models
    # ============================================================
    {
        "label": "openWakeWord Hey Jarvis ONNX",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
        "dest": MODELS_DIR / "wakeword" / "hey_jarvis_v0.1.onnx",
    },
    {
        "label": "openWakeWord Melspectrogram ONNX",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        "dest": MODELS_DIR / "wakeword" / "melspectrogram.onnx",
    },
    {
        "label": "openWakeWord Embedding ONNX",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "dest": MODELS_DIR / "wakeword" / "embedding_model.onnx",
    },
]

# ============================================================
# PYTHON PACKAGE CHECKS
# ============================================================

REQUIRED_PACKAGES = {
    "faster_whisper": "faster-whisper",
    "llama_cpp": "llama-cpp-python",
    "piper": "piper-tts",
    "onnxruntime": "onnxruntime",
    "sounddevice": "sounddevice",
    "psutil": "psutil",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "websockets": "websockets",
    "yaml": "pyyaml",
    "numpy": "numpy",
    "openwakeword": "openwakeword",
}


def check_packages():
    print("\n=== Checking Python packages ===\n")
    missing = []
    for module, pkg in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)
    return missing


def main():
    print("=" * 60)
    print("  EdgeVoice Model Downloader (HNX26EPS08)")
    print("=" * 60)

    # Check packages
    missing = check_packages()
    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        print(f"    Run: pip install {' '.join(missing)}\n")
    else:
        print("\n[OK] All packages installed.\n")

    # Download models
    print("\n=== Downloading models ===\n")
    for item in DOWNLOADS:
        download(item["url"], item["dest"], item["label"])

    # Pre-build TTS cache if TTS is available
    print("\n=== Pre-building TTS cache ===\n")
    try:
        from src.edge.tts import EdgeTTS
        from src.edge.tts_cache import TTSCache
        tts = EdgeTTS()
        if tts.is_available:
            cache = TTSCache(tts_engine=tts, sample_rate=tts.sample_rate)
            cache.prebuild()
        else:
            print("  TTS not available — cache skipped.")
    except Exception as e:
        print(f"  TTS cache prebuild skipped: {e}")

    # Pre-download faster-whisper tiny.en
    print("\n=== Pre-fetching faster-whisper tiny.en ===\n")
    try:
        from faster_whisper import WhisperModel
        print("  Loading faster-whisper tiny.en (downloads if not cached) ...")
        m = WhisperModel("tiny.en", device="cpu", compute_type="int8", num_workers=1, cpu_threads=2)
        print("  [OK] faster-whisper tiny.en ready.")
        del m
    except Exception as e:
        print(f"  [SKIP] faster-whisper download skipped: {e}")

    print("\n" + "=" * 60)
    print("  All models ready. Run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
