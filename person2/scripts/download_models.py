"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Reproducible One-Time Model Downloader & Checksum Generator
Author: Person 2 (Edge Audio Lead)

Problem 2 & Problem 10:
Supports downloading Faster-Whisper tiny.en, tiny-multilingual, base-multilingual, and small-multilingual.
Prints disk sizes, warns about RAM/latency for larger tiers, and hashes all models to models/CHECKSUMS.sha256.

Usage:
    python scripts/download_models.py --model tiny-multi  (Default)
    python scripts/download_models.py --model base-multi  (Higher accuracy, ~1.8x latency)
    python scripts/download_models.py --model all
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request


def download_file(url: str, dest_path: str, expected_size: int = 0):
    """Downloads a file with resumable chunks and retry logic."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    current_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    if expected_size > 0 and current_size >= expected_size:
        print(f"  [OK] {os.path.basename(dest_path)} already exists ({current_size} bytes).")
        return

    print(f"  [DOWNLOADING] {os.path.basename(dest_path)} from {url}...")
    for attempt in range(5):
        current_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        req = urllib.request.Request(url)
        if current_size > 0:
            req.add_header("Range", f"bytes={current_size}-")

        try:
            with urllib.request.urlopen(req, timeout=20) as resp, open(dest_path, "ab" if current_size > 0 else "wb") as f:
                while chunk := resp.read(65536):
                    f.write(chunk)
            break
        except Exception as e:
            print(f"    Retry {attempt + 1}/5 ({e})...")
            time.sleep(1)

    print(f"  [FINISHED] {os.path.basename(dest_path)} ({os.path.getsize(dest_path)} bytes).")


def download_whisper_model(repo_id: str, target_dir: str, ram_warning: str):
    """Fetches CTranslate2 int8 faster-whisper model files from HuggingFace."""
    print(f"\nTarget Directory: {target_dir}")
    print(f"Resource Profile: {ram_warning}")

    base_hf = f"https://huggingface.co/{repo_id}/resolve/main/"
    files = ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"]
    for fname in files:
        download_file(base_hf + fname, os.path.join(target_dir, fname))


def update_checksums():
    """Generates and writes SHA256 hashes for all locally present models."""
    print("\nUpdating SHA256 checksums in models/CHECKSUMS.sha256...")
    checksum_lines = []

    # Check VAD
    vad_path = "models/vad/silero_vad.onnx"
    if os.path.exists(vad_path):
        checksum_lines.append(f"{compute_hash(vad_path)}  {vad_path}")

    # Check all ASR models
    asr_root = "models/asr"
    if os.path.exists(asr_root):
        for model_name in sorted(os.listdir(asr_root)):
            model_dir = os.path.join(asr_root, model_name)
            if os.path.isdir(model_dir):
                for fname in ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]:
                    fpath = os.path.join(model_dir, fname)
                    if os.path.exists(fpath):
                        norm_path = fpath.replace("\\", "/")
                        checksum_lines.append(f"{compute_hash(fpath)}  {norm_path}")

    with open("models/CHECKSUMS.sha256", "w", encoding="utf-8") as out:
        out.write("\n".join(checksum_lines) + "\n")
    print(f"Hashed {len(checksum_lines)} model artifacts to models/CHECKSUMS.sha256")


def compute_hash(fpath: str) -> str:
    hasher = hashlib.sha256()
    with open(fpath, "rb") as f:
        while b := f.read(65536):
            hasher.update(b)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="HackNEX 2026 Model Downloader")
    parser.add_argument(
        "--model",
        choices=["tiny-en", "tiny-multi", "base-multi", "small-multi", "all"],
        default="tiny-multi",
        help="Select which Whisper model to fetch"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HackNEX 2026: Pre-fetching Offline Models (One-Time Setup)")
    print("=" * 70)

    # 1. Silero VAD v5 ONNX Model (~2.3 MB)
    vad_url = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
    download_file(vad_url, "models/vad/silero_vad.onnx", expected_size=2327524)

    # 2. Whisper Models
    if args.model in ("tiny-en", "all"):
        download_whisper_model(
            "Systran/faster-whisper-tiny.en",
            "models/asr/whisper_tiny_en",
            "Disk ~78 MB | RAM ~150 MB | Latency ~300ms (English only)"
        )

    if args.model in ("tiny-multi", "all"):
        download_whisper_model(
            "Systran/faster-whisper-tiny",
            "models/asr/whisper_tiny_multilingual",
            "Disk ~78 MB | RAM ~260 MB | Latency ~550ms (~99 languages, fast)"
        )

    if args.model in ("base-multi", "all"):
        download_whisper_model(
            "Systran/faster-whisper-base",
            "models/asr/whisper_base_multilingual",
            "WARNING: Disk ~145 MB | RAM ~500 MB | Latency ~1000ms (~1.8x vs tiny)"
        )

    if args.model in ("small-multi", "all"):
        download_whisper_model(
            "Systran/faster-whisper-small",
            "models/asr/whisper_small_multilingual",
            "WARNING: Disk ~460 MB | RAM ~1.2 GB | Latency ~2200ms (~3.8x vs tiny)"
        )

    # 3. Update checksums & verify
    update_checksums()
    from src.edge.model_check import verify_all_models
    verify_all_models()
    print("\nALL LOCAL MODELS VERIFIED. System is 100% offline-ready!")


if __name__ == "__main__":
    main()
