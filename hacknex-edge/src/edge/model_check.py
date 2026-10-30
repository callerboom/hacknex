"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Problem 10: Model Integrity, Distinction & Offline Enforcement Guard
Author: Person 2 (Edge Audio Lead)

Features:
- Enforces 100% offline mode via environment variables (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE)
- Validates cryptographic SHA256 checksums of all on-device models
- Verifies that whisper_tiny_en and whisper_tiny_multilingual are distinct models
- Checks vocabulary sizes (tiny.en: 51864, multilingual: 51865)
- Confirms is_multilingual=True flag for multilingual models
- Rejects duplicates, missing, or corrupted weights with explicit actionable error messages
"""

import hashlib
import json
import os
import sys

# Enforce strict offline operation at import time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def compute_sha256(file_path: str) -> str:
    """Computes SHA256 hex digest for a given file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def count_vocab_size(vocab_path: str) -> int:
    """Returns number of lines (tokens) in a vocabulary file."""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)



def is_model_multilingual(model_dir: str) -> bool:
    """
    Checks if a model directory represents a multilingual Whisper model.
    Inspects config.json lang_ids or vocabulary.txt size (51865 vs 51864).
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                lang_ids = cfg.get("lang_ids", [])
                if len(lang_ids) >= 99:
                    return True
        except Exception:
            pass

    vocab_path = os.path.join(model_dir, "vocabulary.txt")
    if os.path.exists(vocab_path):
        vocab_len = count_vocab_size(vocab_path)
        if vocab_len >= 51865:
            return True

    return False


def verify_models(checksum_path: str = "models/CHECKSUMS.sha256") -> bool:
    """
    Verifies all local model files against expected SHA256 checksums.
    Raises FileNotFoundError if any file is missing,
    or ValueError if any file is corrupted or modified.
    """
    if not os.path.exists(checksum_path):
        raise FileNotFoundError(
            f"Checksum file missing: '{checksum_path}'. "
            "Please run 'python scripts/download_models.py' to generate model checksums."
        )

    with open(checksum_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        expected_sha, relative_path = parts[0], parts[1].replace("\\", "/")

        if not os.path.exists(relative_path):
            raise FileNotFoundError(
                f"\n[CRITICAL ERROR] Required offline model is missing: '{relative_path}'\n"
                f"Competition rule violation: Silent downloads are forbidden.\n"
                f"Please run 'python scripts/download_models.py' before running offline."
            )

        actual_sha = compute_sha256(relative_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"\n[INTEGRITY FAILURE] Model file '{relative_path}' is corrupt or modified!\n"
                f"Expected SHA256: {expected_sha}\n"
                f"Actual SHA256:   {actual_sha}"
            )

    return True


def verify_models_distinct(
    en_model_dir: str = "models/asr/whisper_tiny_en",
    multi_model_dir: str = "models/asr/whisper_tiny_multilingual"
) -> bool:
    """
    Problem 10: Verifies that the English and Multilingual models are genuine and distinct.
    Ensures different SHA256 hashes, distinct vocabulary counts, and correct multilingual flag.
    """
    en_bin = os.path.join(en_model_dir, "model.bin")
    multi_bin = os.path.join(multi_model_dir, "model.bin")

    if not os.path.exists(en_bin):
        raise FileNotFoundError(f"Missing English model binary: {en_bin}")
    if not os.path.exists(multi_bin):
        raise FileNotFoundError(f"Missing Multilingual model binary: {multi_bin}")

    en_sha = compute_sha256(en_bin)
    multi_sha = compute_sha256(multi_bin)

    if en_sha == multi_sha:
        raise ValueError(
            f"[CRITICAL ERROR] Model duplicate detected! English and Multilingual models share identical SHA256: {en_sha}"
        )

    # Check vocabulary sizes
    en_vocab = os.path.join(en_model_dir, "vocabulary.txt")
    multi_vocab = os.path.join(multi_model_dir, "vocabulary.txt")
    if os.path.exists(en_vocab) and os.path.exists(multi_vocab):
        en_size = count_vocab_size(en_vocab)
        multi_size = count_vocab_size(multi_vocab)
        if en_size != 51864:
            raise ValueError(f"English model vocabulary size unexpected: {en_size} (expected 51864)")
        if multi_size != 51865:
            raise ValueError(f"Multilingual model vocabulary size unexpected: {multi_size} (expected 51865)")

    # Verify multilingual flags
    if not is_model_multilingual(multi_model_dir):
        raise ValueError(f"Model at '{multi_model_dir}' failed multilingual verification!")
    if is_model_multilingual(en_model_dir):
        raise ValueError(f"English model at '{en_model_dir}' was incorrectly flagged as multilingual!")

    return True


def verify_all_models(checksum_path: str = "models/CHECKSUMS.sha256") -> bool:
    """Verifies both file checksums and model distinctness."""
    verify_models(checksum_path)
    if os.path.exists("models/asr/whisper_tiny_en") and os.path.exists("models/asr/whisper_tiny_multilingual"):
        verify_models_distinct()
    return True


if __name__ == "__main__":
    try:
        verify_all_models()
        print("All model files verified successfully: authentic, distinct, and offline.")
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
