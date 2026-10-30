"""
Tests for Problem 10: Model Integrity, Distinction, and Multilingual Verification.
"""

import os
import pytest
from src.edge.model_check import (
    verify_models,
    verify_models_distinct,
    is_model_multilingual,
    verify_all_models,
    compute_sha256
)


def test_models_are_distinct():
    """whisper_tiny_en and whisper_tiny_multilingual must have different hashes and vocab sizes."""
    en_dir = "models/asr/whisper_tiny_en"
    multi_dir = "models/asr/whisper_tiny_multilingual"

    assert os.path.exists(en_dir), "English model directory must exist"
    assert os.path.exists(multi_dir), "Multilingual model directory must exist"

    # Must pass distinctness validation
    assert verify_models_distinct(en_dir, multi_dir) is True

    # Verify SHA256 of binaries are different
    en_sha = compute_sha256(os.path.join(en_dir, "model.bin"))
    multi_sha = compute_sha256(os.path.join(multi_dir, "model.bin"))
    assert en_sha != multi_sha, "English and Multilingual models must have distinct SHA256 hashes"


def test_multilingual_flag():
    """whisper_tiny_multilingual must report is_multilingual=True and tiny_en must report False."""
    assert is_model_multilingual("models/asr/whisper_tiny_multilingual") is True
    assert is_model_multilingual("models/asr/whisper_tiny_en") is False


def test_checksum_detects_corruption(tmp_path):
    """Corrupted model files or mismatching hashes must raise ValueError loudly."""
    fake_model = tmp_path / "model.bin"
    fake_model.write_bytes(b"corrupted_binary_data")

    fake_checksums = tmp_path / "CHECKSUMS.sha256"
    # Provide wrong SHA256
    fake_checksums.write_text(f"0000000000000000000000000000000000000000000000000000000000000000  {fake_model}\n")

    with pytest.raises(ValueError, match="corrupt or modified"):
        verify_models(str(fake_checksums))


def test_checksum_detects_missing(tmp_path):
    """Missing model files must raise FileNotFoundError loudly."""
    fake_checksums = tmp_path / "CHECKSUMS.sha256"
    fake_checksums.write_text("1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a  non_existent_model.bin\n")

    with pytest.raises(FileNotFoundError, match="missing"):
        verify_models(str(fake_checksums))


def test_duplicate_model_detection(tmp_path):
    """If English model is copied over as multilingual model, verification must fail."""
    en_dir = tmp_path / "en"
    en_dir.mkdir()
    multi_dir = tmp_path / "multi"
    multi_dir.mkdir()

    # Create identical binaries
    (en_dir / "model.bin").write_bytes(b"same_binary_data")
    (multi_dir / "model.bin").write_bytes(b"same_binary_data")

    with pytest.raises(ValueError, match="duplicate detected"):
        verify_models_distinct(str(en_dir), str(multi_dir))


def test_verify_all_models_passes():
    """All production model files in workspace must pass verification."""
    assert verify_all_models() is True
