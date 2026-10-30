"""
Tests for Problem 9: Multilingual Hallucination & Phantom Artifact Filter.
"""

import pytest
from src.edge.asr import clean_text_hallucinations, has_expected_script, is_repeated_text


def test_silence_hallucinations_dropped_en():
    """Silence phantom phrases in English must be stripped when acoustic confidence is low."""
    # Low confidence / silence phantom
    assert clean_text_hallucinations("Thank you.", lang="en", avg_logprob=-1.2, no_speech_prob=0.85) == ""
    assert clean_text_hallucinations("[BLANK_AUDIO]", lang="en") == ""
    assert clean_text_hallucinations("Thanks for watching!", lang="en", avg_logprob=-1.0) == ""
    assert clean_text_hallucinations("Subtitles by the Amara.org community", lang="en") == ""
    assert clean_text_hallucinations("...", lang="en") == ""
    assert clean_text_hallucinations("♪ ♪", lang="en") == ""


def test_common_phantom_artifacts_dropped():
    """Common phantom credits/artifacts must be stripped."""
    assert clean_text_hallucinations("[blank_audio]", lang="en") == ""
    assert clean_text_hallucinations("Visit https://example.com for more", lang="en") == ""


def test_real_speech_thank_you_kept():
    """Genuine spoken 'thank you' with high acoustic confidence must be kept."""
    # Strong acoustic speech signal
    assert clean_text_hallucinations(
        "Thank you.", lang="en", avg_logprob=-0.25, no_speech_prob=0.05, audio_duration_s=1.2
    ) == "Thank you."


def test_repeated_token_and_char_loops_dropped():
    """Repeated n-grams or runaway tokens must be detected and dropped."""
    assert clean_text_hallucinations("battery status battery status battery status battery status", lang="en") == ""
    assert clean_text_hallucinations("hello hello hello hello", lang="en") == ""
    assert is_repeated_text("battery battery battery battery") is True


def test_script_mismatch_in_locked_mode():
    """If language is locked to English, an output in completely unrelated foreign script is rejected."""
    # Locked to English, but hallucinated Russian Cyrillic
    assert clean_text_hallucinations("Спасибо", lang="en", locked_lang="en") == ""
    # Genuine English text is preserved
    assert clean_text_hallucinations("what time is it", lang="en", locked_lang="en") == "what time is it"


def test_whisper_acoustic_signal_veto():
    """High no_speech_prob or high compression_ratio must veto output."""
    assert clean_text_hallucinations("open notepad", lang="en", no_speech_prob=0.85) == ""
    assert clean_text_hallucinations("open calculator", lang="en", compression_ratio=2.8) == ""
    assert clean_text_hallucinations("valid text", lang="en", avg_logprob=-2.0) == ""
