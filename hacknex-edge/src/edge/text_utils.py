"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Unicode Normalization and Language-Aware Metric Scorer
Author: Edge Audio, ASR & Multilingual Lead

Features:
- NFC normalization preserving all Indic combining characters (Mn, Mc, Me), viramas, and pullis.
- Preserves ZWJ (U+200D) and ZWNJ (U+200C) essential for Indic ligatures.
- Selective punctuation and symbol stripping without corrupting non-Latin scripts.
- Lowercasing applied strictly to cased scripts (Latin, Cyrillic, Greek).
- Configurable metric chooser: CER for Indic/agglutinative/non-spaced scripts, WER for space-delimited scripts.
- Bidi mark cleanup for RTL languages (Arabic, Hebrew, Persian, Urdu).
"""

from __future__ import annotations
import unicodedata
import re
from typing import Set, Tuple, Optional
import jiwer

# Languages where Character Error Rate (CER) is the preferred / more meaningful metric
# than Word Error Rate (due to agglutination, complex morphology, or lack of explicit word spacing)
CER_LANGUAGES: Set[str] = {
    "ta",  # Tamil
    "te",  # Telugu
    "ml",  # Malayalam
    "kn",  # Kannada
    "zh",  # Chinese (Mandarin/Cantonese)
    "ja",  # Japanese
    "th",  # Thai
    "lo",  # Lao
    "km",  # Khmer
    "my",  # Burmese
}

# Cased scripts that have distinct upper/lower case
CASED_SCRIPTS: Set[str] = {"Latin", "Cyrillic", "Greek"}

# RTL languages
RTL_LANGUAGES: Set[str] = {"ar", "he", "fa", "ur"}

# BiDi directional control characters
BIDI_CHARS = {
    '\u200e',  # LTR Mark
    '\u200f',  # RTL Mark
    '\u202a',  # LTR Embedding
    '\u202b',  # RTL Embedding
    '\u202c',  # Pop Directional Formatting
    '\u202d',  # LTR Override
    '\u202e',  # RTL Override
    '\u2066',  # LTR Isolate
    '\u2067',  # RTL Isolate
    '\u2068',  # First Strong Isolate
    '\u2069',  # Pop Directional Isolate
}

# Indic joiner characters to preserve
INDIC_JOINERS = {
    '\u200c',  # ZERO WIDTH NON-JOINER (ZWNJ)
    '\u200d',  # ZERO WIDTH JOINER (ZWJ)
}


def normalize_text(text: str, lang: Optional[str] = None) -> str:
    """
    Normalizes text across all scripts safely:
    - Normalizes to Unicode NFC.
    - Preserves Letters (L*), Combining Marks/Vowel Signs (M*), Numbers (N*), Whitespace.
    - Preserves ZWJ/ZWNJ for Indic scripts.
    - Strips Punctuation (P*) and Symbols (S*).
    - Lowercases cased scripts only (Latin, Cyrillic, Greek); preserves Indic, Arabic, CJK.
    - Collapses whitespace.
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization
    nfc_text = unicodedata.normalize("NFC", text)

    # 2. Character-by-character filtering
    out_chars = []
    for ch in nfc_text:
        cat = unicodedata.category(ch)

        # Retain Indic joiners (ZWJ/ZWNJ)
        if ch in INDIC_JOINERS:
            out_chars.append(ch)
            continue

        # Strip BiDi control characters if scoring
        if ch in BIDI_CHARS:
            continue

        # Keep Letters (L), Marks (M), Numbers (N), and Whitespace (Z or standard space/tab/newline)
        if cat.startswith("L") or cat.startswith("M") or cat.startswith("N") or cat.startswith("Z") or ch in " \t\n\r":
            out_chars.append(ch)
        # Drop Punctuation (P*), Symbols (S*), Other Controls (C*)
        else:
            continue

    filtered_text = "".join(out_chars)

    # 3. Lowercase cased scripts only (e.g. Latin, Cyrillic, Greek)
    # Check if string contains cased characters
    has_cased = any(unicodedata.category(c) in ("Lu", "Lt") for c in filtered_text)
    if has_cased:
        filtered_text = filtered_text.lower()

    # 4. Collapse whitespace
    collapsed = re.sub(r"\s+", " ", filtered_text).strip()
    return collapsed


def score_text(ref: str, hyp: str, lang: str = "en") -> Tuple[float, str]:
    """
    Computes transcription error rate using the appropriate metric for the given language:
    - CER for CER_LANGUAGES (Tamil, Telugu, Malayalam, Kannada, Chinese, Japanese, Thai, etc.)
    - WER for space-delimited languages (English, French, Spanish, Russian, Hindi, etc.)
    
    Returns (error_rate, metric_name) where metric_name is 'CER' or 'WER'.
    """
    clean_ref = normalize_text(ref, lang)
    clean_hyp = normalize_text(hyp, lang)

    # If reference is empty
    if not clean_ref:
        return (0.0 if not clean_hyp else 1.0, "WER")

    # Strip BiDi control marks for RTL scoring
    if lang in RTL_LANGUAGES:
        clean_ref = "".join(c for c in clean_ref if c not in BIDI_CHARS)
        clean_hyp = "".join(c for c in clean_hyp if c not in BIDI_CHARS)

    if lang in CER_LANGUAGES:
        # Character Error Rate via jiwer
        cer_val = jiwer.cer(clean_ref, clean_hyp)
        return (float(cer_val), "CER")
    else:
        # Word Error Rate via jiwer
        wer_val = jiwer.wer(clean_ref, clean_hyp)
        return (float(wer_val), "WER")
