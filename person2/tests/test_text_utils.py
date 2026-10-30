"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Unit Tests for Unicode Normalization and Language-Aware Scorer (Problem 1)
Author: Edge Audio, ASR & Multilingual Lead
"""

import unicodedata
import unittest
from src.edge.text_utils import normalize_text, score_text, CER_LANGUAGES


class TestTextUtils(unittest.TestCase):
    """Rigorous tests proving Unicode preservation across diverse scripts."""

    def test_01_english_normalization(self):
        """English text must be lowercased, punctuation stripped, and whitespace collapsed."""
        raw_en = "  Check the BATTERY status, please!  "
        normalized = normalize_text(raw_en, lang="en")
        self.assertEqual(normalized, "check the battery status please")
        self.assertNotIn("!", normalized)
        self.assertNotIn(",", normalized)

    def test_02_hindi_devanagari_preserved(self):
        """Hindi matras, anusvara, and virama must remain intact."""
        raw_hindi = "नमस्ते! आप कैसे हैं?"
        normalized = normalize_text(raw_hindi, lang="hi")

        orig_lm_count = sum(1 for c in unicodedata.normalize("NFC", raw_hindi) if unicodedata.category(c)[0] in ("L", "M"))
        norm_lm_count = sum(1 for c in normalized if unicodedata.category(c)[0] in ("L", "M"))

        self.assertEqual(orig_lm_count, norm_lm_count)
        self.assertEqual(normalized, "नमस्ते आप कैसे हैं")

    def test_03_telugu_and_malayalam_preserved(self):
        """Telugu and Malayalam complex vowel signs and viramas preserved."""
        raw_telugu = "నమస్కారం, మీరు ఎలా ఉన్నారు?"
        norm_te = normalize_text(raw_telugu, lang="te")
        self.assertNotIn(",", norm_te)
        self.assertNotIn("?", norm_te)
        self.assertIn("నమస్కారం", norm_te)

        raw_malayalam = "നമസ്കാരം! സുഖമാണോ?"
        norm_ml = normalize_text(raw_malayalam, lang="ml")
        self.assertNotIn("!", norm_ml)
        self.assertNotIn("?", norm_ml)
        self.assertIn("നമസ്കാരം", norm_ml)

    def test_04_arabic_chinese_russian(self):
        """Arabic, Chinese, and Russian normalization."""
        # Arabic
        raw_ar = "مَرْحَبًا بِكُمْ!"
        norm_ar = normalize_text(raw_ar, lang="ar")
        self.assertNotIn("!", norm_ar)

        # Chinese (no casing)
        raw_zh = "你好，世界！"
        norm_zh = normalize_text(raw_zh, lang="zh")
        self.assertEqual(norm_zh, "你好世界")

        # Russian (Cyrillic has casing, should lowercase)
        raw_ru = "Привет, Мир!"
        norm_ru = normalize_text(raw_ru, lang="ru")
        self.assertEqual(norm_ru, "привет мир")

    def test_05_zwnj_preserved(self):
        """Zero Width Non-Joiner (ZWNJ U+200C) must be preserved for Indic ligatures."""
        # Word with explicit ZWNJ
        text_with_zwnj = "క్\u200cలౌడ్"
        norm = normalize_text(text_with_zwnj)
        self.assertIn("\u200c", norm)

    def test_06_score_text_cer_vs_wer(self):
        """Validates that score_text selects CER for Indic/agglutinative and WER for English."""
        # English -> WER
        score_en, metric_en = score_text("turn on the light", "turn on the light", lang="en")
        self.assertEqual(metric_en, "WER")
        self.assertEqual(score_en, 0.0)

        # Chinese -> CER
        score_zh, metric_zh = score_text("你好", "您好", lang="zh")
        self.assertEqual(metric_zh, "CER")
        self.assertGreater(score_zh, 0.0)


if __name__ == "__main__":
    unittest.main()
