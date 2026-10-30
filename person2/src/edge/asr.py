"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Module A: Dual-Engine Multilingual Offline Automatic Speech Recognition (ASR)
Author: Edge Audio, ASR & Multilingual Lead

Features & Algorithms:
1. Faster-Whisper int8 (CTranslate2 CPU) with Multilingual & English models.
2. Problem 3: Robust Language Detection Architecture:
   - Modes: 'tier1_only' (DEFAULT: strictly {ta, en}), 'auto_all', 'locked:<xx>', 'allowed_list'.
   - Candidate pool pruning: restricts probabilities to active candidate set.
   - Dual-Decode Ambiguity Rule: if top1-top2 < margin, decodes both candidates and selects higher token log-prob.
   - Sticky Session Language: short utterances (<1000ms) prefer previous turn language.
   - Prior bias: tunable prior boost for 'ta' and 'en' in auto_all mode to suppress noise flips.
3. Language-aware Hallucination, Repetition & Background Artifact Cleaner.
4. Full adherence to contracts.py (Transcript dataclass & ASRProtocol).
"""

from __future__ import annotations
import json
import logging
import os
import re
import time
from typing import Iterator, Optional, Any, Tuple, List, Dict
import numpy as np

# Offline environment enforcement
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.edge.contracts import Transcript, ASRProtocol
from src.edge.config import CONFIG
from src.edge.text_utils import normalize_text

logger = logging.getLogger("ASR")

_HALLUCINATION_CACHE: Dict[str, set[str]] = {}


def get_hallucination_phrases(lang: str = "en") -> set[str]:
    """Loads and caches multilingual phantom hallucination phrases from JSON files."""
    if lang in _HALLUCINATION_CACHE:
        return _HALLUCINATION_CACHE[lang]

    phrases = set()
    dir_path = os.path.join(os.path.dirname(__file__), "hallucinations")
    file_path = os.path.join(dir_path, f"{lang}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        phrases.add(str(item).strip().lower())
        except Exception as e:
            logger.warning(f"Failed to load hallucinations for {lang}: {e}")

    # Also load common patterns
    if lang != "common":
        common_path = os.path.join(dir_path, "common.json")
        if os.path.exists(common_path):
            try:
                with open(common_path, "r", encoding="utf-8") as f:
                    common_data = json.load(f)
                    if isinstance(common_data, list):
                        for item in common_data:
                            phrases.add(str(item).strip().lower())
            except Exception:
                pass

    _HALLUCINATION_CACHE[lang] = phrases
    return phrases


def has_expected_script(text: str, lang: str) -> bool:
    """Checks if text contains code points for the expected script of lang."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    if lang == "en":
        return all("a" <= c.lower() <= "z" for c in letters)
    return True



def is_repeated_text(text: str) -> bool:
    """Checks for repeated 1-grams, 2-grams, or character repetitions."""
    words = [w.lower().strip(".,?!:;\"'") for w in text.split()]
    if len(words) >= 4:
        # 1-gram 4x consecutive
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return True
        most_common = max(set(words), key=words.count)
        if words.count(most_common) / len(words) >= 0.60:
            return True

        # 2-gram 3x consecutive (e.g. "battery status battery status battery status")
        if len(words) >= 6:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            for i in range(len(bigrams) - 2):
                if bigrams[i] == bigrams[i+1] == bigrams[i+2]:
                    return True
            # Even-stepped bigrams
            stepped = [(words[i], words[i+1]) for i in range(0, len(words)-1, 2)]
            if len(stepped) >= 3:
                most_common_bi = max(set(stepped), key=stepped.count)
                if stepped.count(most_common_bi) / len(stepped) >= 0.60:
                    return True

    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 4:
        for i in range(len(compact) - 3):
            if compact[i] == compact[i+1] == compact[i+2] == compact[i+3]:
                return True

    return False



def clean_text_hallucinations(
    text: str,
    lang: str = "en",
    avg_logprob: float = -0.5,
    no_speech_prob: float = 0.0,
    compression_ratio: float = 1.0,
    audio_duration_s: float = 1.0,
    locked_lang: Optional[str] = None
) -> str:
    """
    Problem 9: Multilingual Hallucination, Repetition, and Phantom Filter.
    Filters out phantom tokens while preserving genuine speech and non-English scripts.
    """
    raw = text.strip()
    norm = raw.lower().strip()

    if not norm:
        return ""

    # 1. Drop based on Whisper acoustic confidence signals
    if no_speech_prob > CONFIG.asr.no_speech_threshold:
        return ""
    if compression_ratio > CONFIG.asr.compression_ratio_threshold:
        return ""
    if avg_logprob < (CONFIG.asr.logprob_threshold - 0.5):
        return ""

    # 2. Strip pure punctuation or empty symbols
    if re.match(r"^(\.|\?|\!|\s|♪|♫|,|:|;|-)+$", norm):
        return ""

    # 3. Drop subtitle/credit/URL phantoms
    if any(marker in norm for marker in ["subtitles by", "amara.org", "http://", "https://", "www.", "подпишитесь"]):
        return ""

    # 4. Check repeated n-grams or character repetition
    if is_repeated_text(raw):
        return ""

    # 5. Check script mismatch if language is locked
    if locked_lang and not has_expected_script(raw, locked_lang):
        return ""

    # 6. Check against language phantom patterns
    hallucinations = get_hallucination_phrases(lang)
    if norm in hallucinations or norm.strip(".,!?") in hallucinations:
        # Genuine speech exception:
        # Strong speech signal on reasonable duration audio
        GENUINE_PHRASES = {"thank you", "thank you.", "thanks", "நன்றி", "நன்றி.", "धन्यवाद", "धन्यवाद।", "ಧನ್ಯವಾದಗಳು", "നന്ദി"}
        if (
            norm in GENUINE_PHRASES
            and avg_logprob >= -0.35
            and no_speech_prob <= 0.20
            and audio_duration_s >= 0.70
        ):
            return raw
        return ""

    return raw



class FasterWhisperEngine:
    """
    Offline CTranslate2 int8 Faster-Whisper ASR engine with Multilingual capabilities
    and robust language detection state machine.
    """
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        compute_type: Optional[str] = None,
        cpu_threads: Optional[int] = None
    ):
        self.model_dir = model_dir or CONFIG.asr.whisper_model_dir
        self.device = device
        self.compute_type = compute_type or CONFIG.asr.compute_type
        self.cpu_threads = cpu_threads or CONFIG.asr.cpu_threads
        self.model = None
        self.last_detected_lang = "en"
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_dir) or not os.path.exists(os.path.join(self.model_dir, "model.bin")):
            raise FileNotFoundError(
                f"\n[CRITICAL ERROR] Faster-Whisper model missing at '{self.model_dir}'.\n"
                f"Missing file: {os.path.join(self.model_dir, 'model.bin')}\n"
                "Silent runtime downloads are strictly forbidden by competition rules.\n"
                "Please run 'python scripts/download_models.py' to download model weights."
            )

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_dir,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local Faster-Whisper model: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def _resolve_language(
        self,
        audio_f32: np.ndarray,
        mode: str,
        runtime_lang: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, float], bool]:
        """
        Resolves language according to Problem 3 rules:
        Returns (chosen_lang, lang_prob, candidate_dict, is_ambiguous).
        """
        # 1. Explicit override or locked mode
        if runtime_lang:
            return (runtime_lang, 1.0, {runtime_lang: 1.0}, False)

        if mode.startswith("locked:"):
            locked_lang = mode.split(":", 1)[1].strip()
            return (locked_lang, 1.0, {locked_lang: 1.0}, False)

        # 2. Determine Candidate Pool (Strictly Tamil & English)
        if mode in ("bilingual", "tier1_only"):
            candidates = ["ta", "en"]
        elif mode == "allowed_list":
            candidates = list(CONFIG.asr.allowed_languages)
        elif mode == "auto_all":
            candidates = None
        else:
            candidates = ["ta", "en"]


        # 3. Quick initial pass to inspect language probabilities
        # We run transcribe with beam_size=1 on first 3 seconds to get info.all_language_probs
        head_audio = audio_f32[:min(len(audio_f32), 16000 * 3)]
        _, info = self.model.transcribe(
            head_audio,
            beam_size=1,
            language=None, # Auto detect raw
            temperature=0.0
        )

        raw_probs: List[Tuple[str, float]] = getattr(info, "all_language_probs", [])
        if not raw_probs:
            return (info.language or "en", getattr(info, "language_probability", 1.0), {}, False)

        # 4. Filter and re-normalize by candidate pool
        prob_dict = dict(raw_probs)
        if candidates is not None:
            filtered_probs = {c: prob_dict.get(c, 1e-6) for c in candidates}
            total = sum(filtered_probs.values()) or 1.0
            norm_probs = {c: v / total for c, v in filtered_probs.items()}
        else:
            # In auto_all mode, apply prior boost to 'ta' and 'en'
            norm_probs = dict(raw_probs)
            if CONFIG.asr.prior_boost_ta_en > 0.0:
                norm_probs["ta"] = norm_probs.get("ta", 0.0) + CONFIG.asr.prior_boost_ta_en
                norm_probs["en"] = norm_probs.get("en", 0.0) + CONFIG.asr.prior_boost_ta_en
                # Re-normalize
                total = sum(norm_probs.values()) or 1.0
                norm_probs = {k: v / total for k, v in norm_probs.items()}

        sorted_cands = sorted(norm_probs.items(), key=lambda x: x[1], reverse=True)
        top1_lang, top1_prob = sorted_cands[0]
        top2_lang, top2_prob = sorted_cands[1] if len(sorted_cands) > 1 else (top1_lang, 0.0)

        # 5. Sticky Session Language Rule on very short audio (<1000ms)
        audio_duration_ms = (len(audio_f32) / 16000.0) * 1000.0
        if CONFIG.asr.sticky_session_lang and audio_duration_ms < CONFIG.asr.lang_min_audio_ms:
            if self.last_detected_lang in norm_probs and abs(norm_probs[self.last_detected_lang] - top1_prob) < 0.20:
                return (self.last_detected_lang, norm_probs[self.last_detected_lang], norm_probs, False)

        # 6. Ambiguity Check (triggers Dual-Decode)
        margin = top1_prob - top2_prob
        is_ambiguous = (margin < CONFIG.asr.lang_confidence_margin) or (top1_prob < CONFIG.asr.lang_min_prob)

        return (top1_lang, top1_prob, dict(sorted_cands[:5]), is_ambiguous)

    def _decode_with_lang(self, audio_f32: np.ndarray, target_lang: str) -> Tuple[str, float, list, float, float]:
        """Decodes audio with forced target_lang and returns (text, avg_logprob, words, no_speech_prob, compression_ratio)."""
        segments, _ = self.model.transcribe(
            audio_f32,
            beam_size=CONFIG.asr.beam_size,
            best_of=1,
            language=target_lang,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False
        )

        texts = []
        logprobs = []
        words_list = []
        no_speech_probs = []
        comp_ratios = []
        for s in segments:
            clean_t = s.text.strip()
            if clean_t:
                texts.append(clean_t)
                if hasattr(s, "avg_logprob") and s.avg_logprob is not None:
                    logprobs.append(float(s.avg_logprob))
                if hasattr(s, "no_speech_prob") and s.no_speech_prob is not None:
                    no_speech_probs.append(float(s.no_speech_prob))
                if hasattr(s, "compression_ratio") and s.compression_ratio is not None:
                    comp_ratios.append(float(s.compression_ratio))
                if hasattr(s, "words") and s.words:
                    for w in s.words:
                        words_list.append({"word": w.word, "start": w.start, "end": w.end, "prob": w.probability})

        raw_text = " ".join(texts).strip()
        avg_logprob = float(np.mean(logprobs)) if logprobs else -1.0
        no_speech = float(np.max(no_speech_probs)) if no_speech_probs else 0.0
        comp_ratio = float(np.max(comp_ratios)) if comp_ratios else 1.0
        return (raw_text, avg_logprob, words_list, no_speech, comp_ratio)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_mode: Optional[str] = None,
        task: str = "transcribe"
    ) -> Transcript:
        t0 = time.perf_counter()

        if self.model is None:
            self._load_model()

        audio_f32 = np.asarray(audio, dtype=np.float32)
        if len(audio_f32) == 0:
            return Transcript(text="", is_final=True, lang="en", asr_ms=0.0, confidence=0.0)

        mode = language_mode or CONFIG.asr.language_mode

        # 1. Resolve Language & Check for Ambiguity
        chosen_lang, lang_prob, candidate_dict, is_ambiguous = self._resolve_language(
            audio_f32,
            mode=mode,
            runtime_lang=language
        )

        # 2. Dual-Decode if ambiguous
        if is_ambiguous and len(candidate_dict) >= 2 and not language and not mode.startswith("locked:"):
            sorted_cands = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)
            cand1, _ = sorted_cands[0]
            cand2, _ = sorted_cands[1]

            text1, logprob1, words1, ns1, cr1 = self._decode_with_lang(audio_f32, cand1)
            text2, logprob2, words2, ns2, cr2 = self._decode_with_lang(audio_f32, cand2)

            # Pick the decoding with higher average token log-probability
            if logprob1 >= logprob2:
                final_raw_text = text1
                final_words = words1
                chosen_lang = cand1
                chosen_logprob = logprob1
                chosen_ns = ns1
                chosen_cr = cr1
            else:
                final_raw_text = text2
                final_words = words2
                chosen_lang = cand2
                chosen_logprob = logprob2
                chosen_ns = ns2
                chosen_cr = cr2
        else:
            final_raw_text, chosen_logprob, final_words, chosen_ns, chosen_cr = self._decode_with_lang(audio_f32, chosen_lang)

        # 3. Update sticky session state
        self.last_detected_lang = chosen_lang

        # 4. Optional Task="translate" (Tier 3 fallback)
        is_translated = False
        translation_text = ""
        if task == "translate":
            # Translate to English
            t_segments, _ = self.model.transcribe(
                audio_f32,
                beam_size=CONFIG.asr.beam_size,
                task="translate",
                temperature=0.0
            )
            translation_text = " ".join(s.text.strip() for s in t_segments).strip()
            is_translated = True

        asr_ms = (time.perf_counter() - t0) * 1000.0
        audio_dur_s = len(audio_f32) / 16000.0
        locked_code = mode.split(":", 1)[1].strip() if mode.startswith("locked:") else None

        final_text = clean_text_hallucinations(
            final_raw_text,
            lang=chosen_lang,
            avg_logprob=chosen_logprob,
            no_speech_prob=chosen_ns,
            compression_ratio=chosen_cr,
            audio_duration_s=audio_dur_s,
            locked_lang=locked_code
        )
        confidence = float(np.exp(chosen_logprob)) if chosen_logprob < 0.0 else 1.0

        return Transcript(
            text=final_text,
            is_final=True,
            lang=chosen_lang,
            asr_ms=asr_ms,
            confidence=max(0.0, min(1.0, confidence)),
            words=final_words,
            lang_prob=float(lang_prob),
            lang_candidates=candidate_dict,
            translated=is_translated,
            translation_text=translation_text
        )


class SherpaOnnxStreamEngine:
    """Sherpa-ONNX online streaming transducer engine (OPTIONAL)."""
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or CONFIG.asr.sherpa_model_dir
        self.recognizer = None
        self._init_engine()

    def _init_engine(self):
        if not os.path.exists(self.model_dir):
            return
        tokens = os.path.join(self.model_dir, "tokens.txt")
        encoder = os.path.join(self.model_dir, "encoder.onnx")
        decoder = os.path.join(self.model_dir, "decoder.onnx")
        joiner = os.path.join(self.model_dir, "joiner.onnx")
        if not all(os.path.exists(p) for p in [tokens, encoder, decoder, joiner]):
            return
        try:
            import sherpa_onnx
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens, encoder=encoder, decoder=decoder, joiner=joiner,
                num_threads=CONFIG.asr.cpu_threads, sample_rate=CONFIG.audio.sample_rate, feature_dim=80
            )
        except Exception:
            self.recognizer = None

    def is_ready(self) -> bool:
        return self.recognizer is not None


class OfflineASR(ASRProtocol):
    """
    Production Unified Offline ASR Engine for HackNEX 2026.
    Supports Multilingual recognition with robust language detection modes.
    """
    def __init__(
        self,
        whisper_model_dir: Optional[str] = None,
        cpu_threads: Optional[int] = None
    ):
        self.whisper = FasterWhisperEngine(
            model_dir=whisper_model_dir,
            cpu_threads=cpu_threads
        )
        self.sherpa = SherpaOnnxStreamEngine()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_mode: Optional[str] = None,
        task: str = "transcribe"
    ) -> Transcript:
        """Transcribes complete audio array into Transcript with detected language."""
        return self.whisper.transcribe(
            audio,
            sample_rate=sample_rate,
            language=language,
            language_mode=language_mode,
            task=task
        )

    def transcribe_stream(self, chunks: Iterator[np.ndarray]) -> Iterator[Transcript]:
        """Transcribes streaming chunks progressively."""
        accumulated = []
        for i, chunk in enumerate(chunks):
            accumulated.append(chunk)
            if i > 0 and i % 20 == 0:
                yield Transcript(
                    text="",
                    is_final=False,
                    lang="en",
                    asr_ms=0.0,
                    confidence=0.5
                )

        full_audio = np.concatenate(accumulated) if accumulated else np.array([], dtype=np.float32)
        yield self.transcribe(full_audio)
