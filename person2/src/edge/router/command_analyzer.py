"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
English Speech Intent & Secure Command Router
Owned by Person 3 (Router & Brain Lead)

Features:
- Dynamically loads JSON intent tables from src/edge/router/intents/en.json (UTF-8).
- Canonical intent mapping for English system controls.
- Unknown intents return intent=None ("not understood / please repeat"); never guesses.
- Invokes actions strictly through secure actions.py dispatcher with confidence gates.
"""

from __future__ import annotations
import difflib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from src.edge.text_utils import normalize_text
from src.edge.router.actions import execute_whitelisted_action, ActionExecutionResult
from src.edge.config import CONFIG

ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_DIR = os.path.join(ROUTER_DIR, "intents")


@dataclass
class AnalyzedCommand:
    """Complete analyzed intent and executed system action result."""
    raw_text: str                         # Original transcribed speech
    detected_language: str                # 'en'
    intent: Optional[str]                 # Canonical intent ('BATTERY', 'NOTEPAD', etc.) or None
    action_executed: bool                 # True if action was executed
    response_en: str                      # Spoken response in English
    response_ta: str = ""                 # Backward compatibility alias
    latency_ms: float = 0.0               # Analysis runtime in milliseconds
    confidence: float = 1.0               # ASR confidence passed into analyzer
    is_translated: bool = False           # True if Tier 3 translate fallback was used

    def __post_init__(self):
        if not self.response_ta:
            self.response_ta = self.response_en

    @property
    def primary_response(self) -> str:
        """Returns English response."""
        return self.response_en

    @property
    def response(self) -> str:
        """Returns English response."""
        return self.response_en


class MultilingualCommandAnalyzer:
    """
    Data-driven speech intent classifier and system action coordinator.
    """
    def __init__(self, intents_dir: str = INTENTS_DIR):
        self.intents_dir = intents_dir
        self.tables: Dict[str, Dict[str, List[str]]] = {}
        self._load_intent_tables()

    def _load_intent_tables(self):
        """Loads all JSON files from the intents directory."""
        if not os.path.exists(self.intents_dir):
            return

        for fname in os.listdir(self.intents_dir):
            if fname.endswith(".json"):
                lang_code = fname[:-5]
                fpath = os.path.join(self.intents_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        self.tables[lang_code] = json.load(f)
                except Exception:
                    pass

    def _match_keywords(self, norm_text: str, intent_table: Dict[str, List[str]]) -> Optional[str]:
        """Matches normalized input against an intent table."""
        for intent, patterns in intent_table.items():
            for pat in patterns:
                norm_pat = normalize_text(pat)
                # Substring match or word match
                if norm_pat in norm_text or norm_text in norm_pat:
                    return intent
        return None

    def analyze_and_execute(
        self,
        user_text: str,
        detected_lang: Optional[str] = None,
        confidence: float = 1.0,
        is_translated: bool = False,
        translation_text: str = "",
        execute_actions: bool = True
    ) -> AnalyzedCommand:
        """
        Analyzes speech input, matches intent in English,
        and coordinates execution through actions.py.
        """
        t0 = time.perf_counter()
        clean_text = user_text.strip()
        if not clean_text:
            return AnalyzedCommand(
                raw_text="",
                detected_language="en",
                intent=None,
                action_executed=False,
                response_en="Please speak again.",
                response_ta="Please speak again.",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                confidence=0.0
            )

        norm_text = normalize_text(clean_text, lang="en")

        matched_intent: Optional[str] = None

        # Check English intent table
        if "en" in self.tables:
            matched_intent = self._match_keywords(norm_text, self.tables["en"])

        # Execute system action if matched
        if matched_intent:
            if execute_actions:
                act_res: ActionExecutionResult = execute_whitelisted_action(
                    intent=matched_intent,
                    confidence=confidence,
                    is_translated=is_translated
                )
                action_executed = act_res.executed
                resp_en = act_res.message_en
            else:
                action_executed = True
                resp_en = f"Command: {matched_intent}"
        else:
            # Unknown intent: never guess, report friendly repeat prompt
            action_executed = False
            resp_en = "I didn't catch that command. Please repeat."

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AnalyzedCommand(
            raw_text=clean_text,
            detected_language="en",
            intent=matched_intent,
            action_executed=action_executed,
            response_en=resp_en,
            response_ta=resp_en,
            latency_ms=latency_ms,
            confidence=confidence,
            is_translated=is_translated
        )


# Singleton instance
COMMAND_ANALYZER = MultilingualCommandAnalyzer()


def analyze_command(
    user_text: str,
    detected_lang: Optional[str] = None,
    confidence: float = 1.0,
    is_translated: bool = False,
    translation_text: str = "",
    execute_actions: bool = True
) -> AnalyzedCommand:
    """Convenience function to analyze and execute spoken command."""
    return COMMAND_ANALYZER.analyze_and_execute(
        user_text=user_text,
        detected_lang=detected_lang,
        confidence=confidence,
        is_translated=is_translated,
        translation_text=translation_text,
        execute_actions=execute_actions
    )
