"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
English Speech Intent & Offline System Command Analyzer
Author: Person 2 & Person 1 (Edge Audio & Fast-Path Team)

Features:
- Analyzes speech input in English.
- Instant (<1ms) rule-based intent parsing & keyword normalization.
- Executes real Windows OS system actions (Battery, RAM, Notepad, Calc, Time, Date).
- Returns structured AnalyzedCommand with English spoken responses.
- 100% OFFLINE, CPU-ONLY, zero cloud APIs.
"""

from __future__ import annotations
import datetime
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Any
import psutil


@dataclass
class AnalyzedCommand:
    """Result of speech analysis and system command execution."""
    raw_text: str                         # Original transcribed speech
    detected_language: str                # 'en' (English)
    intent: str                           # 'BATTERY', 'RAM', 'NOTEPAD', 'CALCULATOR', 'TIME', 'DATE', 'GREETING', 'UNKNOWN'
    action_executed: bool                 # True if an OS command was triggered
    response_en: str                      # Spoken response in English
    response_ta: str = ""                 # Backward compatibility alias
    latency_ms: float = 0.0               # Analysis runtime in milliseconds

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
    High-performance speech intent classifier and system command engine.
    """
    def __init__(self):
        # Precompile patterns for sub-millisecond execution (<1ms)
        self._patterns = {
            "BATTERY": [
                r"battery status", r"check battery", r"battery level", r"power status", r"how much battery", r"\bbattery\b"
            ],
            "RAM": [
                r"ram status", r"memory usage", r"check ram", r"system memory", r"check memory", r"\bram\b"
            ],
            "NOTEPAD": [
                r"open notepad", r"launch notepad", r"start notepad", r"open text editor", r"\bnotepad\b"
            ],
            "CALCULATOR": [
                r"open calculator", r"open calc", r"launch calculator", r"start calculator", r"\bcalculator\b"
            ],
            "TIME": [
                r"what time is it", r"current time", r"tell me the time", r"what is the time", r"\btime\b"
            ],
            "DATE": [
                r"what is today's date", r"what is the date", r"today's date", r"what day is it", r"\bdate\b"
            ],
            "GREETING": [
                r"\bhello\b", r"\bhi\b", r"hey", r"who are you", r"good morning", r"good evening"
            ]
        }

    def _detect_lang(self, text: str) -> str:
        return "en"

    def analyze_and_execute(self, user_text: str, execute_actions: bool = True) -> AnalyzedCommand:
        """
        Analyzes spoken user text, matches intent,
        and optionally executes Windows system actions natively.
        """
        t0 = time.perf_counter()
        clean_text = user_text.strip()
        lang = "en"
        lower_text = clean_text.lower()

        matched_intent = "UNKNOWN"
        for intent, patterns in self._patterns.items():
            for pat in patterns:
                if re.search(pat, lower_text):
                    matched_intent = intent
                    break
            if matched_intent != "UNKNOWN":
                break

        action_executed = False
        resp_en = ""

        # -------------------------------------------------------------
        # SYSTEM ACTION EXECUTION & RESPONSE GENERATION
        # -------------------------------------------------------------
        if matched_intent == "BATTERY":
            battery = psutil.sensors_battery()
            if battery:
                pct = int(battery.percent)
                charging = battery.power_plugged
                charge_str_en = "charging" if charging else "discharging"
                resp_en = f"Battery is at {pct} percent and currently {charge_str_en}."
            else:
                resp_en = "Battery sensor unavailable (running on AC power)."
            action_executed = True

        elif matched_intent == "RAM":
            mem = psutil.virtual_memory()
            pct = int(mem.percent)
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            resp_en = f"RAM usage is currently at {pct} percent ({used_gb:.1f} GB of {total_gb:.1f} GB used)."
            action_executed = True

        elif matched_intent == "NOTEPAD":
            if execute_actions:
                try:
                    subprocess.Popen("notepad.exe")
                    action_executed = True
                except Exception:
                    pass
            resp_en = "Opening Notepad application."

        elif matched_intent == "CALCULATOR":
            if execute_actions:
                try:
                    subprocess.Popen("calc.exe")
                    action_executed = True
                except Exception:
                    pass
            resp_en = "Opening Calculator application."

        elif matched_intent == "TIME":
            now_str = datetime.datetime.now().strftime("%I:%M %p")
            resp_en = f"The current time is {now_str}."
            action_executed = True

        elif matched_intent == "DATE":
            now_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            resp_en = f"Today is {now_date}."
            action_executed = True

        elif matched_intent == "GREETING":
            resp_en = "Hello! I am your offline voice assistant. How can I help you today?"
            action_executed = True

        else:
            resp_en = "Processing your query..."
            action_executed = False

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AnalyzedCommand(
            raw_text=clean_text,
            detected_language="en",
            intent=matched_intent,
            action_executed=action_executed,
            response_en=resp_en,
            response_ta=resp_en,
            latency_ms=latency_ms
        )


# Singleton analyzer instance
COMMAND_ANALYZER = MultilingualCommandAnalyzer()
