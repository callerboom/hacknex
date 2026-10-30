"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Secure System Action Dispatcher (Problem 6 & Problem 8)
Owned by Person 3 (Router & Brain Lead)

Security Rules:
- Strict WHITELIST dict: intent -> fixed argv list.
- subprocess.Popen(argv_list, shell=False) ALWAYS.
- Never concatenate or execute unsanitized transcript text.
- No shell execution and no shell interpreters allowed.
- Confidence gate: confidence < min_command_confidence rejects execution.
- Cross-platform safe: gracefully handles non-Windows and null battery sensors.
"""

from __future__ import annotations
import datetime
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Dict, List
import psutil

from src.edge.config import CONFIG

# Strict immutable whitelist mapping intent to exact binary argv
WINDOWS_WHITELIST: Dict[str, List[str]] = {
    "NOTEPAD": ["notepad.exe"],
    "CALCULATOR": ["calc.exe"]
}

LINUX_WHITELIST: Dict[str, List[str]] = {
    "NOTEPAD": ["gedit"],
    "CALCULATOR": ["gnome-calculator"]
}


@dataclass
class ActionExecutionResult:
    """Outcome of an attempted system action."""
    executed: bool
    message_en: str
    message_ta: str = ""
    details: Optional[str] = None

    def __post_init__(self):
        if not self.message_ta:
            self.message_ta = self.message_en

    @property
    def message(self) -> str:
        return self.message_en

    @property
    def success(self) -> bool:
        """Alias for executed."""
        return self.executed



def execute_whitelisted_action(
    intent: str,
    confidence: float = 1.0,
    is_translated: bool = False
) -> ActionExecutionResult:
    """
    Executes an action strictly via immutable whitelist and fixed argv.
    Fails safely if confidence is below threshold or OS is unsupported.
    """
    # 1. Confidence Gate
    min_conf = CONFIG.asr.min_command_confidence
    if confidence < min_conf:
        return ActionExecutionResult(
            executed=False,
            message_en="Command confidence too low. Please repeat your instruction."
        )

    # 2. Information Query Actions (Read-only OS sensors)
    if intent == "BATTERY":
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                pct = int(battery.percent)
                charging = battery.power_plugged
                ch_en = "charging" if charging else "discharging"
                return ActionExecutionResult(
                    executed=True,
                    message_en=f"Battery is at {pct} percent and currently {ch_en}."
                )
            else:
                return ActionExecutionResult(
                    executed=True,
                    message_en="Battery information unavailable (running on AC power)."
                )
        except Exception as e:
            return ActionExecutionResult(
                executed=False,
                message_en=f"Failed to read battery status: {e}"
            )

    elif intent == "RAM":
        try:
            mem = psutil.virtual_memory()
            pct = int(mem.percent)
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            return ActionExecutionResult(
                executed=True,
                message_en=f"RAM usage is {pct} percent ({used_gb:.1f} GB of {total_gb:.1f} GB used)."
            )
        except Exception as e:
            return ActionExecutionResult(
                executed=False,
                message_en=f"Failed to read memory status: {e}"
            )

    elif intent == "TIME":
        now_str = datetime.datetime.now().strftime("%I:%M %p")
        return ActionExecutionResult(
            executed=True,
            message_en=f"The current time is {now_str}."
        )

    elif intent == "DATE":
        now_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return ActionExecutionResult(
            executed=True,
            message_en=f"Today is {now_date}."
        )

    elif intent == "GREETING":
        return ActionExecutionResult(
            executed=True,
            message_en="Hello! I am your offline voice assistant. How can I help you today?"
        )

    # 3. Application Launch Actions (Strict Whitelist argv, shell=False)
    whitelist = WINDOWS_WHITELIST if sys.platform == "win32" else LINUX_WHITELIST
    if intent in whitelist:
        argv = whitelist[intent]
        try:
            # shell=False is strictly enforced; argv is a hardcoded list
            subprocess.Popen(argv, shell=False)
            app_name = argv[0].replace(".exe", "").title()
            return ActionExecutionResult(
                executed=True,
                message_en=f"Opening {app_name} application."
            )
        except Exception as e:
            return ActionExecutionResult(
                executed=False,
                message_en=f"Failed to launch application: {e}"
            )

    return ActionExecutionResult(
        executed=False,
        message_en="Intent not mapped to a system action."
    )


# Alias for convenience
execute_action = execute_whitelisted_action

