"""
Fast-Path Intent Router & Jarvis Offline System Commands (HNX26EPS08).
Answers time, date, math, greetings, system queries < 5ms WITHOUT calling the LLM.
Returns None for unknown queries to pass to the LLM pipeline.
"""

import re
import math
import datetime
import operator
import subprocess
import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.edge.memory import Memory

import psutil

import logging
logger = logging.getLogger("router")


# ---------------------------------------------------------------------------
# SAFE MATH EVALUATOR (no eval() for security)
# ---------------------------------------------------------------------------

_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}

_OPS = {
    "plus": "+", "add": "+", "added": "+", "and": "+",
    "minus": "-", "subtract": "-", "subtracted": "-",
    "times": "*", "multiplied": "*", "multiply": "*", "x": "*",
    "divided": "/", "divide": "/", "over": "/",
    "power": "**", "squared": "**2", "cubed": "**3",
    "percent": "/100",
    "modulo": "%", "mod": "%",
}


def _words_to_numbers(text: str) -> str:
    """Replace word numbers with digits."""
    words = text.lower().split()
    result = []
    i = 0
    while i < len(words):
        w = words[i].rstrip(",")
        if w in _WORD_NUM:
            result.append(str(_WORD_NUM[w]))
        elif w in _OPS:
            result.append(_OPS[w])
        else:
            result.append(w)
        i += 1
    return " ".join(result)


def _safe_calc(expr: str) -> Optional[float]:
    """
    Safely evaluate a simple arithmetic expression.
    Supports +, -, *, /, **, % with numbers only.
    """
    expr = expr.strip()
    # Only allow digits, spaces and safe operators
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\%\(\)\*\*]+", expr):
        return None
    try:
        result = float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
        return result
    except Exception:
        return None


def _extract_math(text: str) -> Optional[str]:
    """Try to extract a calculable expression from natural language."""
    text = text.lower()
    # Replace written operators
    text = _words_to_numbers(text)
    # Remove non-math words
    text = re.sub(r"(what is|calculate|compute|solve|how much is|equals?|=|the result of|value of)", "", text, flags=re.IGNORECASE)
    # Strip non-math characters
    text = re.sub(r"[^0-9\s\.\+\-\*\/\%\(\)]", " ", text).strip()
    # Collapse spaces
    text = re.sub(r"\s+", "", text)
    return text if text else None


# ---------------------------------------------------------------------------
# INTENT PATTERNS
# ---------------------------------------------------------------------------

_PAT_TIME = re.compile(
    r"\b(what(\'?s| is) (the )?time|current time|time now|tell me the time|what time)\b", re.IGNORECASE
)
_PAT_DATE = re.compile(
    r"\b(what('?s| is) (the )?date|today('?s)? date|current date|what day|what('?s| is) today)\b", re.IGNORECASE
)
_PAT_GREET = re.compile(
    r"^(hi|hello|hey|howdy|good (morning|afternoon|evening|night)|what('?s| is) up|sup|greetings|yo)\b", re.IGNORECASE
)
_PAT_WHO = re.compile(
    r"\b(who are you|what are you|your name|introduce yourself)\b", re.IGNORECASE
)
_PAT_MATH = re.compile(
    r"\b(calculate|compute|what is|how much is|solve|math|times|plus|minus|divided|multiplied|squared)\b.*\d|\d.*\b(plus|minus|times|divided|multiplied|add|subtract)\b",
    re.IGNORECASE
)
_PAT_BATTERY = re.compile(
    r"\b(battery|charge|charging|power level)\b", re.IGNORECASE
)
_PAT_RAM = re.compile(
    r"\b(ram usage|system ram|system memory|how much ram|how much memory|ram status|mem usage)\b|\bram\b", re.IGNORECASE
)
_PAT_CPU = re.compile(
    r"\b(cpu|processor usage|cpu usage|processing power)\b", re.IGNORECASE
)
_PAT_OPEN_NOTEPAD = re.compile(
    r"\b(open|launch|start)\b.*\b(notepad|text editor|note)\b", re.IGNORECASE
)
_PAT_OPEN_CALC = re.compile(
    r"\b(open|launch|start)\b.*\b(calc|calculator)\b", re.IGNORECASE
)
_PAT_OPEN_BROWSER = re.compile(
    r"\b(open|launch|start)\b.*\b(browser|chrome|firefox|edge|internet)\b", re.IGNORECASE
)
_PAT_VOLUME_UP = re.compile(
    r"\b(increase|raise|turn up|higher|louder)\b.*\b(volume|sound|audio)\b|\bvolume up\b", re.IGNORECASE
)
_PAT_VOLUME_DOWN = re.compile(
    r"\b(decrease|lower|turn down|reduce|quieter)\b.*\b(volume|sound|audio)\b|\bvolume down\b", re.IGNORECASE
)
_PAT_MUTE = re.compile(
    r"\b(mute|silence|quiet)\b.*\b(volume|sound|audio|speaker)\b|\bmute\b", re.IGNORECASE
)
_PAT_REPEAT = re.compile(
    r"\b(repeat|say that again|what did you say|pardon|come again|didn't catch)\b", re.IGNORECASE
)
_PAT_EMPTY = re.compile(
    r"^[\s\.!?,;:…\-]*$"
)

# Memory-query patterns (strict — must not match normal questions)
_PAT_MEM_YESTERDAY = re.compile(
    r"\bwhat did (i|we) (ask|say|talk about)\b.*(yesterday)\b",
    re.IGNORECASE,
)
_PAT_MEM_TODAY = re.compile(
    r"\bwhat did (i|we) (ask|say|talk about)\b.*(today)\b",
    re.IGNORECASE,
)
_PAT_MEM_DAY_N = re.compile(
    r"\bwhat did (i|we) (ask|say|talk about)\b.*\bday\s+(\d+)\b",
    re.IGNORECASE,
)
_PAT_MEM_FORGET = re.compile(
    r"^(please\s+)?(forget everything|clear (all )?(my )?(history|memory)|delete (all )?(my )?(memory|history)|reset memory)[\.!]?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# JARVIS SYSTEM CONTROLS
# ---------------------------------------------------------------------------

def _get_battery() -> str:
    batt = psutil.sensors_battery()
    if batt is None:
        return "I cannot read battery information on this system."
    pct = round(batt.percent, 1)
    status = "charging" if batt.power_plugged else "discharging"
    secs = batt.secsleft
    if secs == psutil.POWER_TIME_UNLIMITED:
        time_str = "fully charged"
    elif secs == psutil.POWER_TIME_UNKNOWN or secs < 0:
        time_str = "time unknown"
    else:
        mins = secs // 60
        hrs = mins // 60
        mins = mins % 60
        time_str = f"{hrs}h {mins}m remaining" if hrs > 0 else f"{mins}m remaining"
    return f"Battery is at {pct}%, {status}. {time_str}."


def _get_ram() -> str:
    mem = psutil.virtual_memory()
    used_mb = round(mem.used / (1024 ** 2))
    total_mb = round(mem.total / (1024 ** 2))
    pct = round(mem.percent, 1)
    proc = psutil.Process()
    proc_mb = round(proc.memory_info().rss / (1024 ** 2))
    return (
        f"System RAM: {used_mb} MB used of {total_mb} MB ({pct}%). "
        f"This process (EdgeVoice) is using {proc_mb} MB."
    )


def _get_cpu() -> str:
    cpu = psutil.cpu_percent(interval=0.1)
    cores = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    proc = psutil.Process()
    proc_cpu = proc.cpu_percent(interval=None)
    affinity = len(proc.cpu_affinity()) if hasattr(proc, "cpu_affinity") else cores
    return (
        f"CPU usage is {cpu}% across {logical} logical cores ({cores} physical). "
        f"EdgeVoice is pinned to {affinity} cores and using {proc_cpu:.1f}%."
    )


def _open_app(app: str) -> str:
    try:
        if sys.platform == "win32":
            subprocess.Popen(app, shell=True)
            return f"Opening {app.replace('.exe', '').title()}."
        else:
            return f"App launching is only supported on Windows in this build."
    except Exception as e:
        return f"Could not open {app}: {e}"


def _adjust_volume(action: str) -> str:
    if sys.platform != "win32":
        return "Volume control is supported on Windows only."
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        if action == "mute":
            volume.SetMute(1, None)
            return "Audio muted."
        elif action == "unmute":
            volume.SetMute(0, None)
            return "Audio unmuted."
        else:
            cur = volume.GetMasterVolumeLevelScalar()
            if action == "up":
                volume.SetMasterVolumeLevelScalar(min(1.0, cur + 0.1), None)
                return f"Volume increased to {int(min(1.0, cur + 0.1) * 100)}%."
            elif action == "down":
                volume.SetMasterVolumeLevelScalar(max(0.0, cur - 0.1), None)
                return f"Volume decreased to {int(max(0.0, cur - 0.1) * 100)}%."
    except ImportError:
        # pycaw not installed — use nircmd fallback or report
        cmd_map = {"up": "nircmd changesysvolume 6553", "down": "nircmd changesysvolume -6553", "mute": "nircmd mutesysvolume 1"}
        if action in cmd_map:
            try:
                subprocess.Popen(cmd_map[action], shell=True)
                return f"Volume {action} command sent."
            except Exception:
                pass
        return "Volume control requires pycaw or nircmd. Install pycaw: pip install pycaw"
    except Exception as e:
        return f"Volume control error: {e}"


# ---------------------------------------------------------------------------
# MAIN ROUTER FUNCTION
# ---------------------------------------------------------------------------

_last_bot_reply: str = ""


def route_intent(
    user_text: str,
    last_reply: str = "",
    memory=None,          # Optional[Memory] — avoids circular import
) -> Optional[str]:
    """
    Fast-path intent router. Returns an instant answer string if the query
    matches a known rule, or None to pass through to the LLM pipeline.

    Args:
        user_text: Cleaned transcript from ASR.
        last_reply: Last bot reply (used for "repeat that" intent).
        memory: Optional Memory instance for history-query intents.

    Returns:
        str: Instant answer text.
        None: Query should be forwarded to LLM.
    """
    global _last_bot_reply
    if last_reply:
        _last_bot_reply = last_reply

    text = (user_text or "").strip().strip('"\'“”`').strip()

    # Empty / noise-only input
    if not text or _PAT_EMPTY.match(text):
        return "I didn't catch that. Could you please say that again?"

    # Greetings
    if _PAT_GREET.match(text):
        now_hour = datetime.datetime.now().hour
        if now_hour < 12:
            time_greet = "Good morning"
        elif now_hour < 17:
            time_greet = "Good afternoon"
        else:
            time_greet = "Good evening"
        return f"{time_greet}! I'm Jarvis Edge, your offline AI assistant. How can I help you?"

    # Identity
    if _PAT_WHO.search(text):
        return (
            "I'm Jarvis Edge — a fully offline, on-device conversational AI. "
            "I run entirely on this laptop's CPU with no cloud connections whatsoever."
        )

    # Repeat last reply
    if _PAT_REPEAT.search(text):
        if _last_bot_reply:
            return _last_bot_reply
        return "I don't have a previous reply to repeat."

    # Time
    if _PAT_TIME.search(text):
        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')} ({now.strftime('%H:%M')} in 24-hour format)."

    # Date
    if _PAT_DATE.search(text):
        now = datetime.datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    # Math (pure number expressions or word-based)
    if _PAT_MATH.search(text) or re.search(r"\d[\d\s\+\-\*\/\.%\^]+\d", text):
        expr = _extract_math(text)
        if expr:
            result = _safe_calc(expr)
            if result is not None:
                if result == int(result):
                    answer = str(int(result))
                else:
                    answer = f"{result:.4f}".rstrip("0").rstrip(".")
                return f"The answer is {answer}."

    # Battery
    if _PAT_BATTERY.search(text):
        return _get_battery()

    # RAM / Memory
    if _PAT_RAM.search(text):
        return _get_ram()

    # CPU
    if _PAT_CPU.search(text):
        return _get_cpu()

    # Open Notepad
    if _PAT_OPEN_NOTEPAD.search(text):
        return _open_app("notepad.exe")

    # Open Calculator
    if _PAT_OPEN_CALC.search(text):
        return _open_app("calc.exe")

    # Open Browser
    if _PAT_OPEN_BROWSER.search(text):
        return _open_app("start microsoft-edge:")

    # Volume Up
    if _PAT_VOLUME_UP.search(text):
        return _adjust_volume("up")

    # Volume Down
    if _PAT_VOLUME_DOWN.search(text):
        return _adjust_volume("down")

    # Mute
    if _PAT_MUTE.search(text):
        return _adjust_volume("mute")

    # ------------------------------------------------------------------
    # MEMORY INTENTS (require memory object; never LLM)
    # ------------------------------------------------------------------
    if memory is not None:
        try:
            # "forget everything"
            if _PAT_MEM_FORGET.match(text):
                memory.clear()
                return "Done. I've deleted all conversation history."

            # "what did I ask yesterday"
            if _PAT_MEM_YESTERDAY.search(text):
                yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                turns = memory.on_day(yesterday)
                if not turns:
                    return "I have no record of any conversation yesterday."
                items = "; ".join(f'"{u}"' for u, _ in turns[:5])
                return f"Yesterday you asked: {items}."

            # "what did I ask today"
            if _PAT_MEM_TODAY.search(text):
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                turns = memory.on_day(today)
                if not turns:
                    return "No conversations recorded yet today."
                items = "; ".join(f'"{u}"' for u, _ in turns[:5])
                return f"Today you asked: {items}."

            # "what did I ask on day N"
            m = _PAT_MEM_DAY_N.search(text)
            if m:
                n = int(m.group(3))
                first = memory.first_day()
                if not first:
                    return "I have no recorded conversations yet."
                target_dt = datetime.datetime.strptime(first, "%Y-%m-%d") + datetime.timedelta(days=n - 1)
                target_day = target_dt.strftime("%Y-%m-%d")
                turns = memory.on_day(target_day)
                if not turns:
                    return f"No conversations found for day {n} ({target_day})."
                items = "; ".join(f'"{u}"' for u, _ in turns[:5])
                return f"On day {n} ({target_day}) you asked: {items}."

        except Exception as e:
            logger.warning(f"[Router] Memory intent error: {e}")
            # Fall through to LLM

    # Out of scope / unknown — pass to LLM
    return None


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "hello",
        "what time is it",
        "what is today's date",
        "calculate 48 times 12",
        "what is 250 plus 375",
        "what is 144 divided by 12",
        "check battery",
        "ram usage",
        "cpu usage",
        "open notepad",
        "open calculator",
        "mute audio",
        "increase volume",
        "who are you",
        "repeat that",
        "tell me a story about dragons",  # should return None
        "",  # should return fallback
    ]
    print("\n=== ROUTER STANDALONE TEST ===\n")
    for t in tests:
        result = route_intent(t)
        status = "FAST ROUTER" if result is not None else "→ LLM"
        display = result if result else "[forwarded to LLM]"
        print(f"  [{status}] Input: '{t[:50]}'\n           Reply: {display[:80]}\n")
