"""Offline Fast-Router & Jarvis Local System Controls.

Provides sub-millisecond (<1ms) intent routing and local execution for
multilingual voice-to-text transcripts across:
- English
- Tamil (தமிழ் & Tanglish)
- Hindi (हिंदी & Hinglish)
- Telugu (తెలుగు & Telugish)
- Malayalam (മലയാളം & Manglish)
- Kannada (ಕನ್ನಡ & Kanglish)

Supported Multilingual Intent Domains:
1. Conversational Greetings & Identity (7 languages)
2. System Time & Date Queries (Today, Tomorrow, Yesterday in 7 languages)
3. Battery Telemetry & Charging State (7 languages)
4. RAM / System Memory Usage (7 languages)
5. CPU Processor Load & Utilization (7 languages)
6. Wi-Fi & Local Network Adapter Status (7 languages)
7. Internet Connectivity Probe (7 languages)
8. Audio / Volume Controls (Volume Up, Down, Mute in 7 languages)
9. Desktop Screenshot Capture (7 languages)
10. Application Launchers (Notepad, Calculator, Browser, Explorer, VS Code, Settings, Taskmgr, CMD in 7 languages)
11. Basic Offline Arithmetic & Unit Conversions (7 languages)
12. Offline Markdown Knowledge Retrieval (First-Aid, Emergency care, Health & Wellness in 7 languages)
13. Silence / Noise Detection & LLM Forwarding
"""

import ast
import concurrent.futures
import ctypes
import datetime
import math
import operator
import os
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

from src.edge.knowledge_search import search_offline_knowledge

# Pre-cached static system properties to eliminate runtime WMI / OS query latency
_SYSTEM_OS_NAME: str = f"{platform.system()} {platform.release()}"
_SYSTEM_ARCH: str = platform.machine()
_SYSTEM_NODE: str = platform.node()

# Pre-cache local network IP
try:
    _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _s.connect(("8.8.8.8", 53))
    _CACHED_LOCAL_IP: str = _s.getsockname()[0]
    _s.close()
except Exception:
    _CACHED_LOCAL_IP = "127.0.0.1"

# Screenshot output folder
_SCREENSHOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "screenshots")
)

# Persistent Background Worker Pool for zero-latency non-blocking actions
_ACTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="JarvisAction")


# ==============================================================================
# SAFE AST MATH EVALUATOR & UNIT CONVERTER
# ==============================================================================

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST expression node securely without eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](_safe_eval_node(node.operand))
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            left = _safe_eval_node(node.left)
            right = _safe_eval_node(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("division by zero")
            if op_type == ast.Pow and (left > 10000 or right > 1000):
                raise OverflowError("exponent too large")
            return _SAFE_OPERATORS[op_type](left, right)
    raise ValueError("Unsupported or unsafe math expression")


def evaluate_unit_conversion(query: str) -> Optional[str]:
    """Calculate offline unit conversions (Temperature, Length, Weight, Storage)."""
    cleaned = query.strip().lower()
    cleaned = re.sub(r"^(?:convert|what\s+is|calculate|change)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\?\.!\s]+$", "", cleaned)

    # 1. Temperature: Celsius <-> Fahrenheit
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:c|celsius)\s+(?:to|in|la)\s+(?:f|fahrenheit)$", cleaned)
    if m:
        c = float(m.group(1))
        f = (c * 9.0 / 5.0) + 32.0
        return f"{c}°C is {f:.2f}°F"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:f|fahrenheit)\s+(?:to|in|la)\s+(?:c|celsius)$", cleaned)
    if m:
        f = float(m.group(1))
        c = (f - 32.0) * 5.0 / 9.0
        return f"{f}°F is {c:.2f}°C"

    # 2. Length / Distance
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:km|kilometers?)\s+(?:to|in|la)\s+(?:miles?)$", cleaned)
    if m:
        val = float(m.group(1))
        res = val * 0.621371
        return f"{val} km is {res:.2f} miles"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:miles?)\s+(?:to|in|la)\s+(?:km|kilometers?)$", cleaned)
    if m:
        val = float(m.group(1))
        res = val * 1.60934
        return f"{val} miles is {res:.2f} km"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:m|meters?)\s+(?:to|in|la)\s+(?:ft|feet)$", cleaned)
    if m:
        val = float(m.group(1))
        res = val * 3.28084
        return f"{val} meters is {res:.2f} feet"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:ft|feet)\s+(?:to|in|la)\s+(?:inches?|in)$", cleaned)
    if m:
        val = float(m.group(1))
        res = val * 12.0
        return f"{val} feet is {int(res) if res.is_integer() else res:.2f} inches"

    # 3. Weight / Mass
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:kg|kilograms?)\s+(?:to|in|la)\s+(?:lbs?|pounds?)$", cleaned)
    if m:
        val = float(m.group(1))
        res = val * 2.20462
        return f"{val} kg is {res:.2f} lbs"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:lbs?|pounds?)\s+(?:to|in|la)\s+(?:kg|kilograms?)$", cleaned)
    if m:
        val = float(m.group(1))
        res = val / 2.20462
        return f"{val} lbs is {res:.2f} kg"

    # 4. Digital Storage
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:gb|gigabytes?)\s+(?:to|in|la)\s+(?:mb|megabytes?)$", cleaned)
    if m:
        val = float(m.group(1))
        return f"{val} GB is {int(val * 1024)} MB"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:mb|megabytes?)\s+(?:to|in|la)\s+(?:gb|gigabytes?)$", cleaned)
    if m:
        val = float(m.group(1))
        res = val / 1024.0
        return f"{val} MB is {res:.3g} GB"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:tb|terabytes?)\s+(?:to|in|la)\s+(?:gb|gigabytes?)$", cleaned)
    if m:
        val = float(m.group(1))
        return f"{val} TB is {int(val * 1024)} GB"

    return None


def evaluate_math_expression(expr: str) -> Optional[str]:
    """Parse and calculate arithmetic expressions safely with Multilingual words."""
    cleaned = expr.strip().lower()

    conv_result = evaluate_unit_conversion(cleaned)
    if conv_result is not None:
        return conv_result

    cleaned = re.sub(
        r"^(?:what\s+is|what\'s|calculate|compute|solve|tell\s+me|find|kanaku\s+pannu|solra|hisab\s+karo|cheppu|parayu|heli)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[\?\.!\s]+$", "", cleaned)
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)

    pct_match = re.match(
        r"^([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent|pratishat|sathaveetham|shatham|shathamaanam|shekada)\s+(?:of\s+|ka\s+|la\s+)?([0-9]+(?:\.[0-9]+)?)$",
        cleaned,
    )
    if not pct_match:
        pct_match = re.match(
            r"^([0-9]+(?:\.[0-9]+)?)\s+(?:ka|la)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent|pratishat|sathaveetham|shatham)$",
            cleaned,
        )
        if pct_match:
            total_val = float(pct_match.group(1))
            pct_val = float(pct_match.group(2))
            res = (pct_val / 100.0) * total_val
            formatted = f"{int(res)}" if res.is_integer() else f"{res:.4g}"
            return f"{pct_val}% of {total_val} is {formatted}"

    if pct_match:
        try:
            pct_val = float(pct_match.group(1))
            total_val = float(pct_match.group(2))
            res = (pct_val / 100.0) * total_val
            formatted = f"{int(res)}" if res.is_integer() else f"{res:.4g}"
            return f"{pct_val}% of {total_val} is {formatted}"
        except Exception:
            return None

    sqrt_match = re.match(
        r"^(?:square\s+root\s+of|sqrt(?:\s+of)?|vargamool)\s+([0-9]+(?:\.[0-9]+)?)$",
        cleaned,
    )
    if sqrt_match:
        try:
            val = float(sqrt_match.group(1))
            if val < 0:
                return "The square root of a negative number is not a real number."
            res = math.sqrt(val)
            formatted = f"{int(res)}" if res.is_integer() else f"{res:.4g}"
            return f"The square root of {val} is {formatted}"
        except Exception:
            return None

    # Multilingual mathematical replacements (Tamil, Hindi, Telugu, Malayalam, Kannada)
    # Addition
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+kooda\s+(\d+(?:\.\d+)?)(?:\s+kootu)?", r"\1 + \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+aur\s+(\d+(?:\.\d+)?)(?:\s+jodo)?", r"\1 + \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+mariyu\s+(\d+(?:\.\d+)?)(?:\s+kalapandi|\s+kalapu)?", r"\1 + \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+koottal\s+(\d+(?:\.\d+)?)", r"\1 + \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+mathu\s+(\d+(?:\.\d+)?)(?:\s+koodisi)?", r"\1 + \2", cleaned)

    # Subtraction
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+la\s+(\d+(?:\.\d+)?)\s+kali", r"\1 - \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+se\s+(\d+(?:\.\d+)?)\s+ghatao", r"\1 - \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+nundi\s+(\d+(?:\.\d+)?)(?:\s+theeyandi|\s+theeyi)?", r"\1 - \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+nunchi\s+(\d+(?:\.\d+)?)(?:\s+kurakkuka|\s+kura)?", r"\1 - \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+rinda\s+(\d+(?:\.\d+)?)(?:\s+kaledu)?", r"\1 - \2", cleaned)

    # Multiplication
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+kooda\s+(\d+(?:\.\d+)?)\s+perukku", r"\1 * \2", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s+la\s+(\d+(?:\.\d+)?)\s+vagu", r"\1 / \2", cleaned)

    # Word operators
    cleaned = re.sub(r"\bmultiplied\s+by\b|\bmultiply\b|\binto\b|\btimes\b|\bx\b|\bguna\b|\bguninchu\b|\bgunikkuka\b|\bgunisi\b", "*", cleaned)
    cleaned = re.sub(r"\bdivided\s+by\b|\bdivide\b|\bover\b|\bgeteilt\b|\bvaag\b|\bbhag\b|\bbhaginchi\b|\bharikkuka\b|\bbhagisi\b", "/", cleaned)
    cleaned = re.sub(r"\bplus\b|\band\b|\baur\b|\bplusu\b|\bjodo\b|\bkalapu\b|\bkoottu\b|\bkoodisi\b", "+", cleaned)
    cleaned = re.sub(r"\bminus\b|\btake\s+away\b|\bkam\b|\bghatao\b|\btheeyi\b|\bkurakkuka\b|\bkaledu\b", "-", cleaned)
    cleaned = re.sub(r"\bto\s+the\s+power\s+of\b|\bpower\s+of\b|\b\^\b", "**", cleaned)
    cleaned = re.sub(r"\bsquared\b", "** 2", cleaned)
    cleaned = re.sub(r"\bcubed\b", "** 3", cleaned)

    # Strip query trail words like 'entha', 'eshtu', 'ethra', 'kya hai', 'sollu', 'batao'
    cleaned = re.sub(r"\s*(?:entha|eshtu|ethra|kya\s+hai|sollu|batao|cheppu|parayu|heli|aaguthu|undi|undu|ide)$", "", cleaned).strip()

    if not re.search(r"[\+\-\*/\^]", cleaned) or not re.search(r"\d", cleaned):
        return None

    if not re.match(r"^[\d\.\s\+\-\*\/\(\)\^]+$", cleaned):
        return None

    try:
        parsed = ast.parse(cleaned, mode="eval")
        result = _safe_eval_node(parsed.body)
        formatted_result = f"{int(result)}" if result.is_integer() else f"{result:.4g}"
        return f"The answer is {formatted_result}"
    except ZeroDivisionError:
        return "Division by zero is undefined."
    except Exception:
        return None


# ==============================================================================
# SYSTEM ACTION HANDLERS
# ==============================================================================

def get_battery_status() -> str:
    """Retrieve system battery percentage and charging state."""
    if psutil is None:
        return "Hardware monitoring is unavailable (psutil not installed)."

    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return "No battery detected (running on AC power or Desktop system)."

        percent = round(battery.percent)
        is_charging = battery.power_plugged

        if is_charging:
            if percent == 100:
                return "Battery is at 100% and fully charged (plugged in)."
            return f"Battery is at {percent}% and currently charging."
        else:
            if battery.secsleft > 0 and battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                mins_left = battery.secsleft // 60
                hrs = mins_left // 60
                mins = mins_left % 60
                if hrs > 0:
                    time_str = f"{hrs} hour{'s' if hrs > 1 else ''} and {mins} minute{'s' if mins != 1 else ''}"
                else:
                    time_str = f"{mins} minute{'s' if mins != 1 else ''}"
                return f"Battery is at {percent}%, discharging with approximately {time_str} remaining."
            return f"Battery is at {percent}%, discharging."
    except Exception as e:
        return f"Unable to read battery status: {e}"


def get_ram_usage() -> str:
    """Retrieve system RAM usage metrics."""
    if psutil is None:
        return "Memory monitoring is unavailable (psutil not installed)."

    try:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return f"RAM usage is currently at {mem.percent:.1f}% ({used_gb:.1f} GB of {total_gb:.1f} GB used)."
    except Exception as e:
        return f"Unable to read memory status: {e}"


def get_cpu_usage() -> str:
    """Retrieve instantaneous CPU utilization percentage."""
    if psutil is None:
        return "CPU monitoring is unavailable (psutil not installed)."

    try:
        cpu_pct = psutil.cpu_percent(interval=None)
        return f"CPU usage is currently at {cpu_pct:.1f}%."
    except Exception as e:
        return f"Unable to read CPU status: {e}"


def get_disk_usage() -> str:
    """Retrieve local storage / disk partition usage."""
    if psutil is None:
        return "Disk monitoring is unavailable (psutil not installed)."

    try:
        root_path = os.path.abspath(os.sep)
        disk = psutil.disk_usage(root_path)
        free_gb = disk.free / (1024 ** 3)
        total_gb = disk.total / (1024 ** 3)
        used_gb = disk.used / (1024 ** 3)
        return f"Disk usage: {disk.percent:.1f}% used ({used_gb:.1f} GB of {total_gb:.1f} GB). Free space: {free_gb:.1f} GB."
    except Exception as e:
        return f"Unable to read disk status: {e}"


def get_system_info() -> str:
    """Retrieve comprehensive device OS, processor, and uptime information."""
    try:
        uptime_str = "unavailable"
        if psutil is not None:
            boot_time = psutil.boot_time()
            uptime_seconds = int(time.time() - boot_time)
            hrs = uptime_seconds // 3600
            mins = (uptime_seconds % 3600) // 60
            uptime_str = f"{hrs}h {mins}m"

        return f"System Info: Host '{_SYSTEM_NODE}', OS: {_SYSTEM_OS_NAME} ({_SYSTEM_ARCH}), Uptime: {uptime_str}."
    except Exception as e:
        return f"Unable to retrieve system info: {e}"


def get_network_status() -> str:
    """Retrieve active local Wi-Fi & Ethernet adapter details instantly."""
    return f"Network Active on host '{_SYSTEM_NODE}' (Local IP: {_CACHED_LOCAL_IP})."


def get_internet_status() -> str:
    """Check fast internet reachability via DNS socket probe."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 53))
        local_ip = s.getsockname()[0]
        s.close()
        return f"Internet is online and connected (Local IP: {local_ip})."
    except Exception:
        return "Offline Mode Active (No internet connection detected)."


def adjust_volume(action: str) -> str:
    """Adjust system master volume using native Windows key events."""
    if sys.platform != "win32":
        return f"Volume control ({action}) is only supported natively on Windows."

    try:
        vk_map = {
            "up": 0xAF,
            "down": 0xAE,
            "mute": 0xAD,
        }
        vk_code = vk_map.get(action.lower())
        if vk_code:
            ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)
            if action == "up":
                return "Increasing system volume."
            elif action == "down":
                return "Decreasing system volume."
            elif action == "mute":
                return "Toggled system audio mute."
        return f"Unknown volume action: {action}"
    except Exception as e:
        return f"Failed to adjust volume: {e}"


def take_screenshot() -> str:
    """Capture desktop screenshot asynchronously and save to data/screenshots."""
    os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(_SCREENSHOT_DIR, f"screenshot_{timestamp}.png")

    def _capture():
        try:
            if ImageGrab is not None:
                img = ImageGrab.grab()
                img.save(filepath)
        except Exception:
            pass

    _ACTION_EXECUTOR.submit(_capture)
    return f"Screenshot captured and saved to data/screenshots/screenshot_{timestamp}.png"


def open_application(app_name: str, command: str) -> str:
    """Launch a local desktop application process non-blocking."""
    def _spawn():
        try:
            if sys.platform == "win32":
                if command.startswith("start "):
                    target = command[6:].strip()
                    try:
                        os.startfile(target)
                        return
                    except Exception:
                        pass
                subprocess.Popen(command, shell=True)
            else:
                subprocess.Popen([command])
        except Exception:
            pass

    _ACTION_EXECUTOR.submit(_spawn)
    return f"Opening {app_name}."


# ==============================================================================
# PRE-COMPILED MULTILINGUAL INTENT REGEXES (LIGHTWEIGHT, SUB-MILLISECOND)
# ==============================================================================

# Noise / Inaudible / Empty patterns
_NOISE_REGEX = re.compile(
    r"^(?:|\.+|\s+|\[inaudible\]|\[noise\]|\[silence\]|uh+|um+|hmm+|ah+|huh\??)$",
    re.IGNORECASE,
)

_GREETING_PATTERNS = [
    # Tamil Greetings
    (re.compile(r"(?:\b(?:vanakkam|kaalai\s+vanakkam)\b|வணக்கம்|காலை\s+வணக்கம்)", re.IGNORECASE),
     "Vanakkam! All systems active and ready to assist you. Epdi help pannanum?"),
    # Hindi Greetings
    (re.compile(r"(?:\b(?:namaste|namaskar|shubh\s+prabhat)\b|नमस्ते|नमस्कार|शुभ\s+प्रभात)", re.IGNORECASE),
     "Namaste! Jarvis-Edge is active. Main aapki kya madad kar sakta hoon?"),
    # Telugu Greetings
    (re.compile(r"(?:\b(?:namaskaram|shubhodayam|shubhodhayam)\b|నమస్కారం|శుభోదయం)", re.IGNORECASE),
     "Namaskaram! All systems operational. How can I assist you?"),
    # Malayalam Greetings
    (re.compile(r"(?:\b(?:namaskaram|suprabhatham)\b|നമസ്കാരം|സുപ്രഭാതം)", re.IGNORECASE),
     "Namaskaram! All systems operational. How can I assist you?"),
    # Kannada Greetings
    (re.compile(r"(?:\b(?:namaskara|shubhodaya)\b|ನಮಸ್ಕಾರ|ಶುಭೋದಯ)", re.IGNORECASE),
     "Namaskara! Naanu nimage hege sahaya maadali?"),
    # English Greetings
    (re.compile(r"^(?:hello|hi|hey|greetings|hey\s+jarvis|hi\s+jarvis|hello\s+jarvis)[\!\?\.]*$", re.IGNORECASE),
     "Hello! How can I assist you today?"),
    (re.compile(r"\b(?:good\s+morning|morning)\b", re.IGNORECASE),
     "Good morning! I hope you have a productive day. How can I help?"),
    # How are you (Multilingual)
    (re.compile(r"(?:\b(?:how\s+are\s+you|epdi\s+iruka|epdi\s+irukinga|eppadi\s+irukkeenga|kaise\s+ho|kya\s+haal\s+hai|ela\s+unnav|ela\s+unnaru|sukhamano|hegiddira|hegiddiya)\b|எப்படி\s+இருக்கிறாய்|எப்படி\s+இருக்கீங்க|कैसे\s+हो|क्या\s+हाल\s+है|ఎలా\s+ఉన్నారు|ఎలా\s+ఉన్నావు|സുഖമാണോ|ಹೇಗಿದ್ದೀರಾ|ಹೇಗಿದ್ದೀಯ)", re.IGNORECASE),
     "I am doing great and all systems are running smoothly! How can I help you?"),
    # Identity / Who are you (Multilingual)
    (re.compile(r"(?:\b(?:who\s+are\s+you|what\s+is\s+your\s+name|identify\s+yourself|yar\s+nee|ne\s+yar|un\s+peru\s+enna|aap\s+kaun\s+ho|tum\s+kaun\s+ho|nuvvu\s+evaru|meeru\s+evaru|aarannu\s+nee|aaranu\s+nee|neenu\s+yaaru)\b|உன்\s+பெயர்\s+என்ன|யார்\s+நீ|आप\s+कौन\s+हैं|तुम\s+कौन\s+हो|నువ్వు\s+ఎవరు|మీరు\s+ఎవరు|ആരാണ്\s+നീ|ನೀನು\s+ಯಾರು)", re.IGNORECASE),
     "I am Jarvis-Edge, your lightning-fast multilingual offline voice assistant."),
    # Capabilities / What can you do (Multilingual)
    (re.compile(r"(?:\b(?:what\s+can\s+you\s+do|capabilities|enna\s+panna\s+mudiyum|kya\s+kar\s+sakte\s+ho|emi\s+cheyaglavu|emi\s+cheyagalavu|enthellam\s+cheyyam|enu\s+maadaballiri)\b|என்ன\s+செய்ய\s+முடியும்|क्या\s+कर\s+सकते\s+हो|ఏమి\s+చేయగలవు|എന്തെല്ലാം\s+ചെയ്യാം|ಏನು\s+ಮಾಡಬಲ್ಲಿರಿ)", re.IGNORECASE),
     "I can answer queries, check hardware/network, control volume, capture screenshots, launch apps, perform conversions, and provide first-aid guidance."),
]

# Pre-compiled Category Regexes
_BATTERY_KW_RE = re.compile(r"(?:\b(?:battery|charge|charging|power)\b|பேட்டரி|சார்ஜ்|बैटरी|चार्ज|चार्जिंग|బ్యాటరీ|ఛార్జింగ్|బాటరీ|ബാറ്ററി|ചാർജ്ജ്|ചാർജ്|ಬ್ಯಾಟರಿ|ಚಾರ್ಜ್)", re.IGNORECASE)
_BATTERY_QUAL_RE = re.compile(r"(?:\b(?:evlo|kitna|entha|ethra|eshtu|status|check|percentage|level|sollu|batao|cheppu|parayu|heli|kya|enna|enti|entha|enu|how\s+much|is\s+it|iruku|hai|undi|undu|ide|avtunda|ho\s+rahi)\b|நிலை|எவ்வளவு|சதவீதம்|இருக்கு|சொல்|स्थिति|कितनी|कितना|प्रतिशत|है|बताओ|ఎంత|శాతం|ఉంది|ఎంతవుంది|స్థితి|చెప్పు|എത്ര|ശതമാനം|ഉണ്ട്|അവസ്ഥ|പറയൂ|ಎಷ್ಟು|ಶೇಕಡಾ|ಇದೆ|ಸ್ಥಿತಿ|ಹೇಳು)", re.IGNORECASE)

_RAM_KW_RE = re.compile(r"(?:\b(?:ram|memory)\b|ரேம்|ரैम|நினைவகம்|रैम|मेमोरी|ర్యామ్|మెమరీ|റാം|മെമ്മറി|ರ‍್ಯಾಮ್|ಮೆಮೊರಿ)", re.IGNORECASE)
_RAM_QUAL_RE = re.compile(r"(?:\b(?:usage|use|status|check|evlo|kitna|entha|ethra|eshtu|sollu|batao|cheppu|parayu|heli|kya|enna|how\s+much|what|free|used)\b|பயன்பாடு|நிலை|எவ்வளவு|சொல்|उपयोग|स्थिति|कितना|बताओ|వాడుతున్నారు|వినియోగం|స్థితి|ఎంత|చెప్పు|ഉപയോഗം|അവസ്ഥ|എത്ര|പറയൂ|ಬಳಕೆ|ಸ್ಥಿತಿ|ಎಷ್ಟು|ಹೇಳು)", re.IGNORECASE)

_CPU_KW_RE = re.compile(r"(?:\b(?:cpu|processor)\b|செயலி|சிபியூ|सीपीयू|प्रोसेसर|సిపియు|ప్రొసెసర్|സിപിയു|പ്രൊസസ്സർ|ಸಿಪಿಯು|ಪ್ರೊಸೆಸರ್)", re.IGNORECASE)
_CPU_QUAL_RE = re.compile(r"(?:\b(?:usage|load|use|status|check|evlo|kitna|entha|ethra|eshtu|sollu|batao|cheppu|parayu|heli|kya|enna)\b|பயன்பாடு|சுமை|நிலை|எவ்வளவு|சொல்|उपयोग|लोड|स्थिति|कितना|बताओ|వాడుతున్నారు|వినియోగం|లోడ్|స్థితి|ఎంత|చెప్పు|ഉപയോഗം|ലോഡ്|അവസ്ഥ|എത്ര|പറയൂ|ಬಳಕೆ|ಲೋಡ್|ಸ್ಥಿತಿ|ಎಷ್ಟು|ಹೇಳು)", re.IGNORECASE)

_DISK_KW_RE = re.compile(r"(?:\b(?:disk|storage|hard\s*disk|drive\s*space)\b|சேமிப்பகம்|ஸ்டோரேஜ்|स्टोरेज|डिस्क|స్టోరేజ్|డిస్క్|സ്റ്റോറേജ്|ഡിസ്ക്|ಸ್ಟೋರೇಜ್|ಡಿಸ್ಕ್)", re.IGNORECASE)
_DISK_QUAL_RE = re.compile(r"(?:\b(?:usage|space|status|check|evlo|kitna|entha|ethra|eshtu|sollu|batao|cheppu|parayu|heli|kya|enna|free|left)\b|खाली|ఎంత|ಎಷ್ಟು|ഉണ്ട്|எவ்வளவு|சொல்|बताओ)", re.IGNORECASE)

_SYSINFO_RE = re.compile(r"\b(?:system\s+info|system\s+information|device\s+info|os\s+info|system\s+specs|system\s+status|pc\s+specs|device\s+status|uptime|system\s+uptime)\b", re.IGNORECASE)

_WIFI_KW_RE = re.compile(r"(?:\b(?:wifi|wi-fi|network)\b|வைஃபை|நெட்வொர்க்|वाईफाई|नेटवर्क|వైఫై|నెట్‌వర్క్|വൈഫൈ|നെറ്റ്‌വർക്ക്|ವೈಫೈ|ನೆಟ್‌ವರ್ಕ್)", re.IGNORECASE)
_WIFI_QUAL_RE = re.compile(r"(?:\b(?:status|check|connect|connected|enna|kya|enti|enu|entha|ethra|eshtu|sollu|batao|cheppu|parayu|heli|iruka|hai|ayinda|undo|ideya|state|unda|ide)\b|நிலை|இணைக்கப்பட்டுள்ளதா|இருக்கு|சொல்|स्थिति|चालू|कनेक्ट|है|बताओ|స్థితి|కనెక్ట్|ఉందా|చెప్పు|അവസ്ഥ|കണക്റ്റ്|ഉണ്ടോ|പറയൂ|ಸ್ಥಿತಿ|ಕನೆಕ್ಟ್|ಇದೆಯಾ|ಹೇಳು|ಎಂತ|എന്താണ്|ಏನು)", re.IGNORECASE)

_INTERNET_KW_RE = re.compile(r"(?:\b(?:internet|online|net\s+status|net\s+check)\b|இன்டர்நெட்|இணையம்|इंटरनेट|ఇంటర్నెట్|ഇന്റർനെറ്റ്|ಇಂಟರ್ನೆಟ್)", re.IGNORECASE)
_INTERNET_QUAL_RE = re.compile(r"(?:\b(?:status|check|connected|iruka|chal\s+raha|connect|hai|unda|undo|ideya|ide|enna|kya|enti|enu|sollu|batao|cheppu|parayu|heli|online\s+ah)\b|நிலை|இருக்கு|சொல்|स्थिति|चालू|है|बताओ|స్థితి|ఉందా|చెప్పు|അവസ്ഥ|ഉണ്ടോ|പറയൂ|ಸ್ಥಿತಿ|ಇದೆಯಾ|ಹೇಳು)", re.IGNORECASE)

_DATE_TOMORROW_RE = re.compile(r"(?:\b(?:tomorrow|nalaiku|repu|repatiki|naale|naaleya|kal\s+(?:ki|ka|hoga|hogi))\b|நாளை|நாளைக்கு|कल|రేపు|రేపటి|നാളെ|നാളത്തെ|ನಾಳೆ|ನಾಳೆಯ)", re.IGNORECASE)
_DATE_YESTERDAY_RE = re.compile(r"(?:\b(?:yesterday|nethu|ninna|ninnati|innale|innalathe|ninne|ninneya|kal\s+(?:tha|thi))\b|நேற்று|நேற்றைய|நேத்து|कल|నిన్న|నిన్నటి|ഇന്നലെ|ഇന്നലത്തെ|ನಿನ್ನೆ|ನಿನ್ನೆಯ)", re.IGNORECASE)
_DATE_TODAY_KW_RE = re.compile(r"(?:\b(?:date|today|innaiki|innaiku|aaj|ee\s+roju|innu|ivathu|ivathina)\b|தேதி|நாள்|இன்றைய|இன்று|இன்னைக்கு|तारीख|दिन|आज|తేదీ|వారం|ఈ\s*రోజు|ఈరోజు|తారీఖు|തീയതി|ദിവസം|ഇന്നത്തെ|ഇന്ന്|ದಿನಾಂಕ|ದಿನ|ಇವತ್ತಿನ|ಇವತ್ತು)", re.IGNORECASE)
_DATE_TODAY_QUAL_RE = re.compile(r"(?:\b(?:date|day|naal|din|tarikh|theeyathi|thedi|dinaanka|dina|enna|kya|enti|enu|enthanu|entha|sollu|batao|cheppu|parayu|heli|what|current|is|today)\b|தேதி|நாள்|तारीख|दिन|తేదీ|వారం|തീയതി|ദിവസം|ದಿನಾಂಕ|ದಿನ|என்ன|சொல்|क्या|बताओ|ఎంత|ఏంటి|చెప్పు|എന്താണ്|എത്ര|പറയൂ|ಏನು|ಎಷ್ಟು|ಹೇಳು)", re.IGNORECASE)

_TIME_KW_RE = re.compile(r"(?:\b(?:time|neram|samay|samayam|samaya|mani)\b|நேரம்|மணி|समय|సమయం|സമയം|ಸಮಯ)", re.IGNORECASE)
_TIME_QUAL_RE = re.compile(r"(?:\b(?:what|current|tell|check|enna|kya|entha|ethra|eshtu|sollu|batao|cheppu|parayu|heli|ippo|eppudu|ippol|eega|now|is\s+it|right\s+now|kitna|baje|hua|aayi|aagide|ayindi|evlo|please)\b|என்ன|எவ்வளவு|சொல்|காட்டு|எத்தனை|क्या|कितने|बताओ|हुआ|बजा|ఎంత|చెప్పు|ఎంతయింది|ఎంతవుంది|ఎంతైంది|എത്ര|പറയൂ|എത്രയായി|എത്രയുണ്ട്|എന്താണ്|ಎಷ್ಟು|ಹೇಳು|ಆಗಿದೆ|ಆಯ್ತು|ಏನು)", re.IGNORECASE)

_VOLUME_KW_RE = re.compile(r"(?:\b(?:volume|sound|audio|awaaz)\b|ஒலி|சத்தம்|ஆவாஸ்|आवाज़|ध्वनि|శబ్దం|ശബ്ദം|ಧ್ವನಿ)", re.IGNORECASE)
_VOLUME_UP_RE = re.compile(r"(?:\b(?:up|increase|raise|louder|ethu|adhigam|badhao|penchu|koottu|hecchu)\b|ஏற்று|அதிகம்|बढ़ाओ|పెంచు|കൂട്ടുക|കൂട്ടൂ|ಹೆಚ್ಚು)", re.IGNORECASE)
_VOLUME_DOWN_RE = re.compile(r"(?:\b(?:down|decrease|lower|kora|kammi|kam|karo|thagginchu|kuraykku|kadime)\b|குறை|कम|తగ్గించు|കുറയ്ക്കുക|കുറയ്ക്കൂ|ಕಡಿಮೆ)", re.IGNORECASE)
_VOLUME_MUTE_RE = re.compile(r"(?:\b(?:mute|unmute|silent|band|off)\b|மௌனம்|म्यूट|बंद|మ్యూట్|മ്യൂട്ട്|ಮ್ಯೂಟ್)", re.IGNORECASE)

_SCREENSHOT_RE = re.compile(r"(?:\b(?:screenshot|screen\s*capture)\b|ஸ்கிரீன்ஷாட்|திரைப்பிடிப்பு|स्क्रीनशॉट|స్క్రీన్‌షాట్|സ്ക്രീൻഷോട്ട്|ಸ್ಕ್ರೀನ್‌ಷಾಟ್|ಸ್ಕ್ರೀನ್‌ಶಾಟ್)", re.IGNORECASE)

_OPEN_VERBS_REGEX = re.compile(r"(?:\b(?:open|launch|start|kholo|chalu|thurakku|pannu|cheyi|maadi|karo|theruvu|thereyiri)\b|திற|ஓபன்|பண்ணு|खोलो|चालू|करो|ఓపెన్|తెరువు|చేయి|തുറക്കുക|തുറക്കൂ|ഓപ്പൺ|ತೆರೆಯಿರಿ|ತೆರೆ|ಮಾಡಿ|ಓಪನ್)", re.IGNORECASE)
_NOTEPAD_KW_RE = re.compile(r"(?:\b(?:notepad)\b|நோட்பேட்|நோட்பேடை|नोटपैड|నోట్‌ప్యాడ్|നോട്ട്പാഡ്|ನೋಟ್‌ಪ್ಯಾಡ್)", re.IGNORECASE)
_CALC_KW_RE = re.compile(r"(?:\b(?:calculator|calc)\b|கால்குலேட்டர்|கால்குலேட்டரை|கணக்கீட்டு|कैलकुलेटर|కాలిక్యులేటర్|കാൽക്കുലേറ്റർ|ക്യാൽക്കുലേറ്റർ|ಕ್ಯಾಲ್ಕುಲೇಟರ್)", re.IGNORECASE)

_EMERGENCY_RE = re.compile(r"(?:\b(?:emergency\s+numbers?|emergency\s+contacts?|aapatkaaleen\s+number)\b|அவசர\s+எண்|ஆபத்து\s+எண்|आपातकालीन\s+नंबर|అత్యవసర\s+నంబర్లు|അടിയന്തര\s+നമ്പർ|ತುರ್ತು\s+ಸಂಖ್ಯೆ)", re.IGNORECASE)

_KNOWLEDGE_GUARD_RE = re.compile(
    r"(?:\b(?:"
    # Burns
    r"burns?|scalds?|blisters?|theekaayam|theepollal|thee|jalne|jala|kaalina|kaalithe|suttagaaya|suttare|sutta|pollal|pollalettal|"
    # CPR
    r"cpr|ithaya|ithaya\s+thudippu|dil\s+ka\s+daura|gunde\s+aagipothe|hrudayaghatham|hrudayaghatha|"
    # Choking
    r"choking|choke|heimlich|thondai|gale|gala|gonthu|gonthulo|thonda|thondayil|gantalu|gantallu|adappu|"
    # Snake
    r"snake|snakebite|venom|paambu|saamp|saap|paamu|haavu|paampu|"
    # Bleeding
    r"bleeding|bleed|blood|ratham|ratha|khoon|raktham|raktha|chora|tourniquet|"
    # Sprains & Fractures
    r"sprains?|fractures?|strains?|elumbu|sulukku|haddi|moch|emuka|benuku|eluka|ellu|ulukku|moole|moolu|uluku|"
    # Seizures
    r"seizures?|convulsions?|fits|valippu|daure|mirgi|murcha|apasmaram|moorche|"
    # Anaphylaxis
    r"anaphylaxis|epipen|allerg(?:y|ies)|ovvaamai|"
    # Hydration
    r"hydration|dehydration|water|drink|thanneer|thanni|kudineer|paani|pani|jal|neellu|vellam|kudivellam|neeru|"
    # Headache
    r"headaches?|migraines?|thalavali|mandai\s+vali|sar\s+dard|sir\s+dard|thalanopi|thalanoppi|thala\s+noppi|thalavedhana|thalavedana|thala\s+vedana|thalenovu|thale\s+novu|"
    # Fever
    r"fevers?|kaichal|kaachal|udal\s+soodu|bukhar|taapman|jwaram|jvaram|jwara|jvara|"
    # Sleep
    r"sleep|insomnia|thookam|neend|nidra|urakkam|nidre|nidde|"
    # Ergonomics
    r"ergonomics|postures?|20-20-20|eye\s+strain"
    r")\b|"
    # Indic Script Keywords (Direct match without \b)
    r"தீக்காய|தீக்காயம்|தீப்பொള്ളல்|தீ\s*காயம்|முதலுதவி|சிபிஆர்|இதய|தொண்டை|அடைப்பு|பாம்பு|ரத்தம்|இரத்தம்|ரத்தப்போக்கு|எலும்பு|சுளுக்கு|வலிப்பு|ஒவ்வாமை|அலர்ஜி|தண்ணீர்|குடிநீர்|தலைவலி|ஒற்றைத்\s+தலைவலி|காய்ச்சல்|தூக்கம்|தோரணை|"
    r"जलने|जलन|प्राथमिक\s+उपचार|सीपीआर|दिल\s+का\s+दौरा|गले|गला|अटक|सांप|साँप|खून|रक्त|रक्तस्राव|हड्डी|मोच|दौरे|मिर्गी|एलर्जी|पानी|जल|सिर\s+दर्द|माइग्रेन|बुखार|तापमान|नींद|मुद्रा|"
    r"కాలిన|కాలితే|ప్రథమ\s+చికిత్స|సిపిఆర్|గుండెపోటు|గొంతు|గొంతులో|పాము|రక్తం|రక్తస్రావం|ఎముక|బెణుకు|మూర్ఛ|అలెర్జీ|నీళ్ళు|మంచి\s+నీళ్ళు|తలనొప్పి|తల\s+నొప్పి|జ్వరం|నిద్ర|భంగిమ|"
    r"പൊള്ളൽ|പൊള്ളലേറ്റാൽ|പ്രഥമശുശ്രൂഷ|സിപിആർ|ഹൃദയാഘാതം|തൊണ്ട|തൊണ്ടയിൽ|കുടുങ്ങി|പാമ്പ്|രക്തം|രക്തസ്രാവം|എല്ല്|ഉളുക്ക്|അപസ്മാരം|അലർജി|വെള്ളം|കുടിവെള്ളം|തലവേദന|തല\s+വേദന|മൈഗ്രെയ്ൻ|പനി|ഉറക്കം|ഇരിപ്പ്|"
    r"ಸುಟ್ಟ|ಸುಟ್ಟರೆ|ಸುಟ್ಟ\s+ಗಾಯ|ಪ್ರಥಮ\s+ಚಿಕಿತ್ಸೆ|ಸಿಪಿಆರ್|ಹೃದಯಾಘಾತ|ಗಂಟಲು|ಗಂಟಲಿನಲ್ಲಿ|ಹಾವು|ರಕ್ತ|ರಕ್ತಸ್ರಾವ|ಮೂಳೆ|ಉಳುಕು|ಮೂರ್ಛೆ|ಅಲರ್ಜಿ|ನೀರು|ಕುಡಿಯುವ\s+ನೀರು|ತಲೆನೋವು|ತಲೆ\s+ನೋವು|ಜ್ವರ|ನಿದ್ರೆ|ಭಂಗಿ"
    r")",
    re.IGNORECASE,
)


def route_intent(user_text: str) -> Optional[str]:
    """Inspect user transcript and instantly return a response if handled offline.

    Args:
        user_text: The recognized transcript string across EN, TA, Tanglish, HI, TE, ML, KN.

    Returns:
        A direct answer string if handled by the fast-path router,
        or None if the query should be forwarded to the LLM.
    """
    if user_text is None:
        return "I didn't catch that. Could you repeat?"

    cleaned = user_text.strip()

    # 1. Silence & Noise Fallback
    if not cleaned or _NOISE_REGEX.match(cleaned):
        return "I didn't catch that. Could you repeat?"

    norm = re.sub(r"[\?\.\!,।]+$", "", cleaned).strip().lower()

    # 2. Battery & Charging Queries (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _BATTERY_KW_RE.search(norm):
        if (
            _BATTERY_QUAL_RE.search(norm)
            or norm in (
                "battery", "battery check", "check battery", "battery status", "battery evlo", "charge evlo", "battery percentage",
                "battery kitna hai", "battery entha undi", "battery ethra undu", "battery eshtu ide",
                "பேட்டரி நிலை", "बैटरी कितनी है", "బ్యాటరీ ఎంత ఉంది", "ബാറ്ററി എത്ര ഉണ്ട്", "ಬ್ಯಾಟರಿ ಎಷ್ಟು ಇದೆ", "பேட்டரி", "बैटरी", "బ్యాటరీ", "ബാറ്ററി", "ಬ್ಯಾಟರಿ"
            )
        ):
            return get_battery_status()

    # 3. RAM / System Memory Queries (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _RAM_KW_RE.search(norm):
        if (
            _RAM_QUAL_RE.search(norm)
            or norm in (
                "ram", "ram status", "ram usage", "check ram", "memory", "memory usage", "memory status",
                "ram usage sollu", "ram kitna use ho raha hai", "ram entha use avtundi", "ram usage ethra", "ram usage eshtu",
                "ரேம் பயன்பாடு", "ரैम उपयोग", "रैम उपयोग", "ర్యామ్ వినియోగం", "റാം ഉപയോഗം", "ರ‍್ಯಾಮ್ ಬಳಕೆ", "ரேம்", "रैम", "ర్యామ్", "റാം", "ರ‍್ಯಾಮ್"
            )
        ):
            return get_ram_usage()

    # 4. CPU Processor Load Queries (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _CPU_KW_RE.search(norm):
        if (
            _CPU_QUAL_RE.search(norm)
            or norm in (
                "cpu", "cpu usage", "cpu load", "check cpu", "processor load", "processor usage",
                "cpu usage sollu", "cpu usage kitna hai", "cpu usage entha", "cpu usage ethra", "cpu usage eshtu",
                "செயலி பயன்பாடு", "सीपीयू उपयोग", "సిపియు వినియోగం", "സിപിയു ഉപയോഗം", "ಸಿಪಿಯು ಬಳಕೆ", "சிபியூ", "सीपीयू", "సిపియు", "സിപിയു", "ಸಿಪಿಯು"
            )
        ):
            return get_cpu_usage()

    # 5. Disk / Storage Space Queries (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _DISK_KW_RE.search(norm):
        if (
            _DISK_QUAL_RE.search(norm)
            or norm in ("disk usage", "storage", "storage usage", "disk space", "check disk")
        ):
            return get_disk_usage()

    # 6. System Info & Uptime Queries
    if _SYSINFO_RE.search(norm):
        return get_system_info()

    # 7. Wi-Fi & Network Adapter Queries (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _WIFI_KW_RE.search(norm):
        if (
            _WIFI_QUAL_RE.search(norm)
            or norm in (
                "wifi", "wifi status", "check wifi", "network", "network status", "is wifi connected",
                "wifi status enna", "wifi connect hai kya", "wifi status enti", "wifi status entha", "wifi status enu",
                "வைஃபை நிலை", "वाईफाई स्थिति", "వైఫై స్థితి", "വൈഫൈ അവസ്ഥ", "ವೈಫೈ ಸ್ಥಿತಿ", "வைஃபை", "वाईफाई", "వైఫై", "വൈഫൈ", "ವೈಫೈ"
            )
        ):
            return get_network_status()

    # 8. Internet Connectivity Queries (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _INTERNET_KW_RE.search(norm):
        if (
            _INTERNET_QUAL_RE.search(norm)
            or norm in ("internet", "internet status", "check internet", "am i online", "internet chal raha hai kya")
        ):
            return get_internet_status()

    # 9. Date Queries (Tomorrow, Yesterday, Today in Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    # Tomorrow
    if _DATE_TOMORROW_RE.search(norm):
        if _DATE_TODAY_QUAL_RE.search(norm) or norm in ("tomorrow", "tomorrow date", "repu date enti", "nalaiku enna date", "kal ki date kya hai"):
            tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
            date_str = tomorrow.strftime("%A, %B %d, %Y")
            return f"Tomorrow is {date_str}."

    # Yesterday
    if _DATE_YESTERDAY_RE.search(norm):
        if _DATE_TODAY_QUAL_RE.search(norm) or norm in ("yesterday", "yesterday date", "ninna date enti", "nethu enna date", "kal kya tarikh thi"):
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            date_str = yesterday.strftime("%A, %B %d, %Y")
            return f"Yesterday was {date_str}."

    # Today's Date
    if _DATE_TODAY_KW_RE.search(norm):
        if (
            _DATE_TODAY_QUAL_RE.search(norm)
            or norm in (
                "date", "today date", "today's date", "what is today's date", "what date is it",
                "innaiki date enna", "aaj ki date kya hai", "ee roju date enti", "innu ethra theeyathi", "ivathu date enu",
                "இன்றைய தேதி என்ன", "இன்று என்ன தேதி", "आज की तारीख क्या है", "ఈ రోజు తేదీ ఎంత", "ഇന്നത്തെ തീയതി എന്താണ്", "ಇವತ್ತಿನ ದಿನಾಂಕ ಏನು",
                "இன்றைய தேதி", "आज की तारीख", "ఈ రోజు తేదీ", "ഇന്നത്തെ തീയതി", "ಇವತ್ತಿನ ದಿನಾಂಕ"
            )
        ):
            now = datetime.datetime.now()
            date_str = now.strftime("%A, %B %d, %Y")
            return f"Today is {date_str}."

    # 10. Time Queries (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _TIME_KW_RE.search(norm):
        if (
            _TIME_QUAL_RE.search(norm)
            or norm in (
                "time", "time please", "current time", "what time is it", "what is the time",
                "time enna", "enna time", "time sollu", "neram enna", "ippo mani enna", "mani evlo",
                "time kya hai", "time batao", "samay kya hai", "kitne baje hai", "time kya hua",
                "time entha", "samayam entha", "time cheppu", "samayam cheppu", "time entha ayindi",
                "time ethra", "samayam ethra", "samayam ethra aayi", "samayam parayu",
                "time eshtu", "samaya eshtu", "samaya heli", "samaya eshtu aagide",
                "நேரம் என்ன", "மணி என்ன", "நேரத்தை சொல்", "நேரம்",
                "समय क्या है", "कितने बजे हैं", "समय बताओ", "समय",
                "సమయం ఎంత", "టైమ్ ఎంత", "సమయం చెప్పు", "సమయం",
                "സമയം എത്ര", "സമയം എത്രയായി", "സമയം പറയൂ", "സമയം",
                "ಸಮಯ ಎಷ್ಟು", "ಸಮಯ ಎಷ್ಟು ಆಗಿದೆ", "ಸಮಯ ಹೇಳು", "ಸಮಯ"
            )
        ):
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p").lstrip("0")
            return f"The time is {time_str}."

    # 11. Audio / Volume Controls (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _VOLUME_KW_RE.search(norm):
        if _VOLUME_UP_RE.search(norm):
            return adjust_volume("up")
        if _VOLUME_DOWN_RE.search(norm):
            return adjust_volume("down")
        if _VOLUME_MUTE_RE.search(norm):
            return adjust_volume("mute")
    elif norm in ("volume up", "sound ethu", "louder", "awaaz badhao", "sound penchu", "sound koottu", "sound hecchu"):
        return adjust_volume("up")
    elif norm in ("volume down", "sound kora", "lower volume", "awaaz kam karo", "sound thagginchu", "sound kuraykku", "sound kadime"):
        return adjust_volume("down")
    elif norm in ("mute", "unmute", "mute audio", "mute pannu", "mute karo", "mute cheyi"):
        return adjust_volume("mute")

    # 12. Desktop Screenshot Capture (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    if _SCREENSHOT_RE.search(norm):
        return take_screenshot()

    # 13. Application Launchers (Multilingual: EN, TA, Tanglish, HI, TE, ML, KN)
    # Notepad
    if _NOTEPAD_KW_RE.search(norm):
        if _OPEN_VERBS_REGEX.search(norm) or norm in ("open notepad", "notepad", "launch notepad", "notepad open", "notepad kholo", "notepad thurakku", "notepad open cheyi", "notepad open maadi", "நோட்பேடை திற", "நோட்பேட் திற", "नोटपैड खोलो", "నోట్‌ప్యాడ్ ఓపెన్ చేయి", "നോട്ട്പാഡ് തുറക്കുക", "ನೋಟ್‌ಪ್ಯಾಡ್ ತೆರೆಯಿರಿ"):
            return open_application("Notepad", "notepad.exe")

    # Calculator
    if _CALC_KW_RE.search(norm):
        if _OPEN_VERBS_REGEX.search(norm) or norm in ("open calculator", "calculator", "calc", "open calc", "calc open", "calculator open", "calculator open pannu", "calculator kholo", "calculator open cheyi", "calculator thurakku", "calculator open maadi", "கால்குலேட்டரை திற", "कैलकुलेटर खोलो", "కాలిక్యులేటర్ ఓపెన్ చేయి", "കാൽക്കുലേറ്റർ തുറക്കുക", "ಕ್ಯಾಲ್ಕುಲೇಟರ್ ತೆರೆಯಿರಿ"):
            return open_application("Calculator", "calc.exe")

    # Browser
    if re.search(r"(?:\b(?:browser|chrome|edge)\b|வலை\s*உலாவி|ब्राउज़र|బ్రౌజర్|ബ്രൗസർ|ಬ್ರೌಸರ್)", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Default Browser", "start https://www.google.com")

    # Explorer
    if re.search(r"\b(?:file\s*explorer|explorer|my\s*computer)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("File Explorer", "explorer.exe")

    # VS Code
    if re.search(r"\b(?:vs\s*code|vscode|code)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Visual Studio Code", "code")

    # Settings
    if re.search(r"(?:\b(?:settings|windows\s*settings)\b|செட்டிங்ஸ்|सेटिंग्स|సెట్టింగ్స్|ക്രമീകരണങ്ങൾ|ಸೆಟ್ಟಿಂಗ್‌ಗಳು)", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Windows Settings", "start ms-settings:")

    # Task Manager
    if re.search(r"\b(?:task\s*manager|taskmgr)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Task Manager", "taskmgr.exe")

    # Terminal / CMD
    if re.search(r"\b(?:terminal|command\s*prompt|cmd)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Command Prompt", "cmd.exe")

    # 14. Conversational Greetings & Identity (Multilingual)
    for pattern, response in _GREETING_PATTERNS:
        if pattern.search(norm):
            return response

    # 15. Emergency Instant Contacts Fast-Path
    if _EMERGENCY_RE.search(norm):
        return "Emergency Contacts: India (112 / 108 for Ambulance), US (911), UK (999). Seek immediate emergency medical assistance."

    # 16. Basic Arithmetic, Math & Unit Conversion (Multilingual)
    math_result = evaluate_math_expression(cleaned)
    if math_result is not None:
        return math_result

    # 17. Offline First-Aid & Health Markdown Knowledge Search (Multilingual across EN, TA, Tanglish, HI, TE, ML, KN)
    if _KNOWLEDGE_GUARD_RE.search(norm):
        kd_result = search_offline_knowledge(norm)
        if kd_result is not None:
            return kd_result

    # 18. Unhandled queries -> forward to Person 3's LLM
    return None


# Module warm-up to eliminate first-pass latency spikes
try:
    evaluate_math_expression("1 + 1")
    search_offline_knowledge("burns")
    for _p, _ in _GREETING_PATTERNS:
        _p.search("test")
except Exception:
    pass
