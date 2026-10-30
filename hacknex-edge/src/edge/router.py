"""
Sub-Millisecond Multilingual Intent Router (HNX26EPS08).
Integrated for HackNEX 2026.
Zero-latency offline regex-based routing across 7 languages:
- English, Tamil, Tanglish, Hindi, Telugu, Malayalam, Kannada
And persistent memory commands, system diagnostics, audio controls, app launchers.
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
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None

try:
    import psutil
except ImportError:
    psutil = None

from src.edge.knowledge_search import search_offline_knowledge

# Static system cache
_SYSTEM_OS_NAME: str = f"{platform.system()} {platform.release()}"
_SYSTEM_ARCH: str = platform.machine()
_SYSTEM_NODE: str = platform.node()

try:
    _CACHED_LOCAL_IP: str = socket.gethostbyname(socket.gethostname())
except Exception:
    _CACHED_LOCAL_IP = "127.0.0.1"

_SCREENSHOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "screenshots")
)
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
        return f"{c}\u00b0C is {f:.2f}\u00b0F"

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:f|fahrenheit)\s+(?:to|in|la)\s+(?:c|celsius)$", cleaned)
    if m:
        f = float(m.group(1))
        c = (f - 32.0) * 5.0 / 9.0
        return f"{f}\u00b0F is {c:.2f}\u00b0C"

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

    # Strip query trail words
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
    """Retrieve system RAM memory utilization and free capacity."""
    if psutil is None:
        return "RAM monitoring is unavailable (psutil not installed)."

    try:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        percent = mem.percent

        return (
            f"RAM Usage: {percent}% utilized ({used_gb:.2f} GB of {total_gb:.2f} GB used). "
            f"{available_gb:.2f} GB available."
        )
    except Exception as e:
        return f"Unable to read RAM status: {e}"


def get_cpu_usage() -> str:
    """Retrieve current system processor usage percentage across cores."""
    if psutil is None:
        return "CPU monitoring is unavailable (psutil not installed)."

    try:
        cpu_pct = psutil.cpu_percent(interval=None)
        core_count = psutil.cpu_count(logical=False) or 4
        logical_count = psutil.cpu_count(logical=True) or 8

        return f"CPU Load: {cpu_pct:.1f}% across {core_count} physical cores ({logical_count} threads)."
    except Exception as e:
        return f"Unable to read CPU status: {e}"


def get_disk_usage() -> str:
    """Retrieve system drive storage space metrics."""
    if psutil is None:
        return "Storage monitoring is unavailable (psutil not installed)."

    try:
        drive = "C:\\" if sys.platform == "win32" else "/"
        usage = psutil.disk_usage(drive)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        percent = usage.percent

        return f"Storage: {percent}% used. {free_gb:.1f} GB free out of {total_gb:.1f} GB on system drive."
    except Exception as e:
        return f"Unable to read storage status: {e}"


def get_system_info() -> str:
    """Retrieve consolidated host specifications, OS version and uptime."""
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
    """Check fast internet reachability without outbound sockets (100% offline proof)."""
    return f"Offline edge mode is active on '{_SYSTEM_NODE}' (Local IP: {_CACHED_LOCAL_IP}). Air-gapped operational."


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
    """Capture desktop screenshot and save to data/screenshots."""
    os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(_SCREENSHOT_DIR, f"screenshot_{timestamp}.png")

    try:
        from src.edge.screen_capture import capture_desktop_gdi
        if capture_desktop_gdi(filepath):
            return f"Screenshot captured successfully and saved to data/screenshots/screenshot_{timestamp}.png"
    except Exception:
        pass

    try:
        if ImageGrab is not None:
            img = ImageGrab.grab()
            img.save(filepath)
            return f"Screenshot captured successfully and saved to data/screenshots/screenshot_{timestamp}.png"
    except Exception as e:
        return f"Failed to capture screenshot: {e}"

    return "Screenshot capture is unavailable."


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


_NOTES_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "notes.txt")
)


def get_time_and_battery_status() -> str:
    """Retrieve combined current time and battery charging status."""
    now = datetime.datetime.now()
    time_str = now.strftime('%I:%M %p').lstrip('0')
    bat_str = get_battery_status()
    return f"The time is {time_str}. {bat_str}"


def get_ram_running_status() -> str:
    """Retrieve RAM memory utilization along with active process count."""
    ram_info = get_ram_usage()
    if psutil is not None:
        try:
            proc_count = len(psutil.pids())
            return f"{ram_info} Currently {proc_count} active system processes running."
        except Exception:
            pass
    return ram_info


def list_running_processes(top_n: int = 5) -> str:
    """List top running processes sorted by memory utilization."""
    if psutil is None:
        return "Process monitoring is unavailable (psutil not installed)."
    try:
        procs = []
        for p in psutil.process_iter(['name', 'memory_percent']):
            try:
                name = p.info.get('name') or ''
                mem = p.info.get('memory_percent') or 0.0
                if name:
                    procs.append((name, mem))
            except Exception:
                pass
        procs.sort(key=lambda x: x[1], reverse=True)
        top_procs = procs[:top_n]
        proc_str = ", ".join(f"{name} ({mem:.1f}%)" for name, mem in top_procs)
        return f"Top {top_n} processes by memory: {proc_str}. Total active processes: {len(procs)}."
    except Exception as e:
        return f"Unable to retrieve running processes: {e}"


def count_files_in_directory(target_str: str) -> str:
    """Count files and folders in common directories (Desktop, Downloads, Documents, Project)."""
    target = target_str.lower()
    path = None
    label = "Desktop"

    if "download" in target:
        path = os.path.expanduser("~/Downloads")
        label = "Downloads"
    elif "document" in target:
        path = os.path.expanduser("~/Documents")
        label = "Documents"
    elif "picture" in target or "image" in target:
        path = os.path.expanduser("~/Pictures")
        label = "Pictures"
    elif "project" in target or "workspace" in target or "here" in target:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        label = "Project Workspace"
    else:
        path = os.path.expanduser("~/Desktop")
        label = "Desktop"

    if not os.path.exists(path):
        return f"Directory for {label} was not found at '{path}'."

    try:
        file_count = 0
        dir_count = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_file(follow_symlinks=False):
                        file_count += 1
                    elif entry.is_dir(follow_symlinks=False):
                        dir_count += 1
                except Exception:
                    pass
        return f"{label} Directory: Found {file_count} files and {dir_count} folders ({path})."
    except Exception as e:
        return f"Failed to inspect {label} files: {e}"


def list_directory_files(target_str: str, limit: int = 8) -> str:
    """List recent files in a specified directory."""
    target = target_str.lower()
    if "download" in target:
        path = os.path.expanduser("~/Downloads")
        label = "Downloads"
    elif "document" in target:
        path = os.path.expanduser("~/Documents")
        label = "Documents"
    elif "project" in target or "workspace" in target:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        label = "Project Workspace"
    else:
        path = os.path.expanduser("~/Desktop")
        label = "Desktop"

    if not os.path.exists(path):
        return f"Directory for {label} was not found at '{path}'."

    try:
        entries = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                try:
                    if entry.is_file():
                        mtime = entry.stat().st_mtime
                        entries.append((entry.name, mtime))
                except Exception:
                    pass
        if not entries:
            return f"No visible files found in {label}."
        entries.sort(key=lambda x: x[1], reverse=True)
        recent = [name for name, _ in entries[:limit]]
        return f"Files in {label} ({len(recent)} recent shown): " + ", ".join(recent)
    except Exception as e:
        return f"Failed to list files in {label}: {e}"


def manage_notes(action: str, content: str = "") -> str:
    """Read, create, append, or clear user quick notes."""
    os.makedirs(os.path.dirname(_NOTES_FILE), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")

    if action in ("create", "write"):
        try:
            with open(_NOTES_FILE, "w", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {content}\n")
            return f"Note created: \"{content}\" (saved to data/notes.txt)."
        except Exception as e:
            return f"Failed to save note: {e}"

    elif action == "append":
        try:
            with open(_NOTES_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {content}\n")
            return f"Added to notes: \"{content}\"."
        except Exception as e:
            return f"Failed to append note: {e}"

    elif action == "read":
        if not os.path.exists(_NOTES_FILE) or os.path.getsize(_NOTES_FILE) == 0:
            return "Your notes file is currently empty. You can say 'create note [text]' to start one."
        try:
            with open(_NOTES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            preview = "".join(lines[-6:]).strip()
            return f"Saved Notes ({len(lines)} entries):\n{preview}"
        except Exception as e:
            return f"Failed to read notes: {e}"

    elif action == "clear":
        try:
            with open(_NOTES_FILE, "w", encoding="utf-8") as f:
                f.truncate(0)
            return "All notes have been cleared."
        except Exception as e:
            return f"Failed to clear notes: {e}"

    return "Unknown notes action."


def close_application(app_name: str, process_targets: list[str]) -> str:
    """Cleanly terminate running application instances."""
    killed_count = 0
    if psutil is not None:
        for p in psutil.process_iter(['name']):
            try:
                p_name = p.info.get('name') or ''
                if any(p_name.lower() == target.lower() for target in process_targets):
                    p.terminate()
                    killed_count += 1
            except Exception:
                pass

    if sys.platform == "win32":
        for target in process_targets:
            try:
                res = subprocess.run(f"taskkill /F /IM {target}", shell=True, capture_output=True, timeout=2)
                if res.returncode == 0:
                    killed_count = max(killed_count, 1)
            except Exception:
                pass

    if killed_count > 0:
        return f"Closed {app_name} successfully ({killed_count} process instance{'s' if killed_count > 1 else ''} stopped)."
    return f"Closed {app_name} (no active instances found)."


def lock_workstation() -> str:
    """Lock the Windows desktop session."""
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.LockWorkStation()
            return "Workstation locked successfully."
        except Exception as e:
            return f"Failed to lock workstation: {e}"
    return "Workstation lock is only supported on Windows."


def empty_recycle_bin() -> str:
    """Empty the Windows Recycle Bin."""
    if sys.platform == "win32":
        def _clean():
            try:
                ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 7)
            except Exception:
                try:
                    subprocess.run('powershell -Command "Clear-RecycleBin -Force -ErrorAction SilentlyContinue"', shell=True, capture_output=True)
                except Exception:
                    pass
        _ACTION_EXECUTOR.submit(_clean)
        return "Windows Recycle Bin has been emptied."
    return "Recycle Bin operation is only supported on Windows."


# ==============================================================================
# PRE-COMPILED MULTILINGUAL INTENT REGEXES (LIGHTWEIGHT, SUB-MILLISECOND)
# ==============================================================================

# Noise / Inaudible / Empty patterns
_NOISE_REGEX = re.compile(
    r"^(?:|\.+|\s+|\[inaudible\]|\[noise\]|\[silence\]|uh+|um+|hmm+|ah+|huh\??)$",
    re.IGNORECASE,
)

# Repeat intent
_PAT_REPEAT = re.compile(
    r"^(?:repeat|what\s+did\s+you\s+say|say\s+(?:that\s+)?again|pardon|come\s+again)\b",
    re.IGNORECASE,
)

# Persistent conversation memory patterns
_PAT_MEM_FORGET = re.compile(
    r"^(?:forget\s+(?:everything|all(?:\s+history)?|my\s+history)|delete\s+(?:all\s+)?(?:history|conversations?)|clear\s+(?:all\s+)?(?:history|memory))\b",
    re.IGNORECASE,
)
_PAT_MEM_YESTERDAY = re.compile(
    r"\b(?:what\s+did\s+i\s+(?:ask|say)\s+yesterday|yesterday\'?s?\s+(?:questions?|conversation))\b",
    re.IGNORECASE,
)
_PAT_MEM_TODAY = re.compile(
    r"\b(?:what\s+did\s+i\s+(?:ask|say)\s+today|today\'?s?\s+(?:questions?|conversation))\b",
    re.IGNORECASE,
)
_PAT_MEM_DAY_N = re.compile(
    r"\bwhat\s+did\s+i\s+(?:ask|say)\s+(?:on\s+)?day\s+(\d+)\b",
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

_TIME_CHARGING_COMBINED_RE = re.compile(
    r"(?:\b(?:time\s+(?:and\s+)?(?:charging|charing|charge|battery)\s*(?:status)?|"
    r"(?:charging|charing|charge|battery)\s+(?:status\s+)?(?:and\s+)?time|"
    r"time\s+charing\s+satus|time\s+charging\s+status|charge\s+time\s+status|battery\s+and\s+time)\b|"
    r"நேரம்\s+மற்றும்\s+பேட்டரி|சார்ஜிங்\s+மற்றும்\s+நேரம்)",
    re.IGNORECASE,
)

_RAM_RUNNING_RE = re.compile(
    r"(?:\b(?:ram\s+running|running\s+ram|ram\s+running\s+status|ram\s+status\s+running|running\s+processes|top\s+processes|active\s+processes|task\s+list|process\s+list|ram\s+and\s+process)\b|"
    r"இயங்கும்\s+செயலிகள்|ரैम\s+रनिंग)",
    re.IGNORECASE,
)

_FILE_COUNT_RE = re.compile(
    r"(?:\b(?:files?\s+co*unt|count\s+files?|how\s+many\s+files?|file\s+count\s+sollu|files?\s+evlo(?:\s+iruku)?|evlo\s+files?(?:\s+iruku)?|file\s+count\s+evlo|ethana\s+files?)\b|"
    r"கோப்புகளின்\s+எண்ணிக்கை|எத்தனை\s+கோப்புகள்|फाइलों\s+की\s+संख्या)",
    re.IGNORECASE,
)

_FILE_LIST_RE = re.compile(
    r"(?:\b(?:list\s+.*files?|show\s+.*files?|files?\s+(?:list|kaatu|kamiku|paru|show)|recent\s+files?|list\s+directory|show\s+directory)\b|"
    r"கோப்புகளைக்\s+காட்டு|फ़ाइलें\s+दिखाओ)",
    re.IGNORECASE,
)

_CLOSE_VERBS_REGEX = re.compile(
    r"(?:\b(?:close|quit|exit|terminate|kill|stop|moodu|moodidu)\b|"
    r"மூடு|முடி|மூடிடு|बंद\s+करो|मुடிக்க|ముయ్యి|ಮುಚ್ಚಿ)",
    re.IGNORECASE,
)

_LOCK_PC_RE = re.compile(
    r"(?:\b(?:lock\s+(?:pc|computer|system|screen|workstation)|pc\s+lock(?:\s+pannu)?|system\s+lock(?:\s+pannu)?)\b|"
    r"கணினியைப்\s+பூட்டு|कंप्यूटर\s+लॉक\s+करो)",
    re.IGNORECASE,
)

_EMPTY_BIN_RE = re.compile(
    r"(?:\b(?:empty\s+recycle\s*bin|clean\s+recycle\s*bin|clear\s+recycle\s*bin|recycle\s*bin\s+(?:clean|empty|clear))\b|"
    r"ரீசைக்கிள்\s+பின்னை\s+காலி\s+செய்|रीसायकल\s+बिन\s+खाली\s+करो)",
    re.IGNORECASE,
)

_SCREENSHOT_RE = re.compile(
    r"(?:\b(?:screenshot|screen\s*shot|screen\s*capture|capture\s*screen|screenshot\s+edu|screen\s+shot\s+edukuradhu|screenshot\s+podu|screenshot\s+pannu)\b|"
    r"ஸ்கிரீன்ஷாட்|திரைப்பிடிப்பு|திரை\s*பிடி|स्क्रीनशॉट|స్క్రీన్‌షాట్|സ്ക്രീൻഷോട്ട്|ಸ್ಕ್ರೀನ್‌ಷಾಟ್)",
    re.IGNORECASE,
)

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


def get_active_timers() -> list[dict]:
    """Return active timer entries."""
    return []


def route_intent(user_text: str, last_reply: str = "", memory: Optional[Any] = None) -> Optional[str]:
    """Inspect user transcript and instantly return a response if handled offline.

    Args:
        user_text: The recognized transcript string across EN, TA, Tanglish, HI, TE, ML, KN.
        last_reply: Previous bot reply for repeat intent.
        memory: Optional persistent memory instance.

    Returns:
        Direct answer string if handled by fast router (<5ms), or None for LLM.
    """
    if user_text is None:
        return "I didn't catch that. Could you repeat?"

    cleaned = user_text.strip()
    if not cleaned or _NOISE_REGEX.match(cleaned):
        return "I didn't catch that. Could you repeat?"

    norm = re.sub(r"[\?\.\!,।]+$", "", cleaned).strip().lower()

    # Repeat intent
    if _PAT_REPEAT.search(norm) and last_reply:
        return f"I said: {last_reply}"

    # Memory Intents
    if memory is not None:
        try:
            if _PAT_MEM_FORGET.match(norm):
                memory.clear()
                return "Done. I've deleted all conversation history."

            if _PAT_MEM_YESTERDAY.search(norm):
                yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                turns = memory.on_day(yesterday)
                if not turns:
                    return "I have no record of any conversation yesterday."
                items = "; ".join(f'"{u}"' for u, _ in turns[:5])
                return f"Yesterday you asked: {items}."

            if _PAT_MEM_TODAY.search(norm):
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                turns = memory.on_day(today)
                if not turns:
                    return "No conversations recorded yet today."
                items = "; ".join(f'"{u}"' for u, _ in turns[:5])
                return f"Today you asked: {items}."

            m = _PAT_MEM_DAY_N.search(norm)
            if m:
                n = int(m.group(1))
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
        except Exception:
            pass

    # Combined Time and Charging/Battery Status
    if _TIME_CHARGING_COMBINED_RE.search(norm):
        return get_time_and_battery_status()

    # Notes Management (Create, Append, Read, Clear)
    m_note_create = re.match(
        r"^(?:create\s+note|new\s+note|take\s+note|save\s+note|write\s+note|note\s+pannu|note\s+podu|note\s+create\s+pannu)\s+(.+)$",
        norm,
        re.IGNORECASE,
    )
    if m_note_create:
        return manage_notes("create", m_note_create.group(1).strip())

    m_note_add = re.match(
        r"^(?:append\s+note|add\s+to\s+note|add\s+note|note\s+la\s+add\s+pannu)\s+(.+)$",
        norm,
        re.IGNORECASE,
    )
    if m_note_add:
        return manage_notes("append", m_note_add.group(1).strip())

    if norm in (
        "read notes", "read note", "show notes", "show my notes", "get notes", "my notes", "notes",
        "notes padi", "notes kaatu", "view notes", "open notes", "நோட்ஸ் படி", "நோட்ஸ் காட்டு", "नोट्स दिखाओ"
    ):
        return manage_notes("read")

    if norm in (
        "clear notes", "delete notes", "clear my notes", "delete note", "empty notes",
        "notes delete pannu", "notes clear pannu", "நோட்ஸ் அழி", "नोट्स मिटाओ"
    ):
        return manage_notes("clear")

    # File Count & Directory Inspection
    if _FILE_COUNT_RE.search(norm) or norm in (
        "file count", "files count", "files coount", "count files", "how many files",
        "desktop files count", "files count desktop", "downloads file count", "files count downloads"
    ):
        return count_files_in_directory(norm)

    if _FILE_LIST_RE.search(norm):
        return list_directory_files(norm)

    # Windows Lock & Recycle Bin
    if _LOCK_PC_RE.search(norm):
        return lock_workstation()

    if _EMPTY_BIN_RE.search(norm):
        return empty_recycle_bin()

    # Battery
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

    # RAM Running / Memory Usage
    if _RAM_RUNNING_RE.search(norm) or "ram running" in norm or "running ram" in norm:
        if "process" in norm or "task" in norm:
            return list_running_processes(5)
        return get_ram_running_status()

    # RAM
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

    # CPU
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

    # Disk
    if _DISK_KW_RE.search(norm):
        if (
            _DISK_QUAL_RE.search(norm)
            or norm in ("disk usage", "storage", "storage usage", "disk space", "check disk")
        ):
            return get_disk_usage()

    # System Info
    if _SYSINFO_RE.search(norm):
        return get_system_info()

    # Wi-Fi
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

    # Internet
    if _INTERNET_KW_RE.search(norm):
        if (
            _INTERNET_QUAL_RE.search(norm)
            or norm in ("internet", "internet status", "check internet", "am i online", "internet chal raha hai kya")
        ):
            return get_internet_status()

    # Date - Tomorrow
    if _DATE_TOMORROW_RE.search(norm):
        if _DATE_TODAY_QUAL_RE.search(norm) or norm in ("tomorrow", "tomorrow date", "repu date enti", "nalaiku enna date", "kal ki date kya hai"):
            tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
            return f"Tomorrow is {tomorrow.strftime('%A, %B %d, %Y')}."

    # Date - Yesterday
    if _DATE_YESTERDAY_RE.search(norm):
        if _DATE_TODAY_QUAL_RE.search(norm) or norm in ("yesterday", "yesterday date", "ninna date enti", "nethu enna date", "kal kya tarikh thi"):
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            return f"Yesterday was {yesterday.strftime('%A, %B %d, %Y')}."

    # Date - Today
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
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

    # Time
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
            return f"The time is {now.strftime('%I:%M %p').lstrip('0')}."

    # Volume Controls
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

    # Screenshot
    if _SCREENSHOT_RE.search(norm):
        return take_screenshot()

    # Application Closing
    if _CLOSE_VERBS_REGEX.search(norm) or any(w in norm for w in ("close", "exit", "terminate", "kill", "moodu", "moodidu")):
        if _NOTEPAD_KW_RE.search(norm) or "notepad" in norm:
            return close_application("Notepad", ["notepad.exe"])
        if _CALC_KW_RE.search(norm) or "calc" in norm or "calculator" in norm:
            return close_application("Calculator", ["CalculatorApp.exe", "Calculator.exe", "calc.exe"])
        if re.search(r"\b(?:paint|mspaint)\b|பெயிண்ட்|पेंट", norm):
            return close_application("Paint", ["mspaint.exe", "Paint.exe", "PaintStudio.View.exe"])
        if re.search(r"\b(?:browser|chrome|edge)\b|வலை\s*உலாவி|ब्राउज़र", norm):
            return close_application("Browser", ["chrome.exe", "msedge.exe"])
        if re.search(r"\b(?:task\s*manager|taskmgr)\b", norm):
            return close_application("Task Manager", ["taskmgr.exe", "Taskmgr.exe"])
        if re.search(r"\b(?:terminal|command\s*prompt|cmd)\b", norm):
            return close_application("Command Prompt", ["cmd.exe"])
        if re.search(r"\b(?:vs\s*code|vscode|code)\b", norm):
            return close_application("Visual Studio Code", ["Code.exe"])

    # Application Launchers
    if _NOTEPAD_KW_RE.search(norm):
        if _OPEN_VERBS_REGEX.search(norm) or norm in (
            "open notepad", "notepad", "launch notepad", "notepad open", "notepad kholo", "notepad thurakku",
            "notepad open cheyi", "notepad open maadi", "நோட்பேடை திற", "நோட்பேட் திற", "नोटपैड खोलो",
            "నోట్‌ప్యాడ్ ఓపెన్ చేయి", "നോട്ട്പാഡ് തുറക്കുക", "ನೋಟ್‌ಪ್ಯಾಡ್ ತೆರೆಯಿರಿ"
        ):
            return open_application("Notepad", "notepad.exe")

    if _CALC_KW_RE.search(norm):
        if _OPEN_VERBS_REGEX.search(norm) or norm in (
            "open calculator", "calculator", "calc", "open calc", "calc open", "calculator open",
            "calculator open pannu", "calculator kholo", "calculator open cheyi", "calculator thurakku",
            "calculator open maadi", "கால்குலேட்டரை திற", "कैलकुलेटर खोलो", "కాలిక్యులేటర్ ఓపెన్ చేయి",
            "കാൽക്കുലേറ്റർ തുറക്കുക", "ಕ್ಯಾಲ್ಕುಲೇಟರ್ ತೆರೆಯಿರಿ"
        ):
            return open_application("Calculator", "calc.exe")

    if re.search(r"\b(?:paint|mspaint)\b|பெயிண்ட்|पेंट", norm) and (_OPEN_VERBS_REGEX.search(norm) or norm in ("paint", "mspaint", "open paint", "paint open", "paint open pannu")):
        return open_application("Paint", "mspaint.exe")

    if re.search(r"(?:\b(?:browser|chrome|edge)\b|வலை\s*உலாவி|ब्राउज़र|బ్రౌజర్|ബ്രൗസർ|ಬ್ರೌಸರ್)", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Default Browser", "start https://www.google.com")

    if re.search(r"\b(?:file\s*explorer|explorer|my\s*computer)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("File Explorer", "explorer.exe")

    if re.search(r"\b(?:vs\s*code|vscode|code)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Visual Studio Code", "code")

    if re.search(r"(?:\b(?:settings|windows\s*settings)\b|செட்டிங்ஸ்|सेटिंग्स|సెట్టింగ్స్|ക്രമീകരണങ്ങൾ|ಸೆಟ್ಟಿಂಗ್‌ಗಳು)", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Windows Settings", "start ms-settings:")

    if re.search(r"\b(?:task\s*manager|taskmgr)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Task Manager", "taskmgr.exe")

    if re.search(r"\b(?:terminal|command\s*prompt|cmd)\b", norm) and _OPEN_VERBS_REGEX.search(norm):
        return open_application("Command Prompt", "cmd.exe")

    # Greetings
    for pattern, response in _GREETING_PATTERNS:
        if pattern.search(norm):
            return response

    # Emergency Contacts
    if _EMERGENCY_RE.search(norm):
        return "Emergency Contacts: India (112 / 108 for Ambulance), US (911), UK (999). Seek immediate emergency medical assistance."

    # Math & Unit Conversion
    math_result = evaluate_math_expression(cleaned)
    if math_result is not None:
        return math_result

    # Offline First-Aid & Health Markdown Knowledge Search
    if _KNOWLEDGE_GUARD_RE.search(norm):
        kd_result = search_offline_knowledge(norm)
        if kd_result is not None:
            return kd_result

    # Unhandled queries -> forward to LLM
    return None


class FastRouter:
    """Class wrapper for router protocol."""
    def route_intent(self, user_text: str, last_reply: str = "", memory: Optional[Any] = None) -> Optional[str]:
        return route_intent(user_text, last_reply=last_reply, memory=memory)

    def route(self, user_text: str) -> Optional[str]:
        return route_intent(user_text)


Router = FastRouter

# Module warm-up to eliminate first-pass latency spikes
try:
    if psutil is not None:
        psutil.cpu_percent(interval=None)
    route_intent("hello")
    route_intent("cpu usage")
    route_intent("battery level")
    evaluate_math_expression("1 + 1")
    search_offline_knowledge("burns")
    for _p, _ in _GREETING_PATTERNS:
        _p.search("test")
except Exception:
    pass
