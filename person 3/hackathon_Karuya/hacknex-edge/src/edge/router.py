"""
router.py — Fast Rule-Based Router, Math Engine, Typo Corrector, and Windows System Inspector
Person 1 & System Lead — HackNEX 2026 HNX26EPS08

Implements the Router protocol from contracts.py:
    route(user_text: str) -> str | None

Features:
  1. Instant Greetings, Identity, & Everyday Common Questions (<1ms)
  2. Typo & Spelling Mistake Tolerance (Automatic fuzzy normalization)
  3. Live System Time and Date
  4. High-Precision Math Calculator (Safe Python evaluation with math module)
  5. Real-Time Windows System & Hardware Stats (CPU, RAM, Disk, Battery, OS version)
  6. Local File & Directory Exploration (Real-time listing of files in directory/Windows)
  7. Offline Jarvis App Launcher (Notepad, Calculator, Task Manager, Explorer, etc.)
  8. Project Astra Multimodal & Assistant Overview (/astra)
  9. Structured Prompt Generator (/prompt <topic>)
  10. Multi-Language Translation Assistant (/translate <text> to <lang>)
  11. Ethical Hacking & Cyber Security Educational Guide (/hack <topic>)
"""
from __future__ import annotations

import datetime
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class FastRouter:
    """
    Sub-5ms deterministic router handling greetings, typo normalization,
    precise mathematics, system telemetry, prompt generation, translation,
    and ethical hacking roadmaps.
    """

    TYPO_MAP = {
        "calcualte": "calculate",
        "calclate": "calculate",
        "calulate": "calculate",
        "claculate": "calculate",
        "calclutor": "calculator",
        "solv": "solve",
        "compte": "compute",
        "systm": "system",
        "sytem": "system",
        "systme": "system",
        "sytem32": "system32",
        "system 32": "system32",
        "wat": "what",
        "wht": "what",
        "wats": "what is",
        "whats": "what is",
        "liss": "list",
        "lit": "list",
        "lsit": "list",
        "filse": "files",
        "fls": "files",
        "fiels": "files",
        "notpad": "notepad",
        "notepd": "notepad",
        "taskmngr": "taskmgr",
        "taskmanagr": "taskmgr",
        "tasmgr": "taskmgr",
        "explor": "explorer",
        "promt": "prompt",
        "promts": "prompt",
        "prpmpt": "prompt",
        "traslate": "translate",
        "translte": "translate",
        "translat": "translate",
        "hak": "hack",
        "hck": "hack",
        "hackin": "hacking",
        "haking": "hacking",
        "helo": "hello",
        "hllo": "hello",
        "thx": "thank you",
        "thnx": "thank you",
        "tks": "thanks",
    }

    def __init__(self, workspace_root: Path | None = None):
        if workspace_root is None:
            self.workspace_root = Path(__file__).resolve().parents[2]
        else:
            self.workspace_root = Path(workspace_root)

    def normalize_typos(self, text: str) -> str:
        """Correct common typing mistakes and phonetic variations."""
        words = text.split()
        fixed = [self.TYPO_MAP.get(w.lower(), w) for w in words]
        return " ".join(fixed)

    def route(self, user_text: str) -> Optional[str]:
        """
        Evaluate user_text against fast deterministic rules.
        Returns response string if handled, or None to pass to LLM.
        """
        raw = user_text.strip()
        lower = self.normalize_typos(raw.lower())

        # 1. Project Astra Overview
        astra_resp = self._check_astra(lower)
        if astra_resp:
            return astra_resp

        # 2. Prompt Generator (/prompt <topic>)
        prompt_resp = self._check_prompt_gen(lower, raw)
        if prompt_resp:
            return prompt_resp

        # 3. Translation (/translate <text> to <lang>)
        trans_resp = self._check_translation(lower, raw)
        if trans_resp:
            return trans_resp

        # 4. Ethical Hacking & Cyber Security Tutor (/hack <topic>)
        hack_resp = self._check_ethical_hacking(lower, raw)
        if hack_resp:
            return hack_resp

        # 5. Greetings & Common Conversational Questions
        greeting_resp = self._check_greetings(lower)
        if greeting_resp:
            return greeting_resp

        # 6. Time & Date
        time_resp = self._check_time_date(lower)
        if time_resp:
            return time_resp

        # 7. Offline Jarvis App Launches
        app_resp = self._check_app_launch(lower)
        if app_resp:
            return app_resp

        # 8. Live Windows System & Hardware Telemetry
        sys_resp = self._check_system_stats(lower)
        if sys_resp:
            return sys_resp

        # 9. Local Filesystem & Directory Inspection
        fs_resp = self._check_filesystem(lower, raw)
        if fs_resp:
            return fs_resp

        # 10. Mathematical Calculations (Exact Python Solver)
        math_resp = self._check_math(lower, raw)
        if math_resp:
            return math_resp

        return None

    # -----------------------------------------------------------------------
    # Rule 1: Project Astra Multimodal Capabilities
    # -----------------------------------------------------------------------
    def _check_astra(self, lower: str) -> Optional[str]:
        if lower in ("astra", "/astra", "what is astra", "project astra", "astra features", "what can astra do"):
            return (
                "[ASTRA] Project Astra Multimodal Capabilities in ED-RA:\n\n"
                "Google Project Astra represents universal AI assistance: seeing, hearing, and reasoning in real-time.\n"
                "In ED-RA, we bring these capabilities straight to your local PC:\n"
                "1. [Vision] Multimodal Vision: Inspect photos, screenshots & diagrams via `/image <path> [question]`.\n"
                "2. [Voice] Natural Voice: Expressive Indian English neural voices (Neerja/Prabhat) + offline Piper.\n"
                "3. [OS] Local Perception: Live system telemetry, file discovery, and Windows OS control.\n"
                "4. [Creative] Creative Intelligence: High-quality prompt generation via `/prompt <topic>`.\n"
                "5. [Language] Translation: Multilingual translations via `/translate <text> to <lang>`.\n"
                "6. [Security] Cyber Security & Coding: In-depth ethical hacking tutorials & production code generation."
            )
        return None

    # -----------------------------------------------------------------------
    # Rule 2: Prompt Generator
    # -----------------------------------------------------------------------
    def _check_prompt_gen(self, lower: str, raw: str) -> Optional[str]:
        topic = None
        if lower.startswith("/prompt"):
            topic = raw[7:].strip()
        elif lower.startswith("generate a prompt for"):
            topic = raw[21:].strip()
        elif lower.startswith("create a prompt for"):
            topic = raw[19:].strip()
        elif lower.startswith("write a prompt for"):
            topic = raw[18:].strip()

        if not topic:
            return None

        clean_topic = topic.strip().strip("'\"")
        if not clean_topic:
            return "Please specify a topic. Example: `/prompt python web scraper` or `/prompt sci-fi story`."

        return (
            f"[PROMPT GENERATOR] Generated High-Impact AI Prompt for '{clean_topic}':\n\n"
            f"--- [COPY PROMPT BELOW] ---\n"
            f"# ROLE & PERSONA:\n"
            f"You are a world-class domain expert specializing in {clean_topic}.\n\n"
            f"# OBJECTIVE:\n"
            f"Deliver an exceptionally thorough, production-grade, and structured solution for: {clean_topic}.\n\n"
            f"# GUIDELINES & CONSTRAINTS:\n"
            f"1. Explain key concepts step-by-step with zero fluff.\n"
            f"2. Provide concrete, copy-pasteable examples and edge-case handling.\n"
            f"3. Follow modern industry standards, clean code, and best practices.\n"
            f"4. Highlight common pitfalls and security/performance optimizations.\n\n"
            f"# DESIRED OUTPUT FORMAT:\n"
            f"Use Markdown headings, bullet points, and syntax-highlighted code blocks where applicable.\n"
            f"--- [END PROMPT] ---"
        )

    # -----------------------------------------------------------------------
    # Rule 3: Translation Assistant
    # -----------------------------------------------------------------------
    def _check_translation(self, lower: str, raw: str) -> Optional[str]:
        # Fast translation dictionary for common phrases
        phrases = {
            "hello": {"tamil": "Vanakkam", "hindi": "Namaste", "french": "Bonjour", "spanish": "Hola", "german": "Hallo"},
            "how are you": {"tamil": "Eppadi irukireergal?", "hindi": "Aap kaise hain?", "french": "Comment allez-vous?", "spanish": "Como estas?", "german": "Wie geht es dir?"},
            "thank you": {"tamil": "Nandri", "hindi": "Dhanyavaad", "french": "Merci", "spanish": "Gracias", "german": "Danke"},
            "good morning": {"tamil": "Kaalai Vanakkam", "hindi": "Shubh Prabhat", "french": "Bonjour", "spanish": "Buenos dias", "german": "Guten Morgen"},
            "good night": {"tamil": "Iniya Iravu", "hindi": "Shubh Ratri", "french": "Bonne nuit", "spanish": "Buenas noches", "german": "Gute Nacht"},
            "welcome": {"tamil": "Nalvaravu", "hindi": "Swagat hai", "french": "Bienvenue", "spanish": "Bienvenido", "german": "Willkommen"},
        }

        # Check for phrase match
        for p, trans in phrases.items():
            if lower == f"translate {p}" or lower == f"translate {p} to tamil":
                return f"'{p.capitalize()}' in Tamil: {trans['tamil']}"
            if lower == f"translate {p} to hindi":
                return f"'{p.capitalize()}' in Hindi: {trans['hindi']}"
            if lower == f"translate {p} to french":
                return f"'{p.capitalize()}' in French: {trans['french']}"
            if lower == f"translate {p} to spanish":
                return f"'{p.capitalize()}' in Spanish: {trans['spanish']}"
            if lower == f"translate {p} to german":
                return f"'{p.capitalize()}' in German: {trans['german']}"
            if lower == f"translate {p}":
                return (
                    f"Translations for '{p.capitalize()}':\n"
                    f"  - Tamil:   {trans['tamil']}\n"
                    f"  - Hindi:   {trans['hindi']}\n"
                    f"  - French:  {trans['french']}\n"
                    f"  - Spanish: {trans['spanish']}\n"
                    f"  - German:  {trans['german']}"
                )

        if lower.startswith("/translate"):
            # If not in simple phrase list, let LLM handle it
            return None

        return None

    # -----------------------------------------------------------------------
    # Rule 4: Ethical Hacking & Cyber Security Educational Tutor
    # -----------------------------------------------------------------------
    def _check_ethical_hacking(self, lower: str, raw: str) -> Optional[str]:
        triggers = [
            "/hack", "teach me hacking", "how to learn hacking", "learn ethical hacking",
            "ethical hacking roadmap", "how to become a hacker", "penetration testing steps",
            "phases of hacking", "hacking basics"
        ]
        if any(lower == t or lower.startswith(t + " ") for t in triggers):
            return (
                "[SECURITY] Ethical Hacking & Penetration Testing Roadmap (Step-by-Step):\n\n"
                "1. Foundations (Networking & Linux):\n"
                "   - Networking: TCP/IP 3-way handshake, OSI layers, DNS, subnetting, Wireshark.\n"
                "   - OS: Linux command line, Bash scripting, Windows internals (System32, processes, registry).\n\n"
                "2. 5 Phases of Ethical Penetration Testing:\n"
                "   - Phase 1: Reconnaissance / OSINT (Passive information gathering: whois, DNS recon, Shodan).\n"
                "   - Phase 2: Scanning & Enumeration (Active discovery: Nmap port scanning, service banners).\n"
                "   - Phase 3: Vulnerability Assessment (Searching CVEs, analyzing web apps, misconfigurations).\n"
                "   - Phase 4: Exploitation (Proving security vulnerabilities ethically in authorized labs).\n"
                "   - Phase 5: Post-Exploitation & Reporting (Privilege escalation, documenting findings, remediation).\n\n"
                "3. Core Web Security (OWASP Top 10):\n"
                "   - SQL Injection (SQLi), Cross-Site Scripting (XSS), CSRF, Broken Authentication, IDOR.\n\n"
                "4. Free Authorized Legal Practice Labs:\n"
                "   - TryHackMe (Beginner friendly), Hack The Box (Intermediate/Advanced), PortSwigger Web Security Academy."
            )

        if "explain nmap" in lower or "what is nmap" in lower:
            return (
                "[NMAP] Network Mapper Essential Guide:\n"
                "Nmap is the industry-standard network security scanner used for discovery and vulnerability analysis.\n\n"
                "Key Scan Commands:\n"
                "  - nmap -sS <target> : Stealth SYN Scan (Fast, doesn't complete TCP handshake)\n"
                "  - nmap -sV <target> : Service Version Detection (Identifies software running on open ports)\n"
                "  - nmap -sC <target> : Default NSE Scripts (Checks for common vulnerabilities & banners)\n"
                "  - nmap -p- <target> : Scan all 65,535 TCP ports\n"
                "  - nmap -A <target>  : Aggressive Scan (Combines OS detection, versioning, script scanning, traceroute)"
            )

        if "explain sql injection" in lower or "what is sql injection" in lower or "explain sqli" in lower:
            return (
                "[WEB SECURITY] SQL Injection (SQLi) Overview & Remediation:\n\n"
                "  - What it is: A vulnerability where untrusted user input is directly concatenated into a backend SQL database query.\n"
                "  - Vulnerable Example: query = \"SELECT * FROM users WHERE user = '\" + input + \"'\"\n"
                "  - Attacker Input: ' OR '1'='1' --\n"
                "  - Result: The condition '1'='1' is always true, bypassing authentication and dumping database records.\n\n"
                "[DEFENSE] Remediation:\n"
                "1. Parameterized Queries / Prepared Statements (Never concatenate raw strings into SQL).\n"
                "2. Use ORM frameworks (SQLAlchemy, Prisma, Hibernate).\n"
                "3. Enforce Principle of Least Privilege on database user accounts."
            )

        return None

    # -----------------------------------------------------------------------
    # Rule 5: Greetings & Common Conversational Questions
    # -----------------------------------------------------------------------
    def _check_greetings(self, lower: str) -> Optional[str]:
        # Greetings
        if lower in ("hello", "hi", "hey", "hello there", "good morning", "good evening", "good afternoon"):
            return "Hello! I am ED-RA, your ultra-fast offline AI Brain and assistant. How can I assist you today?"

        # Common conversation questions
        if lower in ("how are you", "how are you doing", "how are you today"):
            return "I am performing at 100% efficiency! CPU, memory, and neural inference engines are fully operational. How can I help you?"

        if lower in ("who are you", "what is your name", "introduce yourself", "tell me about yourself", "who made you"):
            return (
                "I am ED-RA, a versatile on-device AI Brain engineered for Windows. "
                "I operate 100% offline with zero cloud dependency, featuring natural Indian English speech, "
                "local GGUF LLM inference, Windows system telemetry, mathematics, and comprehensive offline knowledge."
            )

        if lower in ("what can you do", "help", "commands", "features"):
            return (
                "I can assist you with:\n"
                "• Technical problem solving, programming, and system architecture\n"
                "• Mathematics, science, and step-by-step problem solving\n"
                "• Windows system diagnostics, live telemetry, and local file exploration\n"
                "• Natural speech conversations, creative writing, and prompt assistance"
            )

        if lower in ("thank you", "thanks", "thank you so much", "thanks a lot"):
            return "You are very welcome! Let me know if you need anything else."

        if lower in ("bye", "goodbye", "see you", "exit"):
            return "Goodbye! Have a productive session."

        return None

    # -----------------------------------------------------------------------
    # Rule 6: Time & Date
    # -----------------------------------------------------------------------
    def _check_time_date(self, lower: str) -> Optional[str]:
        if any(p in lower for p in ("what time is it", "current time", "tell me the time", "what's the time", "time now")):
            now = datetime.datetime.now()
            return f"The current system time is {now.strftime('%I:%M:%S %p')}."
        if any(p in lower for p in ("what is today's date", "what's today's date", "current date", "what is the date", "today's date")):
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."
        return None

    # -----------------------------------------------------------------------
    # Rule 7: Offline Jarvis App Launcher
    # -----------------------------------------------------------------------
    def _check_app_launch(self, lower: str) -> Optional[str]:
        if sys.platform != "win32":
            return None

        apps = {
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "calc": ["calc.exe"],
            "task manager": ["taskmgr.exe"],
            "taskmgr": ["taskmgr.exe"],
            "command prompt": ["cmd.exe"],
            "cmd": ["cmd.exe"],
            "powershell": ["powershell.exe"],
            "file explorer": ["explorer.exe"],
            "explorer": ["explorer.exe"],
            "registry editor": ["regedit.exe"],
            "regedit": ["regedit.exe"],
            "event viewer": ["eventvwr.exe"],
        }
        for name, cmd in apps.items():
            if lower in (f"open {name}", f"launch {name}", f"start {name}"):
                try:
                    subprocess.Popen(cmd, shell=True)
                    return f"Launched {name.capitalize()} on your Windows system."
                except Exception as e:
                    return f"Could not launch {name}: {e}"
        return None

    # -----------------------------------------------------------------------
    # Rule 8: Live Windows System & Hardware Telemetry
    # -----------------------------------------------------------------------
    def _check_system_stats(self, lower: str) -> Optional[str]:
        trigger_words = [
            "system info", "system stats", "cpu usage", "ram usage",
            "memory usage", "battery status", "hardware specs", "system status"
        ]
        if any(w in lower for w in trigger_words):
            os_info = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
            hostname = platform.node()
            cpu_name = platform.processor() or "AMD/Intel x86_64"

            lines = [
                f"• OS: {os_info}",
                f"• Hostname: {hostname}",
                f"• CPU Model: {cpu_name}",
            ]

            if _PSUTIL_AVAILABLE:
                cpu_pct = psutil.cpu_percent(interval=0.1)
                cores_phys = psutil.cpu_count(logical=False)
                cores_log = psutil.cpu_count(logical=True)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage("C:\\")

                lines.append(f"• CPU Load: {cpu_pct:.1f}% ({cores_phys} physical / {cores_log} logical cores)")
                lines.append(f"• Memory: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB used ({mem.percent}%)")
                lines.append(f"• System Drive C:\\: {disk.free / (1024**3):.1f} GB free out of {disk.total / (1024**3):.1f} GB")

                battery = psutil.sensors_battery()
                if battery:
                    plugged = "Plugged in" if battery.power_plugged else "On battery"
                    lines.append(f"• Battery: {battery.percent:.0f}% ({plugged})")

            return "Current Windows System Status:\n" + "\n".join(lines)
        return None

    # -----------------------------------------------------------------------
    # Rule 9: Local Filesystem & Directory Inspection
    # -----------------------------------------------------------------------
    def _check_filesystem(self, lower: str, raw: str) -> Optional[str]:
        # List files in current directory or workspace
        if lower in (
            "list files", "dir", "ls", "what files are in this folder",
            "show files", "show files in current directory", "list current directory",
            "show directory contents", "what files are here"
        ):
            target_dir = Path.cwd()
            try:
                entries = sorted(target_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                lines = [f"Files and folders in {target_dir}:"]
                for p in entries[:30]:  # limit to top 30 items
                    if p.is_dir():
                        lines.append(f"  [DIR]  {p.name}/")
                    else:
                        size_kb = p.stat().st_size / 1024
                        lines.append(f"  [FILE] {p.name} ({size_kb:.1f} KB)")
                if len(entries) > 30:
                    lines.append(f"  ... and {len(entries) - 30} more items.")
                return "\n".join(lines)
            except Exception as e:
                return f"Error listing directory: {e}"

        # Explain Windows System32 / Windows structure specifically
        if "what is in system32" in lower or "where is system32" in lower or "explain system32" in lower:
            return (
                "C:\\Windows\\System32 is the core 64-bit Windows operating system directory. "
                "It contains vital OS binaries (ntoskrnl.exe, smss.exe, csrss.exe, wininit.exe, services.exe, lsass.exe, svchost.exe), "
                "system consoles (services.msc, eventvwr.msc, taskmgr.exe), device drivers (System32\\drivers), "
                "the hosts file (System32\\drivers\\etc\\hosts), and Windows Registry hives (System32\\config)."
            )

        if "windows file structure" in lower or "windows directory structure" in lower:
            return (
                "Windows Directory Structure Hierarchy:\n"
                "• C:\\Windows : Core operating system files\n"
                "• C:\\Windows\\System32 : 64-bit kernel binaries, DLLs, system utilities\n"
                "• C:\\Windows\\SysWOW64 : 32-bit compatibility binaries on 64-bit OS\n"
                "• C:\\Windows\\WinSxS : Side-by-side component store and assembly cache\n"
                "• C:\\Program Files : 64-bit installed software\n"
                "• C:\\Program Files (x86) : 32-bit installed software\n"
                "• C:\\ProgramData : Shared cross-user application settings\n"
                "• C:\\Users\\<User> : User profiles, Desktop, Documents, Downloads\n"
                "• C:\\Users\\<User>\\AppData\\Local & Roaming : Per-user application data and configurations."
            )

        return None

    # -----------------------------------------------------------------------
    # Rule 10: High-Precision Math Calculator
    # -----------------------------------------------------------------------
    def _check_math(self, lower: str, raw: str) -> Optional[str]:
        math_prefixes = ["calculate ", "what is ", "solve ", "compute ", "evaluate "]
        expr_str = ""

        for pref in math_prefixes:
            if lower.startswith(pref):
                cand = lower[len(pref):].strip().rstrip("?").rstrip(".")
                if re.search(r"[\d\+\-\*\/\^\%\(\)]", cand):
                    expr_str = cand
                    break

        if not expr_str:
            clean = raw.strip().rstrip("?")
            if re.match(r"^[\d\s\+\-\*\/\^\%\(\)\.\,]+$", clean) and any(op in clean for op in "+-*/^%"):
                expr_str = clean

        if not expr_str:
            return None

        # Sanitize and convert operators
        expr_sanitized = expr_str.replace("^", "**").replace("x", "*").replace("X", "*")
        expr_sanitized = expr_sanitized.replace("pi", "math.pi").replace("e", "math.e")

        allowed_names = {
            "math": math,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs,
            "pow": pow,
            "round": round,
            "pi": math.pi,
            "e": math.e,
        }

        if not re.match(r"^[a-zA-Z0-9\s\+\-\*\/\%\(\)\.\,]+$", expr_sanitized):
            return None

        disallowed = ["import", "eval", "exec", "open", "os", "sys", "__", "lambda", "class", "def"]
        if any(d in expr_sanitized.lower() for d in disallowed):
            return None

        try:
            res = eval(expr_sanitized, {"__builtins__": {}}, allowed_names)
            if isinstance(res, float):
                if res.is_integer():
                    res = int(res)
                else:
                    res = round(res, 6)
            return f"Result: {expr_str} = {res}"
        except Exception:
            return None


if __name__ == "__main__":
    router = FastRouter()
    print("Testing Router with Typo Normalization & Astra Features:")
    test_queries = [
        "hllo",
        "astra",
        "/prompt python automation script",
        "translate hello to tamil",
        "teach me hacking",
        "explain nmap",
        "wat time is it",
        "systm info",
        "lit filse",
        "calclate 50 * 4 + sqrt(64)",
    ]
    for q in test_queries:
        ans = router.route(q)
        print(f"Query: '{q}'\n  -> {ans[:120] if ans else 'None (Passes to LLM)'}...\n")
