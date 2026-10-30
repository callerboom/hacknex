"""
llm.py — Local GGUF LLM streaming inference via llama-cpp-python.
Person 3 — HackNEX 2026 HNX26EPS08

Implements LLM protocol from contracts.py:
    stream(user_text: str, history: list[Turn]) -> Iterator[str]

Features:
  - Loads small Q4/Q5 GGUF (e.g. Qwen2.5-0.5B or 1.5B)
  - n_ctx=512, n_batch=128, CPU-only, AVX2/FMA
  - KV-Cache reuse: system prompt pre-loaded at startup
  - Threads pinned to limits.yaml declared core count
  - Yields token strings one by one (streaming)
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterator, List

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import llama_cpp (avoid import error when not installed)
# ---------------------------------------------------------------------------
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    logger.warning("llama_cpp not available - LLM will run in FALLBACK stub mode.")

from src.edge.vision import vision_engine

# ---------------------------------------------------------------------------
# System prompt — Versatile AI Brain (Windows, Math, Science, Coding, Security, Psychology, Health & All Domains)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are ED-RA, an advanced, highly knowledgeable and accurate AI Brain and edge assistant. "
    "You excel in all domains:\n"
    "1. Windows Operating System: Complete filesystem hierarchy (System32, SysWOW64, WinSxS, AppData, Program Files), "
    "core executables (ntoskrnl, smss, csrss, wininit, services, lsass, svchost, explorer), Registry hives, and Event Logs.\n"
    "2. Mathematics & Sciences: Arithmetic, algebra, calculus, physics (Newton's laws, thermodynamics, electromagnetism), "
    "and chemistry (periodic table, bonding, stoichiometry, organic chemistry).\n"
    "3. Systems Coding & Cyber Security: C/C++, Python, Rust, x86_64 assembly, OS kernels, networking (TCP/IP), "
    "ethical hacking methodology, OWASP Top 10, binary exploitation concepts, and Windows security (Kerberos, Sysmon, Event IDs).\n"
    "4. Psychology & Human Behavior: Cognitive biases (confirmation bias, loss aversion), Maslow's hierarchy of needs, "
    "emotional intelligence (EQ), stress management (box breathing, grounding), and developmental psychology.\n"
    "5. Reproductive Health & Sex Education: Scientific human anatomy, menstrual cycle physiology, contraception methods, "
    "STI prevention and screening, and ethical principles of consent (FRIES model) and healthy relationship boundaries.\n"
    "6. World History, Geography, General Knowledge, Cinema & Pop Culture, and Offline First Aid.\n"
    "Always provide precise, factual, well-explained, structured answers. If relevant offline knowledge is provided, utilize it accurately."
)


def _load_limits() -> dict:
    """Read config/limits.yaml to get declared CPU core count."""
    here = Path(__file__).resolve()
    cfg_path = here.parents[2] / "config" / "limits.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_models_cfg() -> dict:
    """Read config/models.yaml for GGUF path and parameters."""
    here = Path(__file__).resolve()
    cfg_path = here.parents[2] / "config" / "models.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# GGUFLocalLLM — Main Class
# ---------------------------------------------------------------------------

class GGUFLocalLLM:
    """
    Streaming GGUF LLM using llama-cpp-python.
    Satisfies the LLM protocol in contracts.py.
    """

    DEFAULT_MODEL_SEARCH = [
        # Check higher capability models first; first match wins
        Path("models/qwen2.5-3b-instruct-q4_k_m.gguf"),
        Path("models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        Path("models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        Path("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        Path("models/phi-2.Q4_K_M.gguf"),
    ]

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_ctx: int = 2048,
        n_batch: int = 256,
        max_tokens: int = 1024,
        temperature: float = 0.4,
        top_p: float = 0.9,
        knowledge_snippet: str = "",
    ):
        cfg = _load_models_cfg()
        llm_cfg = cfg.get("llm", {})
        self.n_ctx = int(llm_cfg.get("n_ctx", n_ctx))
        self.n_batch = int(llm_cfg.get("n_batch", n_batch))
        self.max_tokens = int(llm_cfg.get("max_tokens", max_tokens))
        self.temperature = float(llm_cfg.get("temperature", temperature))
        self.top_p = float(llm_cfg.get("top_p", top_p))
        self.knowledge_snippet = knowledge_snippet  # injected by pipeline
        self._llm: "Llama | None" = None

        # Resolve model path
        self._model_path = self._resolve_model(model_path)
        # Resolve thread count from limits.yaml
        limits = _load_limits()
        self._n_threads = int(limits.get("cores", os.cpu_count() or 4))
        logger.info("LLM will use %d CPU threads (n_ctx=%d, max_tokens=%d).", self._n_threads, self.n_ctx, self.max_tokens)

    @staticmethod
    def _is_valid_gguf(p: Path) -> bool:
        return p.exists() and not p.name.endswith((".tmp", ".part")) and p.stat().st_size >= 400 * 1024 * 1024

    def _resolve_model(self, model_path: str | Path | None) -> Path | None:
        if model_path is not None:
            p = Path(model_path)
            if self._is_valid_gguf(p):
                return p
            logger.warning("Specified model path not found or incomplete: %s", p)

        # Try models.yaml
        cfg = _load_models_cfg()
        if "llm" in cfg and "path" in cfg["llm"]:
            p = Path(cfg["llm"]["path"])
            if self._is_valid_gguf(p):
                return p

        # Search common locations relative to repo root
        here = Path(__file__).resolve()
        repo_root = here.parents[2]
        for rel in self.DEFAULT_MODEL_SEARCH:
            candidate = repo_root / rel
            if self._is_valid_gguf(candidate):
                logger.info("Auto-detected GGUF: %s", candidate)
                return candidate

        logger.warning(
            "No complete GGUF model found (needs >= 400 MB). Using fallback mode."
        )
        return None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialize the Llama model. Call once at startup."""
        if self._llm is not None:
            return

        if not _LLAMA_AVAILABLE:
            logger.error("llama_cpp not installed. Run: pip install llama-cpp-python")
            return

        if self._model_path is None:
            logger.error("No GGUF model path resolved. LLM disabled.")
            return

        logger.info("Loading GGUF model: %s", self._model_path)
        t0 = time.perf_counter()
        try:
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self._n_threads,
                n_gpu_layers=0,       # CPU-only - strict competition rule
                use_mmap=True,        # memory-mapped for fast load
                use_mlock=False,
                verbose=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("GGUF model loaded in %.0f ms.", elapsed)
        except Exception as exc:
            logger.warning("Could not load GGUF model (%s) - activating fallback mode: %s", self._model_path, exc)
            self._llm = None

    # ------------------------------------------------------------------
    # Build prompt (ChatML / Llama format)
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def _build_prompt(self, user_text: str, history: list, knowledge_snippet: str, image_path: str | None = None) -> list:
        """Construct a full ChatML-style prompt list with smart token budgeting and vision support."""
        sys_content = SYSTEM_PROMPT
        if knowledge_snippet:
            sys_content += f"\n\nRelevant offline knowledge:\n{knowledge_snippet}"

        user_content = user_text
        if image_path:
            vision_context = vision_engine.format_vision_context(image_path)
            user_content = f"[Attached Image Details]:\n{vision_context}\n\nUser Request: {user_text}"

        messages = [{"role": "system", "content": sys_content}]

        # Budget remaining tokens for history + user message
        budget = max(512, self.n_ctx - self.max_tokens - 80)
        sys_tokens = self._estimate_tokens(sys_content)
        user_tokens = self._estimate_tokens(user_content)
        rem_budget = budget - sys_tokens - user_tokens

        # Collect history turns backwards to keep newest turns within budget
        valid_history = []
        for turn in reversed(history[-6:]):
            turn_tokens = self._estimate_tokens(turn.content)
            if rem_budget >= turn_tokens:
                valid_history.insert(0, {"role": turn.role, "content": turn.content})
                rem_budget -= turn_tokens
            else:
                break

        messages.extend(valid_history)
        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    # stream() — LLM Protocol implementation
    # ------------------------------------------------------------------

    def stream(self, user_text: str, history: list, image_path: str | None = None) -> Iterator[str]:
        """
        Yields token strings one-by-one as the GGUF LLM generates them.
        Satisfies: LLM.stream(user_text, history) -> Iterator[str]
        Supports optional image_path for vision analysis.
        """
        if self._llm is None:
            self.load()

        if self._llm is None:
            # Fallback: return a stub response for testing
            yield from self._fallback_stream(user_text)
            return

        messages = self._build_prompt(user_text, history, self.knowledge_snippet, image_path=image_path)

        t0 = time.perf_counter()
        first_token = True
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            )
            for chunk in response:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    if first_token:
                        ttft = (time.perf_counter() - t0) * 1000
                        logger.debug("TTFT: %.1f ms", ttft)
                        first_token = False
                    yield token
        except Exception as exc:
            err_str = str(exc)
            if "exceed context window" in err_str.lower() or "too many tokens" in err_str.lower():
                logger.warning("Context exceeded, retrying with fresh single-turn prompt: %s", exc)
                try:
                    fresh_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text[:1200]},
                    ]
                    response = self._llm.create_chat_completion(
                        messages=fresh_messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stream=True,
                    )
                    for chunk in response:
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    return
                except Exception as inner_exc:
                    logger.error("LLM retry failed: %s", inner_exc)
            logger.error("LLM stream error: %s", exc)
            yield f"I encountered an error generating a response: {exc}"

    def _fallback_stream(self, user_text: str) -> Iterator[str]:
        """Stub response when model is unavailable (testing only)."""
        logger.warning("Using LLM fallback stub - no GGUF model loaded.")
        stub = (
            "I'm running in offline stub mode. "
            "Please place a GGUF model in the models/ directory. "
            "Consult a doctor for any medical concerns."
        )
        for word in stub.split():
            yield word + " "
            time.sleep(0.01)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = GGUFLocalLLM()
    llm.load()
    print("Streaming response:")
    t0 = time.perf_counter()
    for tok in llm.stream("How do I treat a minor burn?", []):
        print(tok, end="", flush=True)
    print(f"\n\nTotal: {(time.perf_counter()-t0)*1000:.0f} ms")
