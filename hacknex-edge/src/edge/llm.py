"""llm.py — Local GGUF LLM streaming inference via llama-cpp-python (HNX26EPS08).
Integrated for HackNEX 2026.
Features:
- Loads small Q4/Q5 GGUF on CPU (Qwen2.5-0.5B / 1.5B).
- Prompt KV-cache reuse with pinned CPU threads.
- Yields token strings one by one (streaming).
- Unload and reload support for image processing.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    logger.warning("llama_cpp not available — LLM will run in fallback stub mode.")

from src.edge.contracts import Turn

SYSTEM_PROMPT = (
    "You are Jarvis Edge, a helpful, intelligent, and knowledgeable AI assistant running offline on-device. "
    "Fulfill all user requests thoroughly and with high quality. "
    "When asked for essays, explanations, test answers, or summaries, write well-structured, informative paragraphs with clear points. "
    "IMPORTANT: Always write the COMPLETE response without stopping or cutting off. "
    "If the user asks for 1000 words, write all 1000 words. Never stop mid-sentence or mid-paragraph. "
    "Always conclude your response fully and completely."
)


def _resolve_gguf_path(preferred_path: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if preferred_path:
        candidates.append(Path(preferred_path))

    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates.extend([
        repo_root / "models" / "llm" / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        repo_root / "models" / "llm" / "qwen2.5-0.5b-instruct-q5_k_m.gguf",
        repo_root / "models" / "llm" / "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        Path("models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        Path("models/llm/qwen2.5-0.5b-instruct-q5_k_m.gguf"),
        Path("models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ])

    for p in candidates:
        if p.exists() and not p.name.endswith((".tmp", ".part")) and p.stat().st_size >= 100 * 1024 * 1024:
            return p.resolve()
    return None


class GGUFLocalLLM:
    """Streaming GGUF LLM using llama-cpp-python."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int = 4,
        max_tokens: int = 1024,
        temperature: float = 0.65,
        top_p: float = 0.9,
        repeat_penalty: float = 1.15,
        knowledge_snippet: str = "",
    ):
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.knowledge_snippet = knowledge_snippet
        self._llm: Optional[Llama] = None
        self._model_path = _resolve_gguf_path(str(model_path) if model_path else None)

    @property
    def is_available(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None or not _LLAMA_AVAILABLE:
            return

        if self._model_path is None:
            self._model_path = _resolve_gguf_path()

        if self._model_path is None:
            logger.warning("No GGUF model found — fallback stub mode active.")
            return

        logger.info(f"Loading GGUF LLM from {self._model_path} with {self.n_threads} threads...")
        t0 = time.perf_counter()
        try:
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_threads=self.n_threads,
                n_gpu_layers=0,
                use_mmap=True,
                verbose=False,
            )
            load_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"GGUF model loaded in {load_ms:.0f} ms.")
        except Exception as e:
            logger.warning(f"Could not load GGUF model: {e}")
            self._llm = None

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            import gc
            gc.collect()
            logger.info("GGUF LLM unloaded to free RAM.")

    def reload(self) -> None:
        self.load()

    def _build_prompt(
        self,
        user_text: str,
        history: list,
        knowledge_snippet: str = "",
        image_path: Optional[str] = None,
        memory: Optional[Any] = None,
    ) -> list:
        sys_content = SYSTEM_PROMPT
        if knowledge_snippet or self.knowledge_snippet:
            snippet = knowledge_snippet or self.knowledge_snippet
            sys_content += f"\n\nOffline Knowledge Context:\n{snippet}"

        if memory is not None:
            try:
                recent = memory.recent(2)
                if recent:
                    mem_context = "; ".join(f"User: '{u}', Assistant: '{b}'" for u, b in recent)
                    sys_content += f"\n\nRecent Memory:\n{mem_context}"
            except Exception:
                pass

        user_content = user_text
        if image_path:
            try:
                from src.edge.vision import describe
                v_res = describe(image_path)
                user_content = f"[Attached Image Details]:\n{v_res.get('text', '')}\n\nUser Request: {user_text}"
            except Exception:
                pass

        messages = [{"role": "system", "content": sys_content}]

        for turn in history[-4:]:
            role = getattr(turn, "role", "user")
            content = getattr(turn, "content", getattr(turn, "text", ""))
            if content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_content})
        return messages

    def stream(
        self,
        user_text: str,
        history: Optional[list] = None,
        image_path: Optional[str] = None,
        memory: Optional[Any] = None,
    ) -> Iterator[str]:
        if history is None:
            history = []
        if self._llm is None:
            self.load()

        if self._llm is None:
            yield from self._fallback_stream(user_text)
            return

        messages = self._build_prompt(user_text, history, image_path=image_path, memory=memory)

        # Dynamically scale token budget based on user intent
        q_lower = user_text.lower()
        long_keywords = [
            "essay", "page", "pages", "word", "words", "detail", "detailed",
            "explain", "elaborate", "write", "story", "code", "solve", "article",
            "full", "all", "comprehensive", "analysis", "letter", "report", "describe",
            "how to", "guide", "summary", "summarize", "solution", "test", "question"
        ]
        # Check if user specified a word count (e.g. "1000 words", "500 words")
        import re as _re
        word_count_match = _re.search(r"(\d+)\s*words?", q_lower)
        if word_count_match:
            requested_words = int(word_count_match.group(1))
            # ~1.3 tokens per word on average
            effective_max_tokens = min(4000, max(512, int(requested_words * 1.5)))
        elif any(k in q_lower for k in long_keywords):
            effective_max_tokens = min(self.max_tokens, 1800)
        else:
            effective_max_tokens = min(self.max_tokens, 256)

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repeat_penalty=self.repeat_penalty,
                stream=True,
            )
            for chunk in response:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token
        except Exception as exc:
            logger.error(f"LLM generation error: {exc}")
            yield f"I experienced an error generating a response: {exc}"

    def _fallback_stream(self, user_text: str) -> Iterator[str]:
        stub = (
            "I am Jarvis Edge running offline. "
            "Your question has been processed offline without external APIs. "
            "For medical topics, always seek clinical consultation."
        )
        for word in stub.split():
            yield word + " "
            time.sleep(0.01)


# Compatibility alias
EdgeLLM = GGUFLocalLLM
