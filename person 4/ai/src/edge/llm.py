"""
Local LLM Inference: llama-cpp-python GGUF streaming (HNX26EPS08).
Implements the LLM protocol from contracts.py.
KV-cache reuse of fixed system prompt, hard token cap, thread-pinned CPU inference.
"""

import time
import threading
from pathlib import Path
from typing import Iterator, Optional
import yaml

from src.edge.contracts import Turn

# ---------------------------------------------------------------------------
# SYSTEM PROMPT (narrow daily-helper + first-aid domain)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Jarvis Edge, a fast, helpful, and concise AI assistant running completely offline on a laptop CPU. You have no internet access.

Your capabilities:
- Answer health and basic first-aid questions using offline knowledge.
- Answer general knowledge questions with factual, concise answers.
- Always give short, direct answers (1-3 sentences maximum).
- If you don't know something, say so honestly.

Rules:
- NEVER mention external links, websites, or internet resources.
- For medical topics: always add "Not professional medical advice — consult a doctor."
- Keep answers under 60 words unless the user explicitly asks for more detail.
- Do not start your answer with "I" — vary your sentence openers."""

# ---------------------------------------------------------------------------
# MODEL DOWNLOADER
# ---------------------------------------------------------------------------

def _find_or_suggest_model() -> Optional[Path]:
    """
    Check config/models.yaml or common locations for a GGUF model.
    Returns configured tier 1 model or first found model.
    """
    root = Path(__file__).resolve().parent.parent.parent
    cfg_file = root / "config" / "models.yaml"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            t1_path = cfg.get("llm", {}).get("tier1", {}).get("model_path")
            if t1_path:
                candidate = root / t1_path if not Path(t1_path).is_absolute() else Path(t1_path)
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    model_dir = root / "models" / "llm"
    model_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(model_dir.glob("*.gguf"))
    if candidates:
        return candidates[0]

    print(
        f"\n[LLM] No GGUF model found in {model_dir}\n"
        "  To download a recommended model, run:\n"
        "  python scripts/download_models.py\n"
        "  OR manually place a .gguf file in the models/llm/ directory.\n"
        "  Suggested: Qwen2.5-0.5B-Instruct-Q5_K_M.gguf (~350 MB)\n"
    )
    return None


# ---------------------------------------------------------------------------
# LLM ENGINE
# ---------------------------------------------------------------------------

class EdgeLLM:
    """
    llama-cpp-python GGUF inference with:
    - KV-cache reuse of the fixed system prompt
    - Token streaming via generator
    - Hard max-token cap for low-latency replies
    - Thread count pinned to governor-declared limit
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        n_ctx: int = 512,
        n_batch: int = 128,
        n_threads: int = 4,
        max_tokens: int = 120,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        from llama_cpp import Llama

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._lock = threading.Lock()

        path = Path(model_path) if model_path else _find_or_suggest_model()
        if path is None or not path.exists():
            self._model = None
            self._available = False
            return

        self._path = path
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._n_threads = n_threads

        print(f"[LLM] Loading {path.name} (n_ctx={n_ctx}, n_batch={n_batch}, threads={n_threads}) ...")
        t0 = time.perf_counter()

        self._model = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=0,          # CPU-only — no GPU
            use_mmap=True,           # memory-mapped for fast loading
            use_mlock=False,
            verbose=False,
        )
        self._available = True
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[LLM] Model loaded in {load_ms:.0f}ms")

        # Pre-warm KV cache with system prompt
        self._prime_system_prompt()

    def unload(self) -> None:
        """Unload LLM from memory to maximize headroom for vision/OCR."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._available = False
                import gc
                gc.collect()

    def reload(self) -> None:
        """Reload LLM from disk into memory."""
        with self._lock:
            if self._model is None and self._path and self._path.exists():
                from llama_cpp import Llama
                self._model = Llama(
                    model_path=str(self._path),
                    n_ctx=self._n_ctx,
                    n_batch=self._n_batch,
                    n_threads=self._n_threads,
                    n_gpu_layers=0,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
                self._available = True
                self._prime_system_prompt()

    def _prime_system_prompt(self) -> None:
        """Run a quick forward pass to cache the system prompt tokens."""
        try:
            prompt = self._format_prompt("hello", [])
            list(self._model(
                prompt,
                max_tokens=1,
                stream=False,
                temperature=0.0,
            ))
            print("[LLM] System prompt KV-cache warmed.")
        except Exception as e:
            print(f"[LLM] KV-cache warm-up skipped: {e}")

    @staticmethod
    def _format_prompt(
        user_text: str,
        history: list[Turn],
        memory_context: str = "",
    ) -> str:
        """Format conversation as a ChatML-compatible prompt."""
        parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"]
        # Inject memory snippet (≤~100 tokens) as a non-cached system note
        if memory_context:
            parts.append(f"<|im_start|>system\nEarlier you discussed: {memory_context}<|im_end|>")
        for turn in history[-4:]:   # keep last 2 exchanges for context
            role = "user" if turn.role == "user" else "assistant"
            parts.append(f"<|im_start|>{role}\n{turn.text}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def stream(
        self,
        user_text: str,
        history: list[Turn],
        memory=None,          # Optional[Memory] — looked up here, not imported at top
    ) -> "Iterator[str]":
        """
        Implement LLM protocol: yields token strings one by one.
        Thread-safe via lock (only one inference at a time).
        memory: if provided, search(user_text, k=2) results are prepended as context.
        """
        if not self._available or self._model is None:
            yield "[LLM not loaded — place a GGUF model in models/llm/ and restart]"
            return

        if not user_text.strip():
            return

        # Build memory context snippet (≤ ~100 tokens)
        memory_context = ""
        if memory is not None:
            try:
                hits = memory.search(user_text, k=2)
                if hits:
                    snippets = [f"Q: {u} A: {b}" for u, b in hits]
                    raw = " | ".join(snippets)
                    # Hard-cap at 400 chars (~100 tokens) to not disturb KV cache
                    memory_context = raw[:400]
            except Exception:
                pass

        prompt = self._format_prompt(user_text, history, memory_context)

        with self._lock:
            try:
                for chunk in self._model(
                    prompt,
                    max_tokens=self._max_tokens,
                    stream=True,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    stop=["<|im_end|>", "<|im_start|>", "\n\n\n"],
                    repeat_penalty=1.1,
                ):
                    token = chunk["choices"][0]["text"]
                    if token:
                        yield token
            except Exception as e:
                yield f"[LLM error: {e}]"

    def set_threads(self, n: int) -> None:
        """Hot-update thread count (called by governor)."""
        if self._model and hasattr(self._model, "_llama"):
            try:
                self._model._llama.n_threads = n
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        return self._available


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    llm = EdgeLLM(n_ctx=512, n_batch=128, n_threads=4, max_tokens=80)

    if not llm.is_available:
        print("[LLM] No model loaded. Run scripts/download_models.py first.")
        sys.exit(0)

    test_queries = [
        ("How do I treat a minor burn?", []),
        ("What is the capital of France?", []),
    ]

    print("\n=== LLM STREAMING TEST ===\n")
    for query, hist in test_queries:
        print(f"Q: {query}")
        print("A: ", end="", flush=True)
        t0 = time.perf_counter()
        first_token = True
        full = ""
        for token in llm.stream(query, hist):
            if first_token:
                ttft_ms = (time.perf_counter() - t0) * 1000
                print(f"[TTFT:{ttft_ms:.0f}ms] ", end="", flush=True)
                first_token = False
            print(token, end="", flush=True)
            full += token
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"\n  Total: {total_ms:.0f}ms\n")
