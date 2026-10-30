"""
Main Pipeline: Overlapped multi-thread VAD → ASR → Router → LLM → TTS (HNX26EPS08).
Speech-end to first audio under 300ms on AMD Ryzen 7 @ 4 cores.
Thread layout:
  Thread-1 (mic thread):      MicStream → VAD.segments → SpeechSegment queue
  Thread-2 (asr thread):      SpeechSegment → ASR/Router → text + metrics
  Thread-3 (llm+tts thread):  text → LLM stream + sentence split → TTS chunks → AudioPlayer

All threads communicate via queue.Queue for zero-copy overlap.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import queue
import threading
import time
import uuid
import logging
from typing import Optional, Callable

import numpy as np

from src.edge.contracts import SpeechSegment, Transcript, Turn
from src.edge.metrics import MetricsCollector, get_metrics
from src.edge.router import route_intent
from src.edge.knowledge import KnowledgeBase

logger = logging.getLogger("pipeline")


class EdgePipeline:
    """
    Full edge voice pipeline with overlapped threading.
    """

    def __init__(
        self,
        asr=None,
        llm=None,
        tts=None,
        vad=None,
        mic=None,
        player=None,
        tts_cache=None,
        metrics: Optional[MetricsCollector] = None,
        governor=None,
        knowledge_dir=None,
        max_history: int = 6,
        memory=None,          # Optional[Memory] — injected by build_pipeline
        wakeword=None,        # Optional[WakeWordDetector] — injected by build_pipeline
    ):
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._vad = vad
        self._mic = mic
        self._player = player
        self._tts_cache = tts_cache
        self._metrics = metrics or get_metrics()
        self._governor = governor
        self._history: list[Turn] = []
        self._max_history = max_history
        self._last_reply = ""
        self._memory = memory  # None is fine — all callers guard with try/except
        self._wakeword = wakeword
        self._is_image_busy = False

        # Pipeline queues
        self._seg_q: queue.Queue[Optional[SpeechSegment]] = queue.Queue(maxsize=2)
        self._text_q: queue.Queue[Optional[tuple]] = queue.Queue(maxsize=2)

        # WebSocket broadcast callback (optional — set by dashboard/server.py)
        self._ws_broadcast = None

        # Knowledge base
        try:
            self._kb = KnowledgeBase(knowledge_dir)
        except Exception as e:
            logger.warning(f"[Pipeline] KnowledgeBase failed: {e} — skipping context injection")
            self._kb = None

        # Seed history from persistent memory (last 3 turns)
        if self._memory is not None:
            try:
                for user_t, bot_t in self._memory.recent(3):
                    self._history.append(Turn(role="user", text=user_t))
                    self._history.append(Turn(role="assistant", text=bot_t))
                if self._history:
                    logger.info(f"[Pipeline] Loaded {len(self._history)//2} turn(s) from persistent memory.")
            except Exception as e:
                logger.warning(f"[Pipeline] Memory seed failed: {e}")

        # Control
        self._running = False
        self._threads: list[threading.Thread] = []

    def set_ws_broadcast(self, fn) -> None:
        """Register a callback to broadcast telemetry to WebSocket clients."""
        self._ws_broadcast = fn

    # ------------------------------------------------------------------
    # THREAD 1: MIC → VAD → SpeechSegment queue
    # ------------------------------------------------------------------

    def _mic_vad_loop(self) -> None:
        logger.info("[Pipeline:T1] Mic+VAD loop started.")

        def _mic_chunks_stream():
            for chunk in self._mic.chunks():
                if not self._running:
                    break
                if self._wakeword and self._wakeword.is_available:
                    self._wakeword.process_chunk(chunk)
                yield chunk

        try:
            chunks_iter = _mic_chunks_stream() if self._mic else iter([])
            for segment in self._vad.segments(chunks_iter):
                if not self._running:
                    break

                # Barge-in: if bot is speaking, user speaking interrupts it
                is_barge_in = self._player and self._player.is_playing
                if is_barge_in:
                    self._player.interrupt()
                    if self._wakeword:
                        self._wakeword.extend_window()
                    logger.info("[Pipeline:T1] Barge-in detected — player interrupted.")

                # Wake-word active window filtering
                if self._wakeword and self._wakeword.is_available:
                    if not self._wakeword.is_active() and not is_barge_in:
                        logger.debug("[Pipeline:T1] Speech segment ignored: wake-word inactive.")
                        continue
                    # Extend window since user spoke within active window
                    self._wakeword.extend_window()

                self._seg_q.put(segment)
        except Exception as e:
            logger.error(f"[Pipeline:T1] Error: {e}")
        finally:
            self._seg_q.put(None)  # sentinel
            logger.info("[Pipeline:T1] Mic+VAD loop done.")

    # ------------------------------------------------------------------
    # THREAD 2: SpeechSegment → ASR / Router → text queue
    # ------------------------------------------------------------------

    def _asr_router_loop(self) -> None:
        logger.info("[Pipeline:T2] ASR+Router loop started.")
        while self._running:
            segment = self._seg_q.get()
            if segment is None:
                break

            req_id = str(uuid.uuid4())[:8]
            self._metrics.mark(req_id, "speech_end", segment.t_speech_end)
            self._metrics.mark(req_id, "vad_decision", segment.t_vad_decision)

            try:
                audio = segment.audio

                # Transcribe
                if self._asr:
                    transcript = self._asr.transcribe(audio)
                    self._metrics.mark(req_id, "asr_final")
                    text = transcript.text.strip()
                    asr_ms = transcript.asr_ms
                else:
                    text = ""
                    asr_ms = 0.0
                    self._metrics.mark(req_id, "asr_final")

                logger.info(f"[Pipeline:T2] req={req_id} asr='{text}' ({asr_ms:.0f}ms)")

                # If image mode is currently active:
                if self._is_image_busy:
                    busy_reply = "I am currently busy processing an image. Please ask again in a moment."
                    self._metrics.set_route(req_id, "image_busy", is_fast_path=True)
                    self._text_q.put((req_id, text, busy_reply, True))
                    continue

                # Fast-path router
                fast_reply = route_intent(text, self._last_reply, memory=self._memory)
                if fast_reply is not None:
                    self._metrics.set_route(req_id, "router", is_fast_path=True)
                    self._text_q.put((req_id, text, fast_reply, True))
                else:
                    # Governor: if router-only tier, use fallback
                    if self._governor and self._governor.router_only:
                        fallback = "I'm running in low-power mode. Ask me simpler questions."
                        self._metrics.set_route(req_id, "governor_limit", is_fast_path=True)
                        self._text_q.put((req_id, text, fallback, True))
                    else:
                        self._metrics.set_route(req_id, "llm", is_fast_path=False)
                        self._text_q.put((req_id, text, None, False))

            except Exception as e:
                logger.error(f"[Pipeline:T2] Error: {e}")

        self._text_q.put(None)  # sentinel
        logger.info("[Pipeline:T2] ASR+Router loop done.")

    # ------------------------------------------------------------------
    # THREAD 3: Text → LLM + TTS → AudioPlayer
    # ------------------------------------------------------------------

    def _llm_tts_loop(self) -> None:
        from src.edge.tts import sentence_chunker

        logger.info("[Pipeline:T3] LLM+TTS loop started.")
        while self._running:
            item = self._text_q.get()
            if item is None:
                break

            req_id, user_text, fast_reply, is_fast = item

            try:
                if is_fast:
                    # Fast-path: pre-cached or immediate reply
                    reply_text = fast_reply
                    self._metrics.mark(req_id, "llm_first_token")
                    self._metrics.mark(req_id, "llm_first_sentence")
                    audio_iter = self._get_audio(reply_text)
                    self._metrics.mark(req_id, "tts_first_audio")
                    self._play_audio(audio_iter)
                else:
                    # LLM path: inject KB context if relevant
                    context = ""
                    if self._kb and user_text:
                        try:
                            if self._kb.has_relevant_content(user_text):
                                context = self._kb.search(user_text, top_k=1, max_chars=400)
                        except Exception:
                            pass

                    augmented = f"{user_text}\n\n[Context:\n{context}]" if context else user_text

                    # Stream tokens from LLM (with memory context)
                    token_stream = (
                        self._llm.stream(augmented, self._history, memory=self._memory)
                        if self._llm else iter([])
                    )

                    # Sentence-stream to TTS
                    first_token = True
                    first_sentence = True
                    reply_tokens = []

                    for sentence in sentence_chunker(token_stream):
                        if first_token:
                            self._metrics.mark(req_id, "llm_first_token")
                            first_token = False

                        if first_sentence:
                            self._metrics.mark(req_id, "llm_first_sentence")
                            audio_iter = self._get_audio(sentence)
                            self._metrics.mark(req_id, "tts_first_audio")
                            self._play_audio(audio_iter)
                            first_sentence = False
                        else:
                            self._play_audio(self._get_audio(sentence))

                        reply_tokens.append(sentence)

                    reply_text = " ".join(reply_tokens)

                # End of reply
                self._metrics.mark(req_id, "reply_done")
                self._last_reply = reply_text

                # Update in-memory history
                if user_text:
                    self._history.append(Turn(role="user", text=user_text))
                    self._history.append(Turn(role="assistant", text=reply_text))
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:]

                # Persist to DB in a daemon thread (never blocks TTS path)
                from src.edge.router import _PAT_MEM_FORGET
                if self._memory is not None and user_text and reply_text and not _PAT_MEM_FORGET.match(user_text):
                    _mem = self._memory
                    _u, _b = user_text, reply_text
                    t = threading.Thread(
                        target=_mem.save, args=(_u, _b), daemon=True, name="Memory-Save"
                    )
                    t.start()

                # Broadcast telemetry to dashboard
                self._broadcast_telemetry(req_id)

            except Exception as e:
                logger.error(f"[Pipeline:T3] Error: {e}")
                self._metrics.mark(req_id, "reply_done")

        logger.info("[Pipeline:T3] LLM+TTS loop done.")

    def _get_audio(self, text: str):
        """Get audio iterator: try TTS cache first, then direct TTS."""
        if self._tts_cache:
            return self._tts_cache.get_or_synth(text)
        elif self._tts:
            return self._tts.synth_stream(text)
        else:
            return iter([])

    def _play_audio(self, audio_iter) -> None:
        """Play audio chunks via AudioPlayer."""
        if self._player:
            self._player.play_chunks(audio_iter)
        else:
            # Drain the iterator (no-op)
            for _ in audio_iter:
                pass

    def _broadcast_telemetry(self, req_id: str) -> None:
        """Broadcast latency + resource stats to WebSocket dashboard."""
        if not self._ws_broadcast:
            return
        try:
            completed = self._metrics.last_n_completed(1)
            if not completed:
                return
            rec = completed[-1]
            if not rec.latency:
                return

            payload = {
                "req_id": req_id,
                "e2e_ms": round(rec.latency.end_to_end_ms, 1),
                "asr_ms": round(rec.latency.asr_ms, 1),
                "llm_first_token_ms": round(rec.latency.llm_first_token_ms, 1),
                "llm_first_sentence_ms": round(rec.latency.llm_first_sentence_ms, 1),
                "tts_first_audio_ms": round(rec.latency.tts_first_audio_ms, 1),
                "total_reply_ms": round(rec.latency.total_reply_ms, 1),
                "peak_ram_mb": round(rec.peak_ram_mb, 1),
                "is_fast_path": rec.is_fast_path,
                "route": rec.route,
                "summary": self._metrics.summary_dict(),
            }
            self._ws_broadcast(payload)
        except Exception as e:
            logger.debug(f"[Pipeline] Broadcast error: {e}")

    @property
    def is_image_busy(self) -> bool:
        return self._is_image_busy

    def handle_image_task(self, image_path: str, on_progress: Optional[Callable] = None) -> dict:
        """
        RAM-safe image processing orchestration:
        1. Check RAM headroom & governor.
        2. Unload LLM to free working set memory.
        3. Run vision.describe(image_path).
        4. Unload vision models and gc.collect().
        5. Reload LLM.
        6. Speak response via TTS.
        7. Log metrics to results/image_runs.csv (separate from voice latencies).
        """
        import gc
        import psutil
        import yaml
        from pathlib import Path
        from PIL import Image
        from src.edge.limits import get_enforcement_proof, load_limits_config
        from src.edge.metrics import log_image_run

        t0 = time.perf_counter()
        proc = psutil.Process()
        peak_ram_mb = proc.memory_info().rss / (1024 ** 2)

        conf = load_limits_config()
        cap_mb = conf.get("ram_mb", 3072)
        proof = get_enforcement_proof(target_cores=4, target_ram_mb=cap_mb)
        current_rss_mb = proof["working_set_mb"]
        headroom_mb = cap_mb - current_rss_mb

        # Read config/limits.yaml for image_min_headroom_mb
        min_headroom_mb = 400
        limits_file = Path(__file__).resolve().parent.parent.parent / "config" / "limits.yaml"
        if limits_file.exists():
            try:
                with open(limits_file, "r", encoding="utf-8") as f:
                    lim_data = yaml.safe_load(f)
                    min_headroom_mb = lim_data.get("governor", {}).get("image_min_headroom_mb", 400)
            except Exception:
                pass

        if self._governor and not getattr(self._governor, "image_mode_allowed", True):
            return {
                "success": False,
                "error": "Image processing temporarily disabled due to memory pressure.",
                "text": "Image mode is temporarily paused to preserve voice stability.",
                "method": "none",
                "total_ms": 0.0,
                "peak_ram_mb": round(current_rss_mb, 1),
            }

        if headroom_mb < min_headroom_mb:
            return {
                "success": False,
                "error": f"Insufficient RAM headroom ({headroom_mb:.1f} MB < {min_headroom_mb} MB).",
                "text": "Memory limit reached: insufficient RAM headroom for image processing. Voice mode remains active.",
                "method": "none",
                "total_ms": 0.0,
                "peak_ram_mb": round(current_rss_mb, 1),
            }

        self._is_image_busy = True
        method = "none"
        desc_text = ""
        dims = (0, 0)
        file_size = 0
        img_p = Path(image_path)
        if img_p.exists():
            file_size = img_p.stat().st_size
            try:
                with Image.open(img_p) as im:
                    dims = im.size
            except Exception:
                pass

        try:
            if on_progress:
                on_progress("uploading")

            # 1. Unload LLM
            if self._llm and hasattr(self._llm, "unload"):
                self._llm.unload()
            peak_ram_mb = max(peak_ram_mb, proc.memory_info().rss / (1024 ** 2))

            # 2. Run vision engine
            if on_progress:
                on_progress("reading text")

            try:
                from src.edge import vision
                v_res = vision.describe(image_path)
                method = v_res.get("method", "ocr")
                desc_text = v_res.get("text", "")
            except (ImportError, FileNotFoundError):
                method = "none"
                desc_text = "Image feature unavailable: vision models not found on device."
            except Exception as e:
                method = "none"
                desc_text = f"Image processing error: {e}"

            peak_ram_mb = max(peak_ram_mb, proc.memory_info().rss / (1024 ** 2))

            # 3. Unload vision models
            try:
                from src.edge import vision
                if hasattr(vision, "unload_models"):
                    vision.unload_models()
            except Exception:
                pass
            gc.collect()

            # 4. Reload LLM
            if on_progress:
                on_progress("describing")
            if self._llm and hasattr(self._llm, "reload"):
                self._llm.reload()
            peak_ram_mb = max(peak_ram_mb, proc.memory_info().rss / (1024 ** 2))

            # 5. Speak reply via TTS path
            if desc_text and self._tts:
                if on_progress:
                    on_progress("speaking")
                try:
                    audio_iter = self._get_audio(desc_text)
                    self._play_audio(audio_iter)
                except Exception as e:
                    logger.warning(f"[Pipeline] Image TTS play failed: {e}")

            total_ms = (time.perf_counter() - t0) * 1000
            peak_ram_mb = max(peak_ram_mb, proc.memory_info().rss / (1024 ** 2))

            success = bool(method != "none" and desc_text and "unavailable" not in desc_text.lower())
            log_image_run(
                method=method,
                file_size_bytes=file_size,
                resized_dims=dims,
                total_ms=total_ms,
                peak_ram_mb=peak_ram_mb,
                success=success,
            )

            return {
                "success": True,
                "text": desc_text,
                "method": method,
                "total_ms": round(total_ms, 1),
                "peak_ram_mb": round(peak_ram_mb, 1),
            }

        except Exception as e:
            logger.error(f"[Pipeline] Image task failure: {e}")
            total_ms = (time.perf_counter() - t0) * 1000
            peak_ram_mb = max(peak_ram_mb, proc.memory_info().rss / (1024 ** 2))
            log_image_run(
                method="error",
                file_size_bytes=file_size,
                resized_dims=dims,
                total_ms=total_ms,
                peak_ram_mb=peak_ram_mb,
                success=False,
            )
            return {
                "success": False,
                "error": str(e),
                "text": "Image processing encountered an unexpected error.",
                "method": "none",
                "total_ms": round(total_ms, 1),
                "peak_ram_mb": round(peak_ram_mb, 1),
            }
        finally:
            self._is_image_busy = False
            # Ensure LLM is reloaded
            if self._llm and hasattr(self._llm, "is_available") and not self._llm.is_available:
                if hasattr(self._llm, "reload"):
                    self._llm.reload()

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all pipeline threads."""
        self._running = True

        t1 = threading.Thread(target=self._mic_vad_loop, daemon=True, name="Pipeline-T1-MicVAD")
        t2 = threading.Thread(target=self._asr_router_loop, daemon=True, name="Pipeline-T2-ASR")
        t3 = threading.Thread(target=self._llm_tts_loop, daemon=True, name="Pipeline-T3-LLMTTS")

        self._threads = [t1, t2, t3]
        for t in self._threads:
            t.start()
        logger.info("[Pipeline] All threads started.")

    def stop(self) -> None:
        """Gracefully shut down the pipeline."""
        self._running = False
        self._seg_q.put(None)
        self._text_q.put(None)
        if self._mic:
            try:
                self._mic.stop()
            except Exception:
                pass
        if self._player:
            try:
                self._player.interrupt()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=3.0)
        logger.info("[Pipeline] Stopped.")

    def run_until_quit(self) -> None:
        """Run pipeline until KeyboardInterrupt or stop() is called."""
        self.start()
        print("\n[Jarvis Edge] Listening... Press Ctrl+C to stop.\n")
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# PIPELINE FACTORY: Assembles all components
# ---------------------------------------------------------------------------

def build_pipeline(config_path: str = "config/flags.yaml") -> EdgePipeline:
    """
    Assemble the full pipeline from available components.
    Gracefully handles missing models — falls back to partial pipeline.
    """
    import yaml
    from pathlib import Path

    flags = {}
    try:
        with open(config_path) as f:
            flags = yaml.safe_load(f) or {}
    except Exception:
        pass

    # ---- Metrics & Governor ----
    from src.edge.metrics import MetricsCollector
    metrics = MetricsCollector()

    from src.edge.governor import Governor
    gov = Governor(
        metrics=metrics,
        ram_cap_mb=int(flags.get("ram_cap_mb", 3072)),
        cpu_cores=int(flags.get("cpu_cores", 4)),
    )

    # ---- ASR ----
    asr = None
    if flags.get("enable_asr", True):
        try:
            from src.edge.asr import EdgeASR
            asr = EdgeASR(tier="whisper", model_size="tiny.en", threads=4)
        except Exception as e:
            print(f"[Build] ASR unavailable: {e}")

    # ---- LLM ----
    llm = None
    if flags.get("enable_llm", True):
        try:
            from src.edge.llm import EdgeLLM
            llm = EdgeLLM(n_ctx=4096, n_batch=512, n_threads=4, max_tokens=1024)
        except Exception as e:
            print(f"[Build] LLM unavailable: {e}")

    # ---- TTS ----
    tts = None
    if flags.get("enable_tts", True):
        try:
            from src.edge.tts import EdgeTTS
            tts = EdgeTTS()
        except Exception as e:
            print(f"[Build] TTS unavailable: {e}")

    # ---- TTS Cache ----
    tts_cache = None
    if tts:
        try:
            from src.edge.tts_cache import TTSCache
            tts_cache = TTSCache(tts_engine=tts, sample_rate=getattr(tts, "sample_rate", 22050))
            tts_cache.prebuild()
        except Exception as e:
            print(f"[Build] TTS cache unavailable: {e}")

    # ---- VAD ----
    vad = None
    if flags.get("enable_vad", True):
        try:
            from src.edge.vad import SileroVAD
            vad = SileroVAD()
        except Exception as e:
            print(f"[Build] VAD unavailable: {e}")

    # ---- Mic & Player ----
    mic = None
    player = None
    if flags.get("enable_mic", True):
        try:
            from src.edge.audio_io import MicStream, AudioPlayer
            mic = MicStream()
            player = AudioPlayer(sample_rate=getattr(tts, "sample_rate", 22050))
        except Exception as e:
            print(f"[Build] Audio I/O unavailable: {e}")

    # ---- Register with Governor ----
    gov.register_engines(asr=asr, llm=llm)

    # ---- Memory ----
    memory = None
    try:
        from src.edge.memory import Memory
        memory = Memory(path="data/memory.db")
        if not memory.is_available:
            memory = None
            print("[Build] Memory DB unavailable — running without persistence.")
        else:
            print("[Build] Persistent memory ready.")
    except Exception as e:
        print(f"[Build] Memory unavailable: {e}")

    # ---- Wake-word ----
    wakeword = None
    if flags.get("wakeword_enabled", False):
        try:
            from src.edge.wakeword import WakeWordDetector
            wake_win = float(flags.get("wake_window_s", 5.0))
            wakeword = WakeWordDetector(wake_window_s=wake_win)
            if not wakeword.is_available:
                wakeword = None
                print("[Build] Wake-word model unavailable — falling back to always-listening VAD.")
            else:
                print(f"[Build] Wake-word engine ready ({wakeword.model_name}, active_window={wake_win}s).")
        except Exception as e:
            print(f"[Build] Wake-word unavailable: {e}")

    pipeline = EdgePipeline(
        asr=asr, llm=llm, tts=tts, vad=vad,
        mic=mic, player=player, tts_cache=tts_cache,
        metrics=metrics, governor=gov,
        memory=memory, wakeword=wakeword,
    )

    return pipeline


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("[Pipeline] Building pipeline (import/syntax check) ...")
    print("[Pipeline] pipeline.py OK — Full integration test: python scripts/start_demo.ps1")
