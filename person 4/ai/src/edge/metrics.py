"""
Metrics & Latency Telemetry for EdgeVoice OS (HNX26EPS08).
Implements the Metrics protocol from contracts.py.
Records all required event timestamps and computes stage-by-stage latency breakdown.
Samples full process tree CPU & RAM via psutil in a background thread.
"""

import time
import threading
import os
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
import psutil

# Required event names (from contracts.py spec)
EVENTS = [
    "speech_end",
    "vad_decision",
    "asr_final",
    "llm_first_token",
    "llm_first_sentence",
    "tts_first_audio",
    "reply_done",
]


@dataclass
class StageLatency:
    endpointing_ms: float = 0.0    # vad_decision - speech_end
    asr_ms: float = 0.0            # asr_final - vad_decision
    llm_first_token_ms: float = 0.0  # llm_first_token - asr_final
    llm_first_sentence_ms: float = 0.0  # llm_first_sentence - asr_final
    tts_first_audio_ms: float = 0.0  # tts_first_audio - llm_first_sentence
    end_to_end_ms: float = 0.0     # tts_first_audio - speech_end (COMPETITION METRIC)
    total_reply_ms: float = 0.0    # reply_done - speech_end


@dataclass
class ResourceSnapshot:
    timestamp: float = 0.0
    process_ram_mb: float = 0.0
    process_cpu_pct: float = 0.0
    system_ram_pct: float = 0.0
    system_cpu_pct: float = 0.0
    num_threads: int = 0


@dataclass
class RequestRecord:
    req_id: str
    events: dict = field(default_factory=dict)
    latency: Optional[StageLatency] = None
    peak_ram_mb: float = 0.0
    mean_cpu_pct: float = 0.0
    route: str = "unknown"   # "router" or "llm"
    is_fast_path: bool = False


class MetricsCollector:
    """
    Thread-safe metrics collector implementing the Metrics protocol.
    Records event timestamps per request, computes latencies, and
    samples process resource usage from a background thread.
    """

    def __init__(self, sample_interval: float = 0.25):
        self._lock = threading.Lock()
        self._records: dict[str, RequestRecord] = {}
        self._completed: deque[RequestRecord] = deque(maxlen=100)
        self._resource_samples: deque[ResourceSnapshot] = deque(maxlen=500)
        self._callbacks: list[Callable] = []

        self._proc = psutil.Process()
        self._sample_interval = sample_interval
        self._sampling = True
        self._sampler_thread = threading.Thread(
            target=self._resource_sampler, daemon=True, name="MetricsSampler"
        )
        self._sampler_thread.start()

    # ------------------------------------------------------------------
    # PUBLIC: METRICS PROTOCOL
    # ------------------------------------------------------------------

    def mark(self, req_id: str, event: str, t: Optional[float] = None) -> None:
        """
        Record a named event timestamp for a request.

        Args:
            req_id: Unique request ID string.
            event:  One of the required event names (see EVENTS).
            t:      Optional perf_counter time; uses time.perf_counter() if None.
        """
        ts = t if t is not None else time.perf_counter()
        with self._lock:
            if req_id not in self._records:
                self._records[req_id] = RequestRecord(req_id=req_id)
            self._records[req_id].events[event] = ts

        # Auto-finalize when reply_done received
        if event == "reply_done":
            self._finalize(req_id)

    def set_route(self, req_id: str, route: str, is_fast_path: bool = False) -> None:
        with self._lock:
            if req_id not in self._records:
                self._records[req_id] = RequestRecord(req_id=req_id)
            self._records[req_id].route = route
            self._records[req_id].is_fast_path = is_fast_path

    def on_request_complete(self, callback: Callable[[RequestRecord], None]) -> None:
        """Register a callback that fires when each request is finalized."""
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # LATENCY COMPUTATION
    # ------------------------------------------------------------------

    def _finalize(self, req_id: str) -> None:
        with self._lock:
            rec = self._records.get(req_id)
            if not rec:
                return
            ev = rec.events
            lat = StageLatency()

            def diff_ms(a: str, b: str) -> float:
                if a in ev and b in ev:
                    return max(0.0, (ev[b] - ev[a]) * 1000)
                return 0.0

            lat.endpointing_ms = diff_ms("speech_end", "vad_decision")
            lat.asr_ms = diff_ms("vad_decision", "asr_final")
            lat.llm_first_token_ms = diff_ms("asr_final", "llm_first_token")
            lat.llm_first_sentence_ms = diff_ms("asr_final", "llm_first_sentence")
            lat.tts_first_audio_ms = diff_ms("llm_first_sentence", "tts_first_audio")
            lat.end_to_end_ms = diff_ms("speech_end", "tts_first_audio")
            lat.total_reply_ms = diff_ms("speech_end", "reply_done")
            rec.latency = lat

            # Capture resource samples collected since speech_end
            speech_t = ev.get("speech_end", 0.0)
            reply_t = ev.get("reply_done", time.perf_counter())
            relevant_samples = [
                s for s in self._resource_samples
                if speech_t <= s.timestamp <= reply_t
            ]
            if relevant_samples:
                rec.peak_ram_mb = max(s.process_ram_mb for s in relevant_samples)
                rec.mean_cpu_pct = sum(s.process_cpu_pct for s in relevant_samples) / len(relevant_samples)

            self._completed.append(rec)
            del self._records[req_id]

        for cb in self._callbacks:
            try:
                cb(rec)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # RESOURCE SAMPLER (background thread)
    # ------------------------------------------------------------------

    def _resource_sampler(self) -> None:
        while self._sampling:
            try:
                snap = ResourceSnapshot(
                    timestamp=time.perf_counter(),
                    process_ram_mb=self._proc.memory_info().rss / (1024 ** 2),
                    process_cpu_pct=self._proc.cpu_percent(interval=None),
                    system_ram_pct=psutil.virtual_memory().percent,
                    system_cpu_pct=psutil.cpu_percent(interval=None),
                    num_threads=self._proc.num_threads(),
                )
                with self._lock:
                    self._resource_samples.append(snap)
            except Exception:
                pass
            time.sleep(self._sample_interval)

    def stop(self) -> None:
        self._sampling = False

    # ------------------------------------------------------------------
    # ACCESSORS
    # ------------------------------------------------------------------

    def latest_snapshot(self) -> Optional[ResourceSnapshot]:
        with self._lock:
            return self._resource_samples[-1] if self._resource_samples else None

    def last_n_completed(self, n: int = 10) -> list[RequestRecord]:
        with self._lock:
            return list(self._completed)[-n:]

    def p50_latency_ms(self, n: int = 20) -> float:
        records = self.last_n_completed(n)
        lats = sorted(r.latency.end_to_end_ms for r in records if r.latency and r.latency.end_to_end_ms > 0)
        if not lats:
            return 0.0
        return lats[len(lats) // 2]

    def summary_dict(self) -> dict:
        snap = self.latest_snapshot()
        return {
            "process_ram_mb": round(snap.process_ram_mb, 1) if snap else 0,
            "process_cpu_pct": round(snap.process_cpu_pct, 1) if snap else 0,
            "system_ram_pct": round(snap.system_ram_pct, 1) if snap else 0,
            "p50_e2e_latency_ms": round(self.p50_latency_ms(), 1),
            "completed_requests": len(self._completed),
        }


# Singleton default instance for pipeline use
_default_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def log_image_run(
    method: str,
    file_size_bytes: int,
    resized_dims: tuple[int, int],
    total_ms: float,
    peak_ram_mb: float,
    success: bool,
) -> None:
    """
    Log an image run to results/image_runs.csv.
    Separate from voice end-to-end latency metrics.
    """
    from pathlib import Path
    from datetime import datetime
    import logging

    root = Path(__file__).resolve().parent.parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / "image_runs.csv"

    header = "timestamp,method,file_size_bytes,resized_dims,total_ms,peak_ram_mb,success\n"
    write_header = not csv_file.exists() or csv_file.stat().st_size == 0

    ts = datetime.now().isoformat(timespec="seconds")
    dims_str = f"{resized_dims[0]}x{resized_dims[1]}" if resized_dims else "unknown"
    row = f"{ts},{method},{file_size_bytes},{dims_str},{total_ms:.2f},{peak_ram_mb:.2f},{str(success).lower()}\n"

    try:
        with open(csv_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            f.write(row)
    except Exception as e:
        logging.getLogger("metrics").warning(f"Failed to log image run: {e}")


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    m = MetricsCollector(sample_interval=0.1)
    print("[Metrics] Collector started. Running mock request simulation...\n")

    def on_done(rec: RequestRecord):
        lat = rec.latency
        print(f"  Request '{rec.req_id}' completed:")
        print(f"    End-to-End (competition metric): {lat.end_to_end_ms:.1f} ms")
        print(f"    Endpointing delay:  {lat.endpointing_ms:.1f} ms")
        print(f"    ASR:                {lat.asr_ms:.1f} ms")
        print(f"    LLM first sentence: {lat.llm_first_sentence_ms:.1f} ms")
        print(f"    TTS first audio:    {lat.tts_first_audio_ms:.1f} ms")
        print(f"    Total reply:        {lat.total_reply_ms:.1f} ms")
        print(f"    Peak RAM:           {rec.peak_ram_mb:.1f} MB")
        print(f"    Mean CPU:           {rec.mean_cpu_pct:.1f}%")
        print()

    m.on_request_complete(on_done)

    # Simulate a realistic request timeline
    import uuid
    req = str(uuid.uuid4())[:8]

    t0 = time.perf_counter()
    m.mark(req, "speech_end", t0)
    time.sleep(0.036)
    m.mark(req, "vad_decision")
    time.sleep(0.082)
    m.mark(req, "asr_final")
    time.sleep(0.018)
    m.mark(req, "llm_first_token")
    time.sleep(0.058)
    m.mark(req, "llm_first_sentence")
    time.sleep(0.042)
    m.mark(req, "tts_first_audio")
    time.sleep(0.310)
    m.mark(req, "reply_done")

    time.sleep(0.2)  # let callback fire

    print("Summary:", m.summary_dict())
    m.stop()
