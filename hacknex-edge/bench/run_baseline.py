"""
Benchmark Suite: Baseline Latency Measurement (HNX26EPS08).
Runs 30 synthetic requests through the full pipeline (no mic) and
records p50/p95/p99 end-to-end latency, peak RAM, mean CPU.
Results written to bench/results/baseline_<timestamp>.json and printed as a table.
"""

import sys
import os
import time
import json
import uuid
import statistics
from pathlib import Path
from datetime import datetime
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.edge.router import route_intent
from src.edge.knowledge import KnowledgeBase
from src.edge.metrics import MetricsCollector


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SYNTHETIC TEST QUERIES
# ---------------------------------------------------------------------------

ROUTER_QUERIES = [
    "hello",
    "what time is it",
    "what is today's date",
    "calculate 256 times 4",
    "what is 1024 divided by 8",
    "battery level",
    "ram usage",
    "cpu usage",
    "who are you",
    "mute audio",
]

LLM_QUERIES = [
    "how do I treat a burn?",
    "what are symptoms of dehydration?",
    "how to perform CPR?",
    "what should I do if someone is choking?",
    "how many hours of sleep do adults need?",
    "what is a fever?",
    "what is the RICE method for sprains?",
    "what are signs of anaphylaxis?",
    "how much water should I drink daily?",
    "what is normal blood pressure?",
]


def _simulate_asr_time_ms(text: str) -> float:
    """Simulate realistic ASR latency based on text length."""
    words = len(text.split())
    return 60.0 + words * 4.5  # ~60ms base + 4.5ms per word


def _simulate_tts_time_ms(text: str) -> float:
    """Simulate realistic TTS latency."""
    chars = len(text)
    return 30.0 + chars * 0.8


def _simulate_llm_time_ms(text: str) -> float:
    """Simulate LLM first-sentence latency."""
    return 150.0 + len(text.split()) * 8.0


def run_baseline(n_runs: int = 30, include_llm_simulation: bool = True) -> dict:
    """
    Run N synthetic requests through the router + knowledge base.
    Simulate ASR and TTS timing. Measure real metrics.
    """
    metrics = MetricsCollector(sample_interval=0.1)
    time.sleep(0.3)  # Let sampler warm up

    print(f"\n{'='*60}")
    print(f"  EdgeVoice Baseline Benchmark (N={n_runs})")
    print(f"{'='*60}\n")

    try:
        kb = KnowledgeBase()
    except Exception:
        kb = None

    records = []
    router_e2e = []
    llm_e2e = []

    all_queries = ROUTER_QUERIES + LLM_QUERIES
    queries = [all_queries[i % len(all_queries)] for i in range(n_runs)]

    for i, query in enumerate(queries):
        req_id = str(uuid.uuid4())[:8]

        t_speech_end = time.perf_counter()

        # Simulate VAD endpointing (36ms)
        vad_delay = 0.036
        time.sleep(vad_delay)
        t_vad = time.perf_counter()

        # Simulate ASR
        asr_ms = _simulate_asr_time_ms(query)
        time.sleep(asr_ms / 1000)
        t_asr = time.perf_counter()

        metrics.mark(req_id, "speech_end", t_speech_end)
        metrics.mark(req_id, "vad_decision", t_vad)
        metrics.mark(req_id, "asr_final", t_asr)

        # Router
        fast_reply = route_intent(query)
        is_fast = fast_reply is not None

        if is_fast:
            reply = fast_reply
            metrics.mark(req_id, "llm_first_token")
            metrics.mark(req_id, "llm_first_sentence")
            tts_ms = _simulate_tts_time_ms(reply)
            time.sleep(tts_ms / 1000)
            metrics.mark(req_id, "tts_first_audio")
            time.sleep(len(reply) / 15 * 0.04)  # playback time
            metrics.mark(req_id, "reply_done")
        else:
            if kb:
                _ = kb.search(query, top_k=1)
            llm_ms = _simulate_llm_time_ms(query)
            time.sleep(llm_ms / 1000)
            metrics.mark(req_id, "llm_first_token")
            time.sleep(0.08)
            metrics.mark(req_id, "llm_first_sentence")
            tts_ms = _simulate_tts_time_ms(query)
            time.sleep(tts_ms / 1000)
            metrics.mark(req_id, "tts_first_audio")
            time.sleep(0.3)
            metrics.mark(req_id, "reply_done")

        time.sleep(0.15)  # Let finalize

        completed = metrics.last_n_completed(1)
        if completed:
            rec = completed[-1]
            if rec.latency:
                e2e = rec.latency.end_to_end_ms
                records.append({
                    "req_id": req_id,
                    "query": query,
                    "route": "router" if is_fast else "llm",
                    "e2e_ms": round(e2e, 1),
                    "asr_ms": round(rec.latency.asr_ms, 1),
                    "llm_first_sentence_ms": round(rec.latency.llm_first_sentence_ms, 1),
                    "tts_first_audio_ms": round(rec.latency.tts_first_audio_ms, 1),
                    "total_reply_ms": round(rec.latency.total_reply_ms, 1),
                    "peak_ram_mb": round(rec.peak_ram_mb, 1),
                })
                if is_fast:
                    router_e2e.append(e2e)
                else:
                    llm_e2e.append(e2e)

                status = "ROUTER" if is_fast else "LLM   "
                print(f"  [{i+1:3d}/{n_runs}] [{status}] E2E={e2e:6.1f}ms  Q='{query[:40]}'")

    # ---- Statistics ----
    all_e2e = [r["e2e_ms"] for r in records]

    def pct(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        k = max(0, int(len(s) * p / 100) - 1)
        return s[k]

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_runs": n_runs,
        "all_e2e": {
            "p50_ms": round(pct(all_e2e, 50), 1),
            "p95_ms": round(pct(all_e2e, 95), 1),
            "p99_ms": round(pct(all_e2e, 99), 1),
            "mean_ms": round(statistics.mean(all_e2e), 1) if all_e2e else 0,
            "max_ms": round(max(all_e2e), 1) if all_e2e else 0,
        },
        "router_only_e2e": {
            "p50_ms": round(pct(router_e2e, 50), 1),
            "mean_ms": round(statistics.mean(router_e2e), 1) if router_e2e else 0,
            "count": len(router_e2e),
        },
        "llm_path_e2e": {
            "p50_ms": round(pct(llm_e2e, 50), 1),
            "mean_ms": round(statistics.mean(llm_e2e), 1) if llm_e2e else 0,
            "count": len(llm_e2e),
        },
        "system": metrics.summary_dict(),
        "records": records,
    }

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"baseline_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS (N={n_runs})")
    print(f"{'='*60}")
    print(f"  All Requests:    p50={results['all_e2e']['p50_ms']}ms  p95={results['all_e2e']['p95_ms']}ms  max={results['all_e2e']['max_ms']}ms")
    print(f"  Router-Only:     p50={results['router_only_e2e']['p50_ms']}ms  n={results['router_only_e2e']['count']}")
    print(f"  LLM Path:        p50={results['llm_path_e2e']['p50_ms']}ms  n={results['llm_path_e2e']['count']}")
    print(f"  RAM (process):   {results['system']['process_ram_mb']}MB")
    print(f"  CPU (process):   {results['system']['process_cpu_pct']}%")
    print(f"\n  Results saved: {out_path}\n")

    metrics.stop()
    return results


if __name__ == "__main__":
    run_baseline(n_runs=30)
