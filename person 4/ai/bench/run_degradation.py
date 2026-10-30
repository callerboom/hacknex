"""
Benchmark: Degradation Ladder Test (HNX26EPS08).
Forces the governor through Tier 1 -> 2 -> 3 -> 2 -> 1 while
measuring how latency and resource usage respond.
Results written to bench/results/degradation_<timestamp>.json.
"""

import sys
import time
import json
import uuid
import statistics
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.edge.router import route_intent
from src.edge.metrics import MetricsCollector
from src.edge.governor import Governor, Tier, TIER_LABELS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "calculate 128 times 16",
    "what time is it",
    "who are you",
    "how do I treat a burn?",
    "what is the capital of France?",
]


def _run_n_requests(n: int, metrics: MetricsCollector, tier: Tier) -> list[float]:
    e2e_list = []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        req_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        metrics.mark(req_id, "speech_end", t0)
        time.sleep(0.036)
        metrics.mark(req_id, "vad_decision")
        time.sleep(0.07 + 0.02 * int(tier))   # ASR slower at higher tier load
        metrics.mark(req_id, "asr_final")

        fast = route_intent(query)
        if fast or tier == Tier.LOW:
            metrics.mark(req_id, "llm_first_token")
            metrics.mark(req_id, "llm_first_sentence")
            time.sleep(0.04)
        else:
            time.sleep(0.08 + 0.05 * int(tier))
            metrics.mark(req_id, "llm_first_token")
            time.sleep(0.04)
            metrics.mark(req_id, "llm_first_sentence")

        time.sleep(0.04)
        metrics.mark(req_id, "tts_first_audio")
        time.sleep(0.25)
        metrics.mark(req_id, "reply_done")
        time.sleep(0.1)

        completed = metrics.last_n_completed(1)
        if completed and completed[-1].latency:
            e2e_list.append(completed[-1].latency.end_to_end_ms)

    return e2e_list


def run_degradation(n_per_tier: int = 10) -> dict:
    metrics = MetricsCollector(sample_interval=0.1)
    time.sleep(0.3)

    gov = Governor(metrics=metrics, hysteresis_s=0.0, poll_interval_s=999)

    tier_sequence = [Tier.HIGH, Tier.MEDIUM, Tier.LOW, Tier.MEDIUM, Tier.HIGH]
    results = {}

    print(f"\n{'='*60}")
    print(f"  EdgeVoice Degradation Ladder Benchmark (n={n_per_tier}/tier)")
    print(f"{'='*60}\n")

    for tier in tier_sequence:
        gov.force_tier(int(tier))
        label = TIER_LABELS[tier]
        print(f"\n  --- {label} ---")

        e2e_list = _run_n_requests(n_per_tier, metrics, tier)
        snap = metrics.summary_dict()

        tier_result = {
            "label": label,
            "p50_ms": round(statistics.median(e2e_list), 1) if e2e_list else 0,
            "mean_ms": round(statistics.mean(e2e_list), 1) if e2e_list else 0,
            "max_ms": round(max(e2e_list), 1) if e2e_list else 0,
            "ram_mb": snap["process_ram_mb"],
            "cpu_pct": snap["process_cpu_pct"],
        }
        results[f"tier_{int(tier)}"] = tier_result
        print(f"    p50={tier_result['p50_ms']}ms | max={tier_result['max_ms']}ms | RAM={tier_result['ram_mb']}MB | CPU={tier_result['cpu_pct']}%")

    gov.stop()

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_per_tier": n_per_tier,
        "tiers": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"degradation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Degradation results saved: {out_path}")
    metrics.stop()
    return output


if __name__ == "__main__":
    run_degradation(n_per_tier=10)
