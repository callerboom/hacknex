"""
HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)
Problem 11: Honest Latency, Resource Metrics & Telemetry Tracker
Author: Person 2 & Person 4 (Audio & Governance Leads)

Tracks:
- vad_compute_ms (pure CPU compute for VAD)
- endpointing_delay_ms (audio-time hangover + chunk delay, ~450ms)
- language_detect_ms (candidate detection latency)
- asr_ms (CTranslate2 int8 inference latency per language and model)
- translate_ms (Tier 3 English translation latency)
- analyzer_ms (intent extraction latency, <1ms)
- total_turn_ms (speech end to final action result)
- model_load_ms (cold start model loading time)
- ram_mb (RSS working set memory via psutil)
- barge_in_latency_ms (barge-in detect to stream abort)
- Logs to in-memory history and JSONL file (results/audio_metrics.jsonl)
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class TurnMetrics:
    """Detailed honest timing and resource markers for a single interaction turn."""
    turn_id: str
    language: str = "en"
    model_name: str = "whisper_tiny_multilingual"
    t_speech_start: float = 0.0
    t_speech_end: float = 0.0
    t_vad_decision: float = 0.0
    audio_duration_sec: float = 0.0

    # Individual stage latencies (all in milliseconds)
    vad_compute_ms: float = 0.0
    endpointing_delay_ms: float = 0.0   # Real audio-time hangover
    language_detect_ms: float = 0.0
    asr_ms: float = 0.0
    translate_ms: float = 0.0
    analyzer_ms: float = 0.0
    total_turn_ms: float = 0.0
    model_load_ms: float = 0.0
    barge_in_latency_ms: float = 0.0

    # Resource measurements
    ram_mb: float = 0.0

    # Text & Accuracy
    transcript: str = ""
    wer: Optional[float] = None
    cer: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.ram_mb == 0.0 and psutil is not None:
            try:
                proc = psutil.Process()
                self.ram_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                pass


class MetricsLogger:
    """
    In-memory and JSONL metrics logger for telemetry & benchmarking.
    Calculates honest percentiles (mean, p50, p95) per language and per model.
    """
    def __init__(self, log_path: Optional[str] = "results/audio_metrics.jsonl"):
        self.log_path = log_path
        self.history: list[TurnMetrics] = []
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_turn(self, metrics: TurnMetrics) -> None:
        """Records metrics for a completed turn."""
        self.history.append(metrics)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")

    def get_summary(self) -> dict[str, Any]:
        """Calculates statistical summary across all logged turns."""
        if not self.history:
            return {}

        endpoint_delays = [m.endpointing_delay_ms for m in self.history if m.endpointing_delay_ms > 0]
        vad_computes = [m.vad_compute_ms for m in self.history if m.vad_compute_ms > 0]
        asr_latencies = [m.asr_ms for m in self.history if m.asr_ms > 0.0]
        analyzer_latencies = [m.analyzer_ms for m in self.history if m.analyzer_ms > 0.0]
        total_turn_times = [m.total_turn_ms for m in self.history if m.total_turn_ms > 0.0]
        ram_values = [m.ram_mb for m in self.history if m.ram_mb > 0.0]

        summary = {
            "total_turns": len(self.history),
            "endpointing_delay_ms": {
                "mean": float(np.mean(endpoint_delays)) if endpoint_delays else 0.0,
                "p50": float(np.percentile(endpoint_delays, 50)) if endpoint_delays else 0.0,
                "p95": float(np.percentile(endpoint_delays, 95)) if endpoint_delays else 0.0,
            },
            "vad_compute_ms": {
                "mean": float(np.mean(vad_computes)) if vad_computes else 0.0,
                "p95": float(np.percentile(vad_computes, 95)) if vad_computes else 0.0,
            },
            "asr_ms": {
                "mean": float(np.mean(asr_latencies)) if asr_latencies else 0.0,
                "p50": float(np.percentile(asr_latencies, 50)) if asr_latencies else 0.0,
                "p95": float(np.percentile(asr_latencies, 95)) if asr_latencies else 0.0,
            },
            "analyzer_ms": {
                "mean": float(np.mean(analyzer_latencies)) if analyzer_latencies else 0.0,
                "p95": float(np.percentile(analyzer_latencies, 95)) if analyzer_latencies else 0.0,
            },
            "total_turn_ms": {
                "mean": float(np.mean(total_turn_times)) if total_turn_times else 0.0,
                "p95": float(np.percentile(total_turn_times, 95)) if total_turn_times else 0.0,
            },
            "ram_mb": {
                "current": ram_values[-1] if ram_values else 0.0,
                "max": float(np.max(ram_values)) if ram_values else 0.0,
            },
            "by_language": {}
        }

        # Per-language breakdown
        langs = set(m.language for m in self.history)
        for lang in langs:
            lang_turns = [m for m in self.history if m.language == lang]
            lang_asr = [m.asr_ms for m in lang_turns if m.asr_ms > 0]
            summary["by_language"][lang] = {
                "count": len(lang_turns),
                "asr_ms_mean": float(np.mean(lang_asr)) if lang_asr else 0.0,
                "asr_ms_p95": float(np.percentile(lang_asr, 95)) if lang_asr else 0.0,
            }

        return summary


# Singleton logger instance
METRICS = MetricsLogger()
