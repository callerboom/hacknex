"""
Resource Governor: Graceful degradation ladder with hysteresis (HNX26EPS08).
Monitors CPU, RAM and p50 latency.
Steps down/up model tiers to stay within enforced resource limits.
"""

import time
import threading
import logging
from enum import IntEnum
from typing import Optional, Callable
import psutil

logger = logging.getLogger("governor")

# ---------------------------------------------------------------------------
# TIER DEFINITIONS
# ---------------------------------------------------------------------------

class Tier(IntEnum):
    HIGH   = 1   # Full quality: Whisper base + Llama Q5 + Piper
    MEDIUM = 2   # Compressed: Whisper tiny + Llama Q4
    LOW    = 3   # Router-only: No LLM, no heavy ASR


TIER_LABELS = {
    Tier.HIGH:   "Tier 1: High Precision (Whisper tiny.en + Qwen2.5-1.5B Q4 + Piper TTS)",
    Tier.MEDIUM: "Tier 2: Low Footprint (Whisper tiny.en + Qwen2.5-0.5B Q5)",
    Tier.LOW:    "Tier 3: Router Only (<5ms, no LLM inference)",
}

TIER_ASR = {
    Tier.HIGH:   ("whisper", "tiny.en"),
    Tier.MEDIUM: ("whisper", "tiny.en"),
    Tier.LOW:    ("whisper", "tiny.en"),
}

TIER_LLM_THREADS = {
    Tier.HIGH:   4,
    Tier.MEDIUM: 3,
    Tier.LOW:    2,
}


# ---------------------------------------------------------------------------
# GOVERNOR ENGINE
# ---------------------------------------------------------------------------

class Governor:
    """
    Resource governor that monitors system pressure and steps model tiers
    up or down with configurable hysteresis to prevent flapping.
    """

    def __init__(
        self,
        metrics=None,
        ram_cap_mb: int = 3072,
        cpu_cores: int = 4,
        # Downgrade thresholds
        ram_warn_pct: float = 85.0,
        cpu_warn_pct: float = 88.0,
        p50_warn_ms: float = 600.0,
        # Upgrade thresholds (must be lower than warn thresholds)
        ram_ok_pct: float = 70.0,
        cpu_ok_pct: float = 70.0,
        p50_ok_ms: float = 350.0,
        # Hysteresis: minimum seconds between tier changes
        hysteresis_s: float = 3.5,
        poll_interval_s: float = 1.0,
        on_tier_change: Optional[Callable] = None,
    ):
        self._metrics = metrics
        self._ram_cap_mb = ram_cap_mb
        self._cpu_cores = cpu_cores

        self._ram_warn = ram_warn_pct
        self._cpu_warn = cpu_warn_pct
        self._p50_warn = p50_warn_ms

        self._ram_ok = ram_ok_pct
        self._cpu_ok = cpu_ok_pct
        self._p50_ok = p50_ok_ms

        self._hysteresis = hysteresis_s
        self._poll = poll_interval_s
        self._on_tier_change = on_tier_change

        self._tier = Tier.HIGH
        self._last_change_t = 0.0
        self._running = True
        self._proc = psutil.Process()
        self._image_mode_allowed = True

        self._asr_ref = None   # Set by pipeline to allow hot-swap
        self._llm_ref = None

        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="Governor"
        )
        self._thread.start()
        logger.info(f"[Governor] Started. Initial tier: {TIER_LABELS[self._tier]}")

    def register_engines(self, asr=None, llm=None) -> None:
        """Register ASR and LLM engine refs so governor can hot-swap them."""
        self._asr_ref = asr
        self._llm_ref = llm

    def _get_pressure(self) -> dict:
        """Sample current resource pressure."""
        try:
            proc_ram_mb = self._proc.memory_info().rss / (1024 ** 2)
            ram_pct = (proc_ram_mb / self._ram_cap_mb) * 100
        except Exception:
            ram_pct = 0.0
            proc_ram_mb = 0.0

        try:
            cpu_pct = self._proc.cpu_percent(interval=None)
        except Exception:
            cpu_pct = 0.0

        p50 = 0.0
        if self._metrics:
            try:
                p50 = self._metrics.p50_latency_ms()
            except Exception:
                pass

        return {
            "ram_pct": ram_pct,
            "proc_ram_mb": proc_ram_mb,
            "cpu_pct": cpu_pct,
            "p50_ms": p50,
        }

    def _should_downgrade(self, pressure: dict) -> bool:
        if self._tier == Tier.LOW:
            return False
        return (
            pressure["ram_pct"] > self._ram_warn or
            pressure["cpu_pct"] > self._cpu_warn or
            (pressure["p50_ms"] > self._p50_warn and pressure["p50_ms"] > 0)
        )

    def _should_upgrade(self, pressure: dict) -> bool:
        if self._tier == Tier.HIGH:
            return False
        return (
            pressure["ram_pct"] < self._ram_ok and
            pressure["cpu_pct"] < self._cpu_ok and
            (pressure["p50_ms"] < self._p50_ok or pressure["p50_ms"] == 0)
        )

    def _apply_tier(self, new_tier: Tier, reason: str, pressure: dict) -> None:
        old_tier = self._tier
        self._tier = new_tier
        self._last_change_t = time.monotonic()

        direction = "DOWNGRADE" if new_tier > old_tier else "UPGRADE"
        msg = (
            f"[Governor] {direction}: {TIER_LABELS[old_tier]} -> {TIER_LABELS[new_tier]} | "
            f"Reason: {reason} | "
            f"RAM={pressure['proc_ram_mb']:.0f}MB ({pressure['ram_pct']:.1f}%) "
            f"CPU={pressure['cpu_pct']:.1f}% p50={pressure['p50_ms']:.0f}ms"
        )
        logger.warning(msg)
        print(msg)

        # Apply hot-swap
        if self._asr_ref:
            asr_engine, asr_size = TIER_ASR.get(new_tier, ("whisper", "tiny.en"))
            try:
                self._asr_ref.set_tier(asr_engine)
            except Exception:
                pass

        if self._llm_ref:
            n_threads = TIER_LLM_THREADS.get(new_tier, 2)
            try:
                self._llm_ref.set_threads(n_threads)
            except Exception:
                pass

        if self._on_tier_change:
            try:
                self._on_tier_change(new_tier, TIER_LABELS[new_tier], pressure)
            except Exception:
                pass

    def _monitor_loop(self) -> None:
        while self._running:
            time.sleep(self._poll)
            now = time.monotonic()
            if now - self._last_change_t < self._hysteresis:
                continue  # Hysteresis: too soon to switch again

            pressure = self._get_pressure()

            if self._should_downgrade(pressure):
                new_tier = Tier(min(int(self._tier) + 1, int(Tier.LOW)))
                reasons = []
                if pressure["ram_pct"] > self._ram_warn:
                    reasons.append(f"RAM {pressure['ram_pct']:.1f}%>{self._ram_warn}%")
                    self._image_mode_allowed = False
                if pressure["cpu_pct"] > self._cpu_warn:
                    reasons.append(f"CPU {pressure['cpu_pct']:.1f}%>{self._cpu_warn}%")
                if pressure["p50_ms"] > self._p50_warn > 0:
                    reasons.append(f"p50={pressure['p50_ms']:.0f}ms>{self._p50_warn}ms")
                self._apply_tier(new_tier, " + ".join(reasons), pressure)

            elif self._should_upgrade(pressure):
                new_tier = Tier(max(int(self._tier) - 1, int(Tier.HIGH)))
                if pressure["ram_pct"] < self._ram_ok:
                    self._image_mode_allowed = True
                self._apply_tier(new_tier, "headroom available", pressure)

    def force_tier(self, tier: int) -> None:
        """Manually force a tier (for degradation benchmarks)."""
        t = Tier(max(1, min(3, tier)))
        self._apply_tier(t, "forced by benchmark", self._get_pressure())

    def stop(self) -> None:
        self._running = False

    @property
    def current_tier(self) -> Tier:
        return self._tier

    @property
    def current_tier_label(self) -> str:
        return TIER_LABELS[self._tier]

    @property
    def router_only(self) -> bool:
        return self._tier == Tier.LOW

    @property
    def image_mode_allowed(self) -> bool:
        return self._image_mode_allowed


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO)

    gov = Governor(ram_cap_mb=3072, hysteresis_s=1.0, poll_interval_s=0.5)
    print(f"[Governor] Current tier: {gov.current_tier_label}")
    print("[Governor] Simulating forced tier changes ...")
    time.sleep(0.2)
    gov.force_tier(2)
    time.sleep(0.2)
    gov.force_tier(3)
    time.sleep(0.2)
    gov.force_tier(1)
    time.sleep(0.2)
    gov.stop()
    print("[Governor] governor.py OK")
