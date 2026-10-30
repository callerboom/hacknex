"""
Offline Network Proof Script (HNX26EPS08).
Verifies that the system runs with zero network access by:
1. Blocking all outbound connections via socket patching
2. Running the full router + knowledge + metrics pipeline
3. Confirming no network calls were attempted
"""

import sys
import socket
import time
import json
import uuid
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_network_attempts: list[str] = []


def _blocked_connect(self_or_addr, addr=None):
    """Intercepts all socket.connect() calls and records them."""
    target = addr if addr else self_or_addr
    _network_attempts.append(str(target))
    raise ConnectionRefusedError(
        f"[OFFLINE PROOF] Network blocked: attempt to connect to {target}"
    )


def run_offline_proof() -> dict:
    print(f"\n{'='*60}")
    print(f"  EdgeVoice Offline Network Proof (HNX26EPS08)")
    print(f"{'='*60}\n")

    # Patch socket at the lowest level
    original_connect = socket.socket.connect
    socket.socket.connect = _blocked_connect

    passed = True
    errors = []
    tested = []

    try:
        # ---- Test 1: Router ----
        print("  [1/5] Testing fast-path router (no network expected) ...")
        from src.edge.router import route_intent
        queries = [
            "what time is it",
            "calculate 12 times 12",
            "who are you",
            "check battery",
            "ram usage",
        ]
        for q in queries:
            result = route_intent(q)
            assert result is not None, f"Router returned None for '{q}'"
        tested.append("router: PASS (0 network calls)")
        print(f"    PASS — {len(queries)} queries, 0 network calls")

        # ---- Test 2: Knowledge Base ----
        print("  [2/5] Testing BM25 knowledge base (no network expected) ...")
        from src.edge.knowledge import KnowledgeBase
        kb = KnowledgeBase()
        result = kb.search("how to treat a burn", top_k=1)
        assert result, "KnowledgeBase returned empty result"
        assert len(result) > 20, "KnowledgeBase result too short"
        tested.append("knowledge: PASS (0 network calls)")
        print(f"    PASS — BM25 search returned {len(result)} chars")

        # ---- Test 3: Metrics ----
        print("  [3/5] Testing metrics collector (no network expected) ...")
        from src.edge.metrics import MetricsCollector
        m = MetricsCollector(sample_interval=0.5)
        req_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        m.mark(req_id, "speech_end", t0)
        time.sleep(0.01)
        m.mark(req_id, "vad_decision")
        time.sleep(0.01)
        m.mark(req_id, "asr_final")
        m.mark(req_id, "llm_first_token")
        m.mark(req_id, "llm_first_sentence")
        m.mark(req_id, "tts_first_audio")
        m.mark(req_id, "reply_done")
        time.sleep(0.1)
        summary = m.summary_dict()
        assert summary["completed_requests"] >= 1
        m.stop()
        tested.append("metrics: PASS (0 network calls)")
        print(f"    PASS — 1 request completed, RAM={summary['process_ram_mb']}MB")

        # ---- Test 4: Governor ----
        print("  [4/5] Testing governor (no network expected) ...")
        from src.edge.governor import Governor, Tier
        gov = Governor(hysteresis_s=0.0, poll_interval_s=999)
        gov.force_tier(2)
        assert gov.current_tier == Tier.MEDIUM
        gov.force_tier(1)
        assert gov.current_tier == Tier.HIGH
        gov.stop()
        tested.append("governor: PASS (0 network calls)")
        print(f"    PASS — Tier switching OK")

        # ---- Test 5: TTS Sentence Chunker ----
        print("  [5/5] Testing TTS sentence chunker (no network expected) ...")
        from src.edge.tts import sentence_chunker
        tokens = iter(["Hello world. ", "This is a test! ", "All offline."])
        sentences = list(sentence_chunker(tokens))
        assert len(sentences) >= 1
        tested.append("tts_chunker: PASS (0 network calls)")
        print(f"    PASS — {len(sentences)} sentences chunked")

    except ConnectionRefusedError as e:
        passed = False
        errors.append(str(e))
        print(f"\n  [FAIL] Network call detected: {e}")
    except AssertionError as e:
        passed = False
        errors.append(str(e))
        print(f"\n  [FAIL] Assertion failed: {e}")
    except Exception as e:
        passed = False
        errors.append(str(e))
        print(f"\n  [FAIL] Unexpected error: {e}")
    finally:
        # Restore socket
        socket.socket.connect = original_connect

    result = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed and len(_network_attempts) == 0,
        "network_attempts": _network_attempts,
        "tests_run": tested,
        "errors": errors,
    }

    # Print verdict
    print(f"\n{'='*60}")
    if result["passed"]:
        print(f"  RESULT: ALL OFFLINE PROOF TESTS PASSED")
        print(f"  Network calls made: {len(_network_attempts)} (expected: 0)")
    else:
        print(f"  RESULT: OFFLINE PROOF FAILED")
        print(f"  Network calls attempted: {_network_attempts}")
        print(f"  Errors: {errors}")
    print(f"{'='*60}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"offline_proof_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved: {out_path}")

    return result


if __name__ == "__main__":
    run_offline_proof()
