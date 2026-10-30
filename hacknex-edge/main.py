"""
main.py: EdgeVoice OS Entry Point (HNX26EPS08).
Starts the dashboard server, then runs the full voice pipeline.
"""

import sys
import time
import threading
import logging
import asyncio
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BANNER = """
  ███████╗██████╗  ██████╗ ███████╗    ███████╗██████╗  ██████╗ ███████╗
  ██╔════╝██╔══██╗██╔════╝ ██╔════╝    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
  █████╗  ██║  ██║██║  ███╗█████╗      █████╗  ██║  ██║██║  ███╗█████╗
  ██╔══╝  ██║  ██║██║   ██║██╔══╝      ██╔══╝  ██║  ██║██║   ██║██╔══╝
  ███████╗██████╔╝╚██████╔╝███████╗    ███████╗██████╔╝╚██████╔╝███████╗
  ╚══════╝╚═════╝  ╚═════╝ ╚══════╝    ╚══════╝╚═════╝  ╚═════╝ ╚══════╝

  HNX26EPS08 | Offline Edge AI | 100% CPU | 0 Cloud Calls
  Dashboard: http://localhost:8000
"""


def start_dashboard(pipeline=None):
    """Start FastAPI dashboard server in a background thread."""
    import uvicorn
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import dashboard.server as srv
    if pipeline:
        srv.set_pipeline(pipeline)
    config = uvicorn.Config(
        srv.app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True, name="Dashboard")
    t.start()
    return t


def main():
    print(BANNER)

    # ---- Resource enforcement ----
    try:
        from src.edge.limits import enforce_limits, load_limits_config
        _limits_conf = load_limits_config()
        _proof = enforce_limits(_limits_conf["cores"], _limits_conf["ram_mb"])
        print(
            f"[Main] Limits enforced: {_proof['status']} | "
            f"cores={_proof['effective_cores']}/{_proof['declared_cores']} | "
            f"RAM cap={_proof['declared_ram_cap_mb']}MB (win32_job={_proof['win32_job_object_active']})"
        )
    except Exception as e:
        print(f"[Main] Limits enforcement skipped: {e}")

    # ---- Build pipeline ----
    print("[Main] Building EdgeVoice pipeline ...")
    try:
        from src.edge.pipeline import build_pipeline
        pipeline = build_pipeline()
    except Exception as e:
        print(f"[Main] Pipeline build error: {e}")
        pipeline = None

    # ---- Dashboard ----
    print("[Main] Starting dashboard at http://localhost:8000 ...")
    dash_thread = start_dashboard(pipeline)
    time.sleep(1.5)  # Let dashboard start

    print("[Main] Dashboard ready.")

    if pipeline is None:
        print("[Main] Pipeline unavailable — running in dashboard-only mode.")
        print("  Run scripts/download_models.py to download required models.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    # ---- Connect telemetry broadcast to pipeline ----
    try:
        import dashboard.server as srv
        pipeline.set_ws_broadcast(srv.broadcast_telemetry_sync)
    except Exception as e:
        print(f"[Main] WebSocket telemetry hookup skipped: {e}")

    # ---- Run ----
    print("[Main] EdgeVoice is listening. Press Ctrl+C to stop.\n")
    pipeline.run_until_quit()

    print("\n[Main] Shutting down. Goodbye.")


if __name__ == "__main__":
    main()
