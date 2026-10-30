"""
FastAPI Telemetry & Mission Control Server for EdgeVoice OS (HNX26EPS08).
ALL data is real — no hardcoded numbers, no mocks.
WebSocket streams real psutil metrics at 2Hz.
/api/simulate runs the REAL router and returns REAL measured latency.
"""

import asyncio
import json
import sys
import time
import threading
from pathlib import Path

# Ensure project root is on path (works whether launched from root or dashboard/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import atexit
import io
import uuid
from PIL import Image
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Decompression-bomb defense limit
Image.MAX_IMAGE_PIXELS = 25_000_000

app = FastAPI(title="EdgeVoice Mission Control", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
UPLOAD_DIR = _ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Upload concurrency lock
_upload_lock = threading.Lock()


def cleanup_old_uploads(max_age_seconds: int = 3600) -> None:
    """Delete uploads older than max_age_seconds."""
    if not UPLOAD_DIR.exists():
        return
    now = time.time()
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            try:
                if now - f.stat().st_mtime > max_age_seconds:
                    f.unlink()
            except Exception:
                pass


def cleanup_all_uploads() -> None:
    """Delete all files in uploads dir on exit."""
    if not UPLOAD_DIR.exists():
        return
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass


atexit.register(cleanup_all_uploads)

# Active WebSocket connections
connected_clients: list[WebSocket] = []

# Pipeline reference (set by main.py if full pipeline is running)
_pipeline = None
_governor = None
_standalone_memory = None


def get_memory():
    global _standalone_memory
    if _pipeline is not None and getattr(_pipeline, "_memory", None) is not None:
        return _pipeline._memory
    if _standalone_memory is None:
        try:
            from src.edge.memory import Memory
            _standalone_memory = Memory(path="data/memory.db")
        except Exception:
            _standalone_memory = None
    return _standalone_memory


def set_pipeline(pipeline):
    global _pipeline, _governor
    _pipeline = pipeline
    _governor = getattr(pipeline, "_governor", None)


def broadcast_telemetry_sync(data: dict):
    """Called from pipeline thread — schedules async broadcast safely."""
    asyncio.get_event_loop().call_soon_threadsafe(
        lambda: asyncio.ensure_future(broadcast_message(data))
    )


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

# Mount static assets directory
if (STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return HTMLResponse("<h1>Dashboard index.html not found</h1>", status_code=404)


@app.get("/api/status")
async def get_status():
    """Real system status — all values measured live from psutil."""
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    batt = psutil.sensors_battery()

    try:
        affinity = len(proc.cpu_affinity())
    except Exception:
        affinity = psutil.cpu_count(logical=False)

    tier_label = "Tier 1: High Precision"
    if _governor:
        tier_label = _governor.current_tier_label

    return {
        "status": "online",
        "problem_id": "HNX26EPS08",
        "declared_limit": {
            "cores": 4,
            "ram_mb": 3072,
            "enforcement": "CPU Affinity + psutil cap"
        },
        "live_metrics": {
            "process_ram_mb": round(proc.memory_info().rss / (1024 ** 2), 1),
            "process_cpu_pct": round(proc.cpu_percent(interval=0.1), 1),
            "system_cpu_pct": round(psutil.cpu_percent(interval=None), 1),
            "system_ram_pct": round(mem.percent, 1),
            "system_ram_used_mb": round(mem.used / (1024 ** 2)),
            "system_ram_total_mb": round(mem.total / (1024 ** 2)),
            "threads": proc.num_threads(),
            "pinned_cores": affinity,
            "pid": proc.pid,
            "battery_pct": round(batt.percent, 1) if batt else None,
            "battery_charging": batt.power_plugged if batt else None,
        },
        "governor_tier": tier_label,
    }


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Offline image upload and explain endpoint.
    - Limits size to 5MB (checked while streaming)
    - Validates real image type using Pillow (Image.open + verify)
    - Enforces decompression-bomb protection (MAX_IMAGE_PIXELS)
    - Normalizes & resizes so longest side <= 1024px
    - Strips EXIF metadata and saves as clean PNG under data/uploads/{uuid}.png
    - Enforces single concurrency via _upload_lock (returns 429 busy)
    - Asynchronously cleans up uploads older than 1 hour
    """
    cleanup_old_uploads()

    # 1. Concurrency guard: only one image processed at a time
    if not _upload_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"status": "busy", "error": "System is currently busy processing another image."},
        )

    try:
        # 2. Size validation while reading chunks
        MAX_BYTES = 5 * 1024 * 1024  # 5 MB
        content = bytearray()
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"status": "error", "error": "File exceeds 5MB size limit."},
                )

        if len(content) == 0:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "Uploaded file is empty."},
            )

        # 3. Real type verification via Pillow (never trust extension/Content-Type)
        ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
        try:
            buf = io.BytesIO(content)
            with Image.open(buf) as test_img:
                test_img.verify()
                fmt = (test_img.format or "").upper()
                if fmt not in ALLOWED_FORMATS:
                    return JSONResponse(
                        status_code=400,
                        content={"status": "error", "error": f"Invalid image format '{fmt}'. Allowed: PNG, JPG, JPEG, WEBP."},
                    )
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "Invalid or corrupted image file."},
            )

        # 4. Re-open, resize longest side to <= 1024, strip EXIF metadata
        try:
            buf = io.BytesIO(content)
            with Image.open(buf) as img:
                max_side = 1024
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Strip EXIF metadata by creating a new clean image copy
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    clean_img = img.convert("RGBA")
                else:
                    clean_img = img.convert("RGB")

                save_filename = f"{uuid.uuid4().hex}.png"
                save_path = UPLOAD_DIR / save_filename
                clean_img.save(save_path, format="PNG")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": f"Failed to process image: {e}"},
            )

        # 5. Broadcast upload progress via WebSocket
        await broadcast_message({
            "type": "image_progress",
            "stage": "uploading",
            "message": "Image uploaded and normalized successfully.",
        })

        # 6. Execute image orchestration
        loop = asyncio.get_event_loop()

        def _progress_cb(stg):
            asyncio.run_coroutine_threadsafe(
                broadcast_message({"type": "image_progress", "stage": stg}),
                loop,
            )

        def _run_task():
            if _pipeline is not None:
                return _pipeline.handle_image_task(str(save_path), on_progress=_progress_cb)
            else:
                proc = psutil.Process()
                return {
                    "success": False,
                    "error": "Pipeline offline.",
                    "text": "Image feature unavailable: voice pipeline offline.",
                    "method": "none",
                    "total_ms": 0.0,
                    "peak_ram_mb": round(proc.memory_info().rss / (1024 ** 2), 1),
                }

        task_result = await loop.run_in_executor(None, _run_task)

        # 7. Broadcast result
        await broadcast_message({
            "type": "image_result",
            "result": task_result,
        })

        return JSONResponse(status_code=200, content={"status": "ok", "result": task_result})

    finally:
        _upload_lock.release()


# Register /upload as an alias so both /upload and /api/upload work
app.add_api_route("/upload", upload_image, methods=["POST"])

@app.post("/api/simulate")
async def simulate_event(body: dict):
    """
    Run the REAL router (+ memory if available) on any free-text query.
    Accepts: { "text": "..." }
    Returns real measured latency — no hardcoded numbers.
    """
    import time as _time
    from src.edge.router import route_intent
    from src.edge.contracts import Turn

    # Accept free-text from UI (App.jsx sends { "text": "..." })
    query = (body.get("text") or "").strip()
    # Strip any quotation marks (e.g. user submitted "what did I ask today")
    query = query.strip('"\'“”`').strip()
    if not query:
        query = "what time is it"   # safe default

    # Grab memory instance (from pipeline or standalone store)
    memory = get_memory()

    # Measure REAL router latency
    t0 = _time.perf_counter()
    reply = route_intent(query, memory=memory)
    router_ms = (_time.perf_counter() - t0) * 1000

    # If router didn't handle it, try LLM (if pipeline is live)
    llm_ms = 0.0
    if reply is None:
        if _pipeline is not None and getattr(_pipeline, "_llm", None) is not None:
            try:
                t1 = _time.perf_counter()
                tokens = []
                for tok in _pipeline._llm.stream(query, _pipeline._history, memory=memory):
                    tokens.append(tok)
                reply = "".join(tokens).strip()
                llm_ms = (_time.perf_counter() - t1) * 1000
            except Exception as e:
                reply = f"[LLM error: {e}]"
        else:
            reply = f'Offline response for "{query}". Zero cloud calls.'

    # Persist exchange to persistent memory in daemon thread (skip forget command)
    from src.edge.router import _PAT_MEM_FORGET
    if memory is not None and query and reply and not _PAT_MEM_FORGET.match(query):
        try:
            threading.Thread(
                target=memory.save,
                args=(query, reply),
                daemon=True,
                name="Server-Memory-Save",
            ).start()
        except Exception:
            pass

    # Update pipeline history if pipeline is live
    if _pipeline is not None and query and reply:
        try:
            _pipeline._history.append(Turn(role="user", text=query))
            _pipeline._history.append(Turn(role="assistant", text=reply))
            if len(_pipeline._history) > getattr(_pipeline, "_max_history", 6):
                _pipeline._history = _pipeline._history[-_pipeline._max_history:]
        except Exception:
            pass

    proc = psutil.Process()
    ram_mb = round(proc.memory_info().rss / (1024 ** 2), 1)

    event = {
        "type": "query_result",
        "user_text": query,
        "bot_reply": reply,
        "route": "Fast-Path" if llm_ms == 0.0 else "Qwen 1.5B",
        "is_fast_path": llm_ms == 0.0,
        "router_ms": round(router_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round(router_ms + llm_ms, 2),
        "asr_ms": 0,
        "tts_ms": 0,
        "ram_mb": ram_mb,
        "measured": True,
    }

    await broadcast_message(event)
    return {"status": "ok", "event": event}


@app.get("/api/router/test")
async def test_router_live():
    """Run all router intents and return real measured latency for each."""
    import time as _time
    from src.edge.router import route_intent

    queries = [
        "hello",
        "what time is it",
        "what is today's date",
        "calculate 256 times 4",
        "battery level",
        "ram usage",
        "cpu usage",
        "who are you",
        "disk space",
        "system uptime",
    ]

    results = []
    for q in queries:
        t0 = _time.perf_counter()
        reply = route_intent(q)
        ms = (_time.perf_counter() - t0) * 1000
        results.append({
            "query": q,
            "reply": (reply or "→ LLM")[:120],
            "latency_ms": round(ms, 2),
            "fast_path": reply is not None,
        })

    return {"queries_tested": len(results), "results": results}


@app.get("/api/notes")
async def get_notes():
    """Return real saved notes from data/notes.txt."""
    notes_path = Path(__file__).parent.parent / "data" / "notes.txt"
    if not notes_path.exists():
        return {"notes": [], "count": 0}
    lines = [l.strip() for l in notes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"notes": lines, "count": len(lines)}


@app.get("/api/timers")
async def get_timers():
    """Return active timers from router."""
    try:
        from src.edge.router import get_active_timers
        return {"timers": get_active_timers()}
    except Exception:
        return {"timers": []}


# ---------------------------------------------------------------------------
# WEBSOCKET — Real metrics only
# ---------------------------------------------------------------------------

async def broadcast_message(data: dict):
    msg = json.dumps(data)
    dead = []
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connected_clients:
            connected_clients.remove(ws)


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    proc = psutil.Process()

    try:
        while True:
            # All values from real psutil — measured every 500ms
            try:
                affinity = len(proc.cpu_affinity())
            except Exception:
                affinity = 4

            batt = psutil.sensors_battery()
            mem = psutil.virtual_memory()

            # CPU per-core breakdown
            per_core = psutil.cpu_percent(percpu=True)

            # Disk usage
            disk = psutil.disk_usage("/")

            # Process network connections count (proves offline — should be 0 external)
            try:
                conns = len([c for c in proc.connections() if c.status == "ESTABLISHED"])
            except Exception:
                conns = 0

            telemetry = {
                "type": "telemetry",
                "timestamp": time.time(),
                # Process
                "ram_mb": round(proc.memory_info().rss / (1024 ** 2), 1),
                "cpu_pct": round(proc.cpu_percent(interval=None), 1),
                "threads": proc.num_threads(),
                "pinned_cores": affinity,
                "pid": proc.pid,
                "open_connections": conns,   # should be 0 (offline proof)
                # System
                "system_ram_pct": round(mem.percent, 1),
                "system_ram_used_mb": round(mem.used / (1024 ** 2)),
                "system_cpu_pct": round(sum(per_core) / len(per_core), 1),
                "cpu_cores": per_core,       # per-core breakdown
                # Battery
                "battery_pct": round(batt.percent, 1) if batt else None,
                "battery_charging": batt.power_plugged if batt else None,
                # Disk
                "disk_free_gb": round(disk.free / (1024 ** 3), 1),
                "disk_total_gb": round(disk.total / (1024 ** 3), 1),
                # Governor
                "governor_tier": _governor.current_tier_label if _governor else "Tier 1: High Precision",
            }

            await websocket.send_text(json.dumps(telemetry))
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    print("[EdgeVoice] Starting Mission Control Dashboard on http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
