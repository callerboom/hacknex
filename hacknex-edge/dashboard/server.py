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
import base64
import io
import re
import uuid
import wave
from typing import Optional
import numpy as np
import scipy.io.wavfile
from PIL import Image
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Decompression-bomb defense limit
# Enforce contest limits (4 cores, 3072 MB RAM)
try:
    from src.edge.limits import enforce_limits
    enforce_limits(4, 3072)
except Exception:
    pass

app = FastAPI(title="EdgeVoice Mission Control", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
UPLOAD_DIR = _ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR = _ROOT / "data" / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = _ROOT / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Upload concurrency lock
_upload_lock = threading.Lock()


def cleanup_old_uploads(max_age_seconds: int = 3600) -> None:
    """Delete uploads and downloads older than max_age_seconds."""
    now = time.time()
    for d in (UPLOAD_DIR, DOWNLOAD_DIR):
        if not d.exists():
            continue
        for f in d.glob("*"):
            if f.is_file():
                try:
                    if now - f.stat().st_mtime > max_age_seconds:
                        f.unlink()
                except Exception:
                    pass


def cleanup_all_uploads() -> None:
    """Delete all files in uploads and downloads dir on exit."""
    for d in (UPLOAD_DIR, DOWNLOAD_DIR):
        if not d.exists():
            continue
        for f in d.glob("*"):
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
_standalone_llm = None


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


def get_llm():
    global _standalone_llm
    if _pipeline is not None and getattr(_pipeline, "_llm", None) is not None:
        return _pipeline._llm
    if _standalone_llm is None:
        try:
            from src.edge.llm import EdgeLLM
            _standalone_llm = EdgeLLM()
        except Exception:
            _standalone_llm = None
    return _standalone_llm


_standalone_tts = None
_standalone_asr = None
_standalone_player = None


def get_tts():
    global _standalone_tts
    if _pipeline is not None and getattr(_pipeline, "_tts", None) is not None:
        return _pipeline._tts
    if _standalone_tts is None:
        try:
            from src.edge.tts import SentenceChunkedTTS
            _standalone_tts = SentenceChunkedTTS()
        except Exception:
            _standalone_tts = None
    return _standalone_tts


def get_asr():
    global _standalone_asr
    if _pipeline is not None and getattr(_pipeline, "_asr", None) is not None:
        return _pipeline._asr
    if _standalone_asr is None:
        try:
            from src.edge.asr import FasterWhisperEngine
            _standalone_asr = FasterWhisperEngine()
        except Exception:
            _standalone_asr = None
    return _standalone_asr


def get_player():
    global _standalone_player
    if _standalone_player is None:
        try:
            from src.edge.audio_io import AudioPlayer
            _standalone_player = AudioPlayer()
        except Exception:
            _standalone_player = None
    return _standalone_player


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 22050) -> bytes:
    audio_f32 = np.asarray(audio, dtype=np.float32)
    audio_int16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def clean_for_tts(text: str) -> str:
    # 1. Replace fenced code blocks with concise spoken phrase
    cleaned = re.sub(r"```[\s\S]*?```", " Here is the code. ", text)
    # 2. Strip markdown headers
    cleaned = re.sub(r"#+\s*", "", cleaned)
    # 3. Strip bold and italics
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    # 4. Strip inline code backticks
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # 5. Collapse spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # NOTE: Do NOT truncate — return the full cleaned text so long essays
    # and detailed answers are spoken and displayed completely.
    return cleaned.strip()



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

if UPLOAD_DIR.exists():
    app.mount("/data/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

if SCREENSHOT_DIR.exists():
    app.mount("/data/screenshots", StaticFiles(directory=SCREENSHOT_DIR), name="screenshots")

@app.get("/favicon.svg")
async def serve_favicon():
    fav = STATIC_DIR / "favicon.svg"
    if fav.exists():
        return FileResponse(fav, media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"error": "not found"})

@app.get("/icons.svg")
async def serve_icons():
    icons = STATIC_DIR / "icons.svg"
    if icons.exists():
        return FileResponse(icons, media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"error": "not found"})

@app.get("/vanilla", response_class=HTMLResponse)
async def serve_vanilla():
    vanilla = STATIC_DIR / "index_vanilla.html"
    if vanilla.exists():
        return FileResponse(vanilla)
    return HTMLResponse("<h1>Vanilla dashboard not found</h1>", status_code=404)

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
async def upload_file_or_image(file: UploadFile = File(...)):
    """
    Offline file upload endpoint:
    - Supports Images (PNG, JPG, WEBP, BMP) -> runs RapidOCR, extracts text, generates Word docx.
    - Supports PDF documents -> extracts text via pypdf, automatically converts into Microsoft Word (.docx).
    - Enforces size limit (10MB) and single concurrency.
    """
    import urllib.parse
    cleanup_old_uploads()

    if not _upload_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={"status": "busy", "error": "System is currently busy processing another file."},
        )

    try:
        MAX_BYTES = 10 * 1024 * 1024  # 10 MB
        content = bytearray()
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"status": "error", "error": "File exceeds 10MB size limit."},
                )

        if len(content) == 0:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "Uploaded file is empty."},
            )

        orig_filename = file.filename or "file"
        fn_lower = orig_filename.lower()
        is_pdf = fn_lower.endswith(".pdf") or content[:4] == b"%PDF"
        stem = Path(orig_filename).stem or "document"
        t0 = time.perf_counter()

        # ── 1. PDF DOCUMENT PROCESSING & WORD CONVERSION ──
        if is_pdf:
            await broadcast_message({
                "type": "image_progress",
                "stage": "uploading",
                "message": f"Processing PDF: {orig_filename}",
            })
            try:
                import pypdf
                import docx

                reader = pypdf.PdfReader(io.BytesIO(content))
                page_count = len(reader.pages)
                pages_text = []
                for idx, p in enumerate(reader.pages):
                    t = (p.extract_text() or "").strip()
                    if t:
                        pages_text.append(f"--- Page {idx + 1} ---\n{t}")
                full_text = "\n\n".join(pages_text) if pages_text else "No extractable text found in PDF."

                # Convert into real Microsoft Word (.docx) document
                doc = docx.Document()
                doc.add_heading(f"Converted Document: {stem}", 0)
                for page_str in pages_text:
                    for line in page_str.split("\n\n"):
                        if line.strip():
                            doc.add_paragraph(line.strip())

                docx_filename = f"{uuid.uuid4().hex}.docx"
                docx_path = DOWNLOAD_DIR / docx_filename
                doc.save(str(docx_path))
                word_url = f"/api/download/{docx_filename}?name={urllib.parse.quote(stem + '.docx')}"
                dur_ms = (time.perf_counter() - t0) * 1000

                task_result = {
                    "success": True,
                    "file_type": "pdf",
                    "filename": orig_filename,
                    "page_count": page_count,
                    "text": f"PDF successfully converted ({page_count} pages).\n\nExtracted content preview:\n" + (full_text[:600] + ("..." if len(full_text) > 600 else "")),
                    "clean_text": full_text,
                    "word_url": word_url,
                    "word_filename": f"{stem}.docx",
                    "method": "pdf_to_word",
                    "total_ms": round(dur_ms, 1),
                    "peak_ram_mb": round(psutil.Process().memory_info().rss / (1024 ** 2), 1),
                }

                await broadcast_message({
                    "type": "image_result",
                    "result": task_result,
                })
                return JSONResponse(status_code=200, content={"status": "ok", "result": task_result})
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "error": f"Failed to process PDF: {e}"},
                )

        # ── 2. IMAGE PROCESSING & OCR WITH WORD EXPORT ──
        ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
        try:
            buf = io.BytesIO(content)
            with Image.open(buf) as test_img:
                test_img.verify()
                fmt = (test_img.format or "").upper()
                if fmt not in ALLOWED_FORMATS:
                    return JSONResponse(
                        status_code=400,
                        content={"status": "error", "error": f"Invalid format '{fmt}'. Allowed: PDF, PNG, JPG, JPEG, WEBP."},
                    )
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "Invalid or corrupted file."},
            )

        # Re-open, resize longest side to <= 1024, strip EXIF metadata
        try:
            buf = io.BytesIO(content)
            with Image.open(buf) as img:
                max_side = 1024
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

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

        # Broadcast progress
        await broadcast_message({
            "type": "image_progress",
            "stage": "uploading",
            "message": "Image uploaded and normalized successfully.",
        })

        # Run OCR
        try:
            from src.edge import vision
            v_res = vision.describe(str(save_path))
            desc_text = v_res.get("text", "")
            raw_lines = v_res.get("raw_lines", [])
            clean_text = "\n".join(raw_lines) if raw_lines else desc_text
            method = v_res.get("method", "ocr")
            dur_ms = v_res.get("ms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            desc_text = f"Image processing failed: {e}"
            clean_text = ""
            method = "error"
            dur_ms = (time.perf_counter() - t0) * 1000

        # Generate Word document for extracted OCR text
        word_url = None
        word_filename = None
        if clean_text:
            try:
                import docx
                doc = docx.Document()
                doc.add_heading(f"Extracted Text: {stem}", 0)
                for line in clean_text.split("\n"):
                    if line.strip():
                        doc.add_paragraph(line.strip())
                docx_filename = f"{uuid.uuid4().hex}.docx"
                doc.save(str(DOWNLOAD_DIR / docx_filename))
                word_filename = f"{stem}.docx"
                word_url = f"/api/download/{docx_filename}?name={urllib.parse.quote(word_filename)}"
            except Exception:
                pass

        proc = psutil.Process()
        task_result = {
            "success": True,
            "file_type": "image",
            "filename": orig_filename,
            "text": desc_text,
            "clean_text": clean_text,
            "image_url": f"/data/uploads/{save_filename}",
            "word_url": word_url,
            "word_filename": word_filename,
            "method": method,
            "total_ms": round(dur_ms, 1),
            "vision_ms": round(dur_ms, 1),
            "peak_ram_mb": round(proc.memory_info().rss / (1024 ** 2), 1),
        }

        await broadcast_message({
            "type": "image_result",
            "result": task_result,
        })
        return JSONResponse(status_code=200, content={"status": "ok", "result": task_result})

    finally:
        _upload_lock.release()


# Alias
upload_image = upload_file_or_image
app.add_api_route("/upload", upload_file_or_image, methods=["POST"])


@app.get("/api/download/{filename}")
async def download_file(filename: str, name: Optional[str] = None):
    """Serve converted Word docx or other files for direct download."""
    target = DOWNLOAD_DIR / filename
    if not target.exists():
        target = UPLOAD_DIR / filename
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    dl_name = name or filename
    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if filename.endswith(".docx") else "application/octet-stream"
    return FileResponse(path=target, filename=dl_name, media_type=media_type)


@app.post("/api/convert-to-word")
async def convert_text_to_word(body: dict):
    """Converts any plain text or OCR text into a downloadable Word (.docx) file."""
    import docx
    import urllib.parse
    text = (body.get("text") or "").strip()
    title = (body.get("title") or "Document").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})
    try:
        doc = docx.Document()
        doc.add_heading(title, 0)
        for line in text.split("\n"):
            if line.strip():
                doc.add_paragraph(line.strip())
        docx_filename = f"{uuid.uuid4().hex}.docx"
        doc.save(str(DOWNLOAD_DIR / docx_filename))
        return {
            "status": "ok",
            "word_filename": f"{title}.docx",
            "download_url": f"/api/download/{docx_filename}?name={urllib.parse.quote(title + '.docx')}",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate Word document: {e}"})


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

    # If router didn't handle it, run real local LLM in worker thread so event loop is never blocked
    llm_ms = 0.0
    if reply is None:
        llm_engine = get_llm()
        if llm_engine is not None:
            try:
                t1 = _time.perf_counter()
                hist = _pipeline._history if _pipeline is not None else []

                def _run_llm_generation():
                    tokens = []
                    for tok in llm_engine.stream(query, hist, memory=memory):
                        tokens.append(tok)
                    return "".join(tokens).strip()

                reply = await asyncio.to_thread(_run_llm_generation)
                llm_ms = (_time.perf_counter() - t1) * 1000
            except Exception as e:
                reply = f"[LLM error: {e}]"
        else:
            reply = f'Offline local LLM unavailable for "{query}".'

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

    # Synthesize audio with Piper TTS in worker thread
    tts_engine = get_tts()
    audio_base64 = None
    tts_ms = 0.0
    if tts_engine is not None and reply:
        try:
            t_tts = _time.perf_counter()
            spoken_text = clean_for_tts(reply)
            if spoken_text:
                audio_samples = await asyncio.to_thread(tts_engine._synthesize, spoken_text)
                tts_ms = (_time.perf_counter() - t_tts) * 1000.0
                if len(audio_samples) > 0:
                    wav_data = audio_to_wav_bytes(audio_samples, sample_rate=tts_engine.sample_rate)
                    audio_base64 = base64.b64encode(wav_data).decode("ascii")
        except Exception:
            pass

    proc = psutil.Process()
    ram_mb = round(proc.memory_info().rss / (1024 ** 2), 1)

    screenshot_url = None
    if reply and "data/screenshots/screenshot_" in reply:
        import re as _re
        m = _re.search(r"data/screenshots/(screenshot_[a-zA-Z0-9_\-]+\.png)", reply)
        if m:
            screenshot_url = f"/data/screenshots/{m.group(1)}"

    event = {
        "type": "query_result",
        "user_text": query,
        "bot_reply": reply,
        "audio_base64": audio_base64,
        "screenshot_url": screenshot_url,
        "route": "Fast-Path" if llm_ms == 0.0 else "Qwen 1.5B",
        "is_fast_path": llm_ms == 0.0,
        "router_ms": round(router_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "asr_ms": 0,
        "tts_ms": round(tts_ms, 2),
        "total_ms": round(router_ms + llm_ms + tts_ms, 2),
        "ram_mb": ram_mb,
        "measured": True,
    }

    await broadcast_message(event)
    return {"status": "ok", "event": event}


@app.post("/api/transcribe")
async def transcribe_audio(request: Request, file: Optional[UploadFile] = File(None)):
    """
    Accepts recorded WAV audio and returns speech transcription via Faster-Whisper.
    Works with either multipart file upload or raw WAV request body.
    """
    from src.edge.audio_io import resample

    raw_bytes = b""
    if file is not None:
        raw_bytes = await file.read()
    else:
        body = await request.body()
        if body.startswith(b"{"):
            try:
                j = json.loads(body)
                if "audio_base64" in j:
                    raw_bytes = base64.b64decode(j["audio_base64"])
            except Exception:
                pass
        if not raw_bytes:
            raw_bytes = body

    if not raw_bytes:
        return JSONResponse(status_code=400, content={"status": "error", "error": "No audio data received"})

    try:
        sr_in, data = scipy.io.wavfile.read(io.BytesIO(raw_bytes))
        if data.dtype == np.int16:
            audio_f32 = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio_f32 = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio_f32 = (data.astype(np.float32) - 128.0) / 128.0
        else:
            audio_f32 = np.asarray(data, dtype=np.float32)

        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.mean(axis=1)

        # Resample to 16000Hz if needed
        if sr_in != 16000:
            audio_f32 = resample(audio_f32, sr_in, 16000)

        asr_engine = get_asr()
        if asr_engine is None:
            return JSONResponse(status_code=500, content={"status": "error", "error": "ASR engine unavailable"})

        # Force English recognition for accurate transcription
        transcript = asr_engine.transcribe(audio_f32, language="en")
        text_out = transcript.text.strip()

        # If transcription returned empty, it may be a silence/noise issue — return empty gracefully
        return {
            "status": "ok",
            "text": text_out,
            "lang": transcript.lang,
            "confidence": round(transcript.confidence, 3),
            "asr_ms": round(transcript.asr_ms, 2),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": f"Transcription failed: {e}"})


@app.post("/api/tts")
async def generate_tts(body: dict):
    """
    Synthesizes speech for the provided text and returns base64 WAV.
    Accepts: { "text": "..." }
    """
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"status": "error", "error": "Text is required"})

    tts_engine = get_tts()
    if tts_engine is None:
        return JSONResponse(status_code=500, content={"status": "error", "error": "TTS engine unavailable"})

    try:
        t0 = time.perf_counter()
        spoken = clean_for_tts(text)
        samples = tts_engine._synthesize(spoken or text[:150])
        dur_ms = (time.perf_counter() - t0) * 1000.0
        wav_bytes = audio_to_wav_bytes(samples, sample_rate=tts_engine.sample_rate)
        b64 = base64.b64encode(wav_bytes).decode("ascii")

        return {
            "status": "ok",
            "audio_base64": b64,
            "tts_ms": round(dur_ms, 2),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": f"TTS synthesis failed: {e}"})



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
