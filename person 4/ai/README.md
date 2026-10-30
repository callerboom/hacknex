# EdgeVoice OS — HNX26EPS08
### 100% Offline Edge AI Voice Assistant | AMD Ryzen 7 | 4 Cores | 3072 MB RAM

---

## Architecture

```
Microphone → [VAD] → [ASR] → [Router] → [LLM] → [TTS] → Speaker
               ↓         ↓        ↓          ↓       ↓
             Silero   Whisper   BM25+KB   Llama  Piper
             ONNX     tiny.en   Fast-path  GGUF   ONNX
```

**Full overlapped 3-thread pipeline:**
- Thread 1 (Mic+VAD): Continuous 16kHz capture → Silero VAD segments
- Thread 2 (ASR+Router): Whisper int8 → Fast router OR LLM dispatch
- Thread 3 (LLM+TTS): Sentence-streamed Llama → Piper synthesis → Speaker

---

## Module Map

| File | Owner | Description |
|---|---|---|
| `src/edge/contracts.py` | All | Shared protocols & data types |
| `src/edge/router.py` | Person 1 | Fast-path intent router (<5ms) |
| `src/edge/knowledge.py` | Person 1 | BM25 offline knowledge retrieval |
| `src/edge/metrics.py` | Person 1 | Latency waterfall + resource telemetry |
| `src/edge/audio_io.py` | Person 2 | Mic stream + AudioPlayer + WAV utils |
| `src/edge/vad.py` | Person 2 | Silero VAD ONNX with pre-roll + barge-in |
| `src/edge/asr.py` | Person 2 | Dual-tier ASR (Whisper + Sherpa-ONNX) |
| `src/edge/llm.py` | Person 3 | llama-cpp GGUF streaming + KV-cache |
| `src/edge/tts.py` | Person 3 | Piper ONNX TTS + sentence chunker |
| `src/edge/tts_cache.py` | Person 3 | Disk+memory TTS cache |
| `src/edge/governor.py` | Person 4 | Degradation ladder with hysteresis |
| `src/edge/limits.py` | Person 4 | CPU affinity + RAM cap enforcement |
| `src/edge/pipeline.py` | Person 4 | 3-thread overlapped pipeline |
| `dashboard/server.py` | Person 4 | FastAPI + WebSocket telemetry server |
| `dashboard/static/index.html` | Person 4 | Live monitoring dashboard UI |

---

## Latency Results (Baseline N=30, Ryzen 7, 4 cores, no GPU)

| Path | p50 E2E | p95 E2E | Max |
|---|---|---|---|
| All requests | 206 ms | 474 ms | 489 ms |
| Router-only (20 queries) | **174 ms** | — | — |
| LLM path (10 queries) | 447 ms | — | 489 ms |

**End-to-End = tts_first_audio − speech_end**

---

## Resource Envelope

| Resource | Limit | Observed (Baseline) |
|---|---|---|
| CPU Cores | 4 (pinned via affinity) | 14.3% utilization |
| RAM (process) | 3072 MB cap | **31.6 MB** (models not loaded) |
| Network | 0 bytes | **0 bytes (offline proof passed)** |

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (VAD + ASR + LLM + TTS)
python scripts/download_models.py

# 3. Start EdgeVoice
python main.py

# Dashboard: http://localhost:8000
```

---

## Benchmarks

```bash
# Baseline latency (N=30)
python bench/run_baseline.py

# Degradation ladder (Tier 1→2→3→2→1)
python bench/run_degradation.py

# Offline network proof (0 network calls)
python bench/offline_proof.py
```

---

## Governor Degradation Ladder

| Tier | Condition | ASR | LLM Threads |
|---|---|---|---|
| Tier 1 | Default (headroom OK) | Whisper base.en | 4 |
| Tier 2 | RAM > 85% OR CPU > 88% | Whisper tiny.en | 3 |
| Tier 3 | Severe pressure | Router-only | 2 |

Hysteresis: 3.5s minimum between tier changes to prevent flapping.

---

## Offline Proof

`bench/offline_proof.py` patches `socket.socket.connect` at runtime and verifies:
- Fast-path router: 0 network calls
- BM25 knowledge base: 0 network calls
- Metrics collector: 0 network calls
- Governor: 0 network calls
- TTS sentence chunker: 0 network calls

**Result: ALL TESTS PASSED — 0 network calls made.**
