# HackNEX 2026: On-Device Conversational Stack (HNX26EPS08)

An offline, CPU-only, ultra-low-latency voice-to-voice assistant running natively on Windows with strict OS-level hardware resource governance and real-time telemetry.

---

## 1. System Architecture Overview

```
                      +-------------------+
                      |   Laptop Mic      |
                      +---------+---------+
                                | 16kHz float32
                                v
               +----------------------------------+
               |  Silero VAD (ONNX, 64-ctx)       | <--- Pre-roll Ring Buffer (384ms)
               +----------------+-----------------+      Barge-in Interruption Detector
                                | SpeechSegment
                                v
               +----------------------------------+
               | Faster-Whisper ASR (CTranslate2) | <--- Multilingual & Int8 Greedy Decoding
               +----------------+-----------------+      Phantom Hallucination Filter
                                | Clean Transcript
                                v
               +----------------------------------+
               | Fast-Path Intent Router (<1ms)   | ---> Instant Response (Time, Date, Math,
               +----------------+-----------------+      Battery, RAM, CPU, Apps, First-Aid)
                                | Unhandled queries
                                v
               +----------------------------------+
               | BM25 Knowledge Retrieval (In-Mem)| ---> Context Injection
               +----------------+-----------------+
                                |
                                v
               +----------------------------------+
               | Local GGUF LLM (llama-cpp-python)| ---> Streaming Token Generator
               +----------------+-----------------+      KV-Cache Pre-warmed
                                | Token stream
                                v
               +----------------------------------+
               | Sentence-Chunked TTS (Piper ONNX)| ---> Audio Chunks (<150ms TTFA)
               +----------------+-----------------+      + 0ms TTS Audio Cache
                                |
                                v
                      +-------------------+
                      | Speaker Playback  |
                      +-------------------+
```

---

## 2. 4-Module Integration Map

| Module | Lead | Responsibilities | Key Files |
| :--- | :--- | :--- | :--- |
| **Fast Router & Knowledge** | Person 1 | Instant OS control, math, time, battery, first-aid | `src/edge/router.py`, `src/edge/knowledge_search.py`, `data/knowledge/` |
| **Module A (Listening)** | Person 2 | Resampling mic capture, Silero VAD, barge-in, Whisper int8 | `src/edge/audio_io.py`, `src/edge/vad.py`, `src/edge/asr.py` |
| **Module B (Thinking & Voice)** | Person 3 | BM25 retrieval, GGUF streaming LLM, sentence-chunked Piper TTS | `src/edge/knowledge.py`, `src/edge/llm.py`, `src/edge/tts.py`, `src/edge/tts_cache.py` |
| **Module C & D (Governance & UI)**| Person 4 | Win32 Job Object RAM cap, CPU affinity, metrics waterfall, live dashboard | `src/edge/limits.py`, `src/edge/metrics.py`, `src/edge/governor.py`, `dashboard/server.py` |

---

## 3. Strict Contest Constraints Verified

1. **100% On-Device & CPU-Only:** Zero cloud API calls, zero GPU requirement.
2. **Natively Enforced Limits:** 4 CPU cores (pinned via `psutil.cpu_affinity`) and 3072 MB RAM cap (enforced via Windows Win32 Job Object).
3. **No Mocks or Stubs:** All benchmarks, audio processing, and metrics reflect real local execution.
4. **End-to-End Latency Metric:** Measured from `speech_end` to `tts_first_audio`.

---

## 4. Quickstart (<60 Seconds)

### Step 1: Launch Full Demo
```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
```
This script automatically:
1. Enforces the 4-core and 3072 MB RAM limits natively on Windows.
2. Spawns the live FastAPI telemetry server and WebSocket bridge on port 8000.
3. Opens the glassmorphism Mission Control dashboard at `http://localhost:8000`.

### Step 2: Run Automated Tests
```powershell
# Run Module A verification (VAD + ASR + WER + Barge-in)
python tests\test_asr_vad.py

# Run Module B verification (BM25 + LLM TTFT + TTS TTFA + Cache)
python tests\test_llm_tts.py
```

### Step 3: Run Benchmark Suite
```powershell
# Run Baseline comparison
python bench\run_baseline.py

# Run Ablation study (Router, Cache, Full Stack)
python bench\run_ablation.py

# Run Offline Network Proof (Monkeysockets 0 outbound connections)
python bench\offline_proof.py
```
