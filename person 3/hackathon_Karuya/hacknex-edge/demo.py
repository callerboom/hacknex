import sys, time, re, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.edge.contracts import Turn
from src.edge.knowledge import KnowledgeRetriever
from src.edge.llm import GGUFLocalLLM
from src.edge.router import FastRouter
from src.edge.tts import SentenceChunkedTTS
from src.edge.tts_cache import TTSCache, CachedTTS
import numpy as np

# ---------------------------------------------------------------------------
# Natural Human Speech Manager (Indian English Neural + Piper Offline Fallback)
# ---------------------------------------------------------------------------
import ctypes, asyncio

VOICE_MAP = {
    "indian": "en-IN-PrabhatNeural",
    "prabhat": "en-IN-PrabhatNeural",
    "neerja": "en-IN-NeerjaNeural",
}

mci = ctypes.windll.winmm.mciSendStringW if sys.platform == "win32" else None

def play_mp3_native(file_path: str) -> bool:
    if not mci:
        return False
    abs_path = os.path.abspath(file_path)
    alias = f"tts_{int(time.time() * 1000)}"
    try:
        ret1 = mci(f'open "{abs_path}" type mpegvideo alias {alias}', None, 0, 0)
        ret2 = mci(f'play {alias} wait', None, 0, 0)
        mci(f'close {alias}', None, 0, 0)
        return ret1 == 0 and ret2 == 0
    except Exception:
        return False

class SpeechManager:
    def __init__(self, voice_name: str = "prabhat", rate_pct: int = 15):
        self.voice_name = voice_name.lower()
        self.rate_pct = rate_pct  # +15% is natural, snappy conversational pace
        self.rate = 165
        self.enabled = True

    def speak(self, text: str, piper_chunks=None, piper_sample_rate=22050):
        if not self.enabled:
            return

        has_code = "```" in text or "def " in text or "import " in text or "class " in text
        spoken_body = re.sub(r"```[\s\S]*?```", "", text).strip()
        spoken_body = re.sub(r"[*#_`~>|]", "", spoken_body).strip()

        sentences = [s.strip() for s in spoken_body.split(".") if len(s.strip()) > 3]
        if sentences:
            speak_text = ". ".join(sentences[:2]) + "."
        else:
            speak_text = "Here is the solution you requested."

        if has_code and not speak_text.endswith("on your screen."):
            speak_text += " The complete code is displayed on your screen."

        if not speak_text.strip():
            return

        # 1. Indian English Neural Voice (Prabhat / Neerja)
        if self.voice_name in VOICE_MAP:
            target_voice = VOICE_MAP[self.voice_name]
            label = "Indian Male (Prabhat)" if "Prabhat" in target_voice else "Indian Female (Neerja)"
            rate_str = f"+{self.rate_pct}%" if self.rate_pct >= 0 else f"{self.rate_pct}%"
            print(f"  [AUDIO] Speaking ({label} @ {rate_str} speed)...")
            try:
                import edge_tts
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

                async def _synth():
                    tmp_file = os.path.abspath(f"tts_voice_{int(time.time()*1000)}.mp3")
                    comm = edge_tts.Communicate(speak_text, target_voice, rate=rate_str)
                    await comm.save(tmp_file)
                    return tmp_file

                tmp_mp3 = asyncio.run(_synth())
                if os.path.exists(tmp_mp3):
                    ok = play_mp3_native(tmp_mp3)
                    try:
                        os.remove(tmp_mp3)
                    except Exception:
                        pass
                    if ok:
                        return
            except Exception as e:
                print(f"  [AUDIO] Indian voice offline fallback: {e}")

        # 2. Piper Neural Voice (Offline ONNX)
        if (self.voice_name == "piper" or piper_chunks) and piper_chunks:
            try:
                import sounddevice as sd
                full_audio = np.concatenate(piper_chunks)
                audio_sec = len(full_audio) / piper_sample_rate
                print(f"  [AUDIO] Speaking aloud via Piper Neural Voice ({audio_sec:.1f}s)...")
                sd.play(full_audio, piper_sample_rate)
                sd.wait()
                return
            except Exception:
                pass

        # 3. Windows SAPI5 (David / Zira)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            for v in engine.getProperty('voices'):
                if self.voice_name in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', 1.0)
            print(f"  [AUDIO] Speaking aloud via Windows SAPI ({self.voice_name.capitalize()})...")
            engine.say(speak_text)
            engine.runAndWait()
            del engine
            return
        except Exception:
            pass

        # 4. Direct PowerShell System.Speech fallback
        try:
            import subprocess
            escaped = speak_text.replace("'", "").replace('"', "")
            ps_code = f"Add-Type -AssemblyName System.speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 0; $s.Speak('{escaped}');"
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_code], timeout=15)
        except Exception:
            pass

speech = SpeechManager(voice_name="neerja", rate_pct=15)

print()
print("=" * 65)
print("  ED-RA — ON-DEVICE CONVERSATIONAL AI")
print("=" * 65)
print()

# 1. Knowledge Base (Top-K Enabled)
print("[1/5] Loading Knowledge Base...")
kr = KnowledgeRetriever()
kr.load()
print(f"      [OK] Knowledge index ready ({len(kr._index._docs)} chunks)")

# 2. Fast Deterministic Router
print("[2/5] Initializing Fast Router...")
router = FastRouter()
print("      [OK] Sub-5ms fast path engine ready")

# 3. Local LLM
print("[3/5] Loading LLM Engine...")
llm = GGUFLocalLLM()
llm.load()
model_status = "Real GGUF Model (Qwen2.5 0.5B)" if llm._llm is not None else "Fallback Mode"
print(f"      [OK] LLM Mode: {model_status}")

# 4. Cached TTS
print("[4/5] Loading TTS & Audio Cache...")
live_tts = SentenceChunkedTTS()
cache = TTSCache(sample_rate=live_tts.sample_rate)

EMERGENCY_QC_PHRASES = [
    "Call emergency services immediately.",
    "Stay calm, help is on the way.",
    "Do not apply ice directly to a burn.",
    "Cool the burn with running cold water for at least 10 minutes.",
    "Check if the person is responsive and breathing.",
    "I am Edra, your offline first aid assistant.",
]
for p in EMERGENCY_QC_PHRASES:
    if cache.get(p) is None:
        try:
            chunks = list(live_tts.synth_stream(p))
            if chunks:
                cache.put(p, np.concatenate(chunks))
        except Exception:
            pass

cache.warm(live_tts)
cached_tts = CachedTTS(tts=live_tts, cache=cache)
print(f"      [OK] TTS Cache warmed ({len(cache._ram)} phrases)")

# 5. Speaker Voice Status
voice_label = "Indian English Female (Neerja)" if speech.voice_name == "neerja" else ("Indian English Male (Prabhat)" if speech.voice_name in ("prabhat", "indian") else speech.voice_name.capitalize())
print(f"[5/5] Speaker Voice: ENABLED ({voice_label})")

# Multi-Turn Conversation Memory
history: list[Turn] = []

print()
print("=" * 65)
print("  ED-RA READY  (Type your message or 'quit' to exit)")
print("=" * 65)

while True:
    print()
    try:
        user_input = input("  User > ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not user_input:
        continue

    # Commands
    cmd = user_input.lower()
    if cmd in ("quit", "exit", "q"):
        break
    if cmd == "/clear":
        history.clear()
        print("  [INFO] Conversation memory cleared! Fresh session started.")
        continue
    if cmd == "/history":
        print(f"  [INFO] Memory ({len(history)} turns stored):")
        for i, h in enumerate(history, 1):
            role = "User" if h.role == "user" else "Edra"
            print(f"    {i}. {role}: {h.content[:70]}...")
        continue
    if cmd == "/mute":
        speech.enabled = False
        print("  [INFO] Speaker voice MUTED.")
        continue
    if cmd == "/unmute":
        speech.enabled = True
        print("  [INFO] Speaker voice UNMUTED.")
        continue
    if cmd.startswith("/voice"):
        parts = cmd.split()
        if len(parts) > 1 and parts[1] in ("prabhat", "indian", "neerja", "piper", "zira", "david"):
            speech.voice_name = parts[1]
            desc_map = {
                "neerja": "Indian English (Female - Expressive Natural)",
                "prabhat": "Indian English (Male - Natural Accent)",
                "indian": "Indian English (Male - Natural Accent)",
                "piper": "Piper Neural Voice (US English - Offline)",
                "zira": "Windows Zira (Offline SAPI)",
                "david": "Windows David (Offline SAPI)",
            }
            v_desc = desc_map.get(parts[1], parts[1].capitalize())
            print(f"  [INFO] Voice switched to: {v_desc}")
            speech.speak(f"Voice changed to {v_desc}. How can I assist you?")
        else:
            print("  [INFO] Available voices:")
            print("         /voice neerja   -> Indian English Female (Default)")
            print("         /voice prabhat  -> Indian English Male")
            print("         /voice piper    -> Piper US English (Offline)")
            print("         /voice zira     -> Windows Zira")
            print("         /voice david    -> Windows David")
        continue
    if cmd.startswith("/speed"):
        parts = cmd.split()
        if len(parts) > 1 and (parts[1].isdigit() or (parts[1].startswith(("-", "+")) and parts[1][1:].isdigit())):
            val = int(parts[1])
            if val > 50:
                speech.rate = val
                speech.rate_pct = int((val - 150) / 1.5)
            else:
                speech.rate_pct = val
                speech.rate = int(150 + val * 1.5)
            print(f"  [INFO] Speaking speed set to: +{speech.rate_pct}% ({speech.rate} WPM).")
            speech.speak(f"Speaking speed set to {speech.rate_pct} percent.")
        else:
            print("  [INFO] Usage: /speed 15 (Percentage: 0 to 40) or /speed 165 (WPM: 130 to 200)")
        continue

    # Image Upload / Vision Query Handling
    image_target = None
    if cmd.startswith("/image"):
        parts = user_input.split(maxsplit=2)
        if len(parts) > 1:
            image_target = parts[1].strip()
            user_input = parts[2].strip() if len(parts) > 2 else "Please inspect this image and describe its key elements in detail."
            print(f"  [VISION] Analyzing uploaded image: {image_target}")
        else:
            print("  [INFO] Usage: /image <path/to/image.png> [optional question]")
            continue

    t_start = time.perf_counter()

    # Step 0: Fast Deterministic Router (Math, System Telemetry, Local Files, System32, Launcher)
    if not image_target:
        fast_ans = router.route(user_input)
        if fast_ans is not None:
            route_ms = (time.perf_counter() - t_start) * 1000
            print()
            print("  EDRA: ", end="", flush=True)
            print(fast_ans)
            print()
            print(f"  [Metrics] Fast Router: {route_ms:.1f}ms | 0ms LLM Delay")
            print(f"  [State]   Memory: {len(history)} turns | Deterministic Fast Path")

            history.append(Turn(role="user", content=user_input))
            history.append(Turn(role="assistant", content=fast_ans))
            if len(history) > 6:
                history = history[-6:]

            if speech.enabled:
                speech.speak(fast_ans)
            continue

    # Step 1: Top-K Knowledge Injection
    t_k0 = time.perf_counter()
    top_k_snippet = kr.search(user_input, top_k=2)
    bm25_ms = (time.perf_counter() - t_k0) * 1000
    llm.knowledge_snippet = top_k_snippet

    # Step 2: LLM Streaming with Multi-Turn Memory
    t_llm0 = time.perf_counter()
    ttft = None
    answer_tokens = []
    print()
    print("  EDRA: ", end="", flush=True)

    for tok in llm.stream(user_input, history, image_path=image_target):
        if ttft is None:
            ttft = (time.perf_counter() - t_llm0) * 1000
        answer_tokens.append(tok)
        print(tok, end="", flush=True)

    llm_ms = (time.perf_counter() - t_llm0) * 1000
    answer = "".join(answer_tokens).strip()

    # Fallback Quality Control Guard
    if not answer or len(answer) < 5:
        answer = "I am ready to help with coding, explanations, and problem solving."
        print(answer)

    print()

    # Save to Multi-turn History (keep last 6 turns)
    history.append(Turn(role="user", content=user_input))
    history.append(Turn(role="assistant", content=answer))
    if len(history) > 6:
        history = history[-6:]

    # Step 3: Fast Audio Preparation (Zero Delay)
    t_tts0 = time.perf_counter()
    audio_chunks = []
    if speech.voice_name == "piper":
        first_sentence = answer.split(".")[0].strip() + "." if "." in answer else answer
        cached_first = cache.get(first_sentence)
        if cached_first is not None:
            cache_status = "CACHE HIT (0ms TTFA)"
            audio_chunks.extend(cache.stream(first_sentence))
        else:
            cache_status = "LIVE TTS SYNTH"
            audio_chunks.extend(live_tts.stream_from_tokens(iter(answer.split())))
    else:
        cache_status = f"EDGE NEURAL ({speech.voice_name.upper()} - FAST)"

    tts_ms = (time.perf_counter() - t_tts0) * 1000
    total_ms = (time.perf_counter() - t_start) * 1000
    total_samples = sum(len(c) for c in audio_chunks)
    audio_sec = total_samples / live_tts.sample_rate if total_samples else 0.0

    print()
    print(f"  [Metrics] BM25(Top-2): {bm25_ms:.1f}ms | LLM(TTFT {ttft or 0:.0f}ms): {llm_ms:.0f}ms | Total: {total_ms:.0f}ms")
    print(f"  [State]   Memory: {len(history)} turns | Cache: {cache_status}")

    # Step 4: Real Human Voice Playback through Speakers
    if speech.enabled:
        speech.speak(answer, piper_chunks=audio_chunks, piper_sample_rate=live_tts.sample_rate)

print()
print("  Person 3 Module B - Session Ended.")
