# HackNEX 2026: English Module A Integration Notes

**Author:** Edge Audio & ASR Lead (Module A)  
**Recipients:** Person 1 (Router / Brain Lead), Person 3 (LLM / TTS Lead), Person 4 (Integrator & Live Delivery Lead)  
**Target Platform:** Windows 11 Offline CPU Laptop (AMD Ryzen 7 7445HS)  
**Language:** English (`en`)

---

## 1. What Downstream Modules Receive

Module A exposes `SpeechSegment` and `Transcript` strictly meeting `src/edge/contracts.py`:

```python
@dataclass
class Transcript:
    text: str                          # Normalized clean transcribed English text
    is_final: bool                     # True for complete turn
    lang: str                          # Strictly 'en'
    asr_ms: float                      # Pure CTranslate2 inference latency (~230-260ms)
    confidence: float                  # Acoustic confidence score in [0.0, 1.0]
    words: list[dict]                  # Word timestamps [{'word', 'start', 'end', 'prob'}]
    lang_prob: float                   # 1.0 for English
    lang_candidates: dict[str, float]  # {'en': 1.0}
    translated: bool                   # False
    translation_text: str              # Empty
```

### Guarantees for Downstream Modules:
1. **Language Code is Strictly `"en"`:** No foreign language ambiguity.
2. **Confidence Gate:** Actions require confidence $\ge 0.40$. Low-confidence utterances or phantoms are sanitized to `""`.
3. **Empty Text Handling:** Empty transcripts, silence, or noise return `text=""` and `confidence=0.0`. Downstream modules return polite repeat prompt.
4. **Instant Action Support:** Common OS queries (Battery, RAM, Time, Date, Notepad, Calculator) are processed under 1ms.

---

## 2. Voice Playback & Barge-In Protection

Person 3 & Person 4 can safely use the `AudioPlayer` from `src/edge/audio_io.py`:
- `player.play_array(audio_array)`
- `player.stop()`: Halts output instantly (<10ms) when user interrupts.
- Voice assets: Piper TTS `en_US-lessac-medium` (`models/tts/en_piper`).
