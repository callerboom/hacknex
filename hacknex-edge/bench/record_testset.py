"""Judge Testset & Latency Benchmark Runner.

Prepares 100+ structured judge evaluation queries across multilingual fast-path categories
(English, Tamil, Tanglish, Hindi, Telugu, Malayalam, Kannada) and offline knowledge queries:
- Greeting (7 languages)
- Time & Date (7 languages)
- Hardware Telemetry: Battery, RAM, CPU (7 languages)
- Network & Internet Telemetry: Wi-Fi, Internet (7 languages)
- System Controls: Volume, Screenshot (7 languages)
- App Launchers: Calculator, Notepad (7 languages)
- First-Aid & Emergency Guidance (7 languages)
- Health & Wellness Guidelines (7 languages)
- Basic Calculator & Unit Conversions (7 languages)
- Noise / Silence detection
- General Creative & Complex LLM queries (forwarded to LLM -> returns None)
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.edge.router import route_intent


TEST_QUESTIONS: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # 1. GREETING & IDENTITY (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 1, "category": "GREETING", "language": "English", "query": "hello", "fast_routed": True, "description": "English Hello"},
    {"id": 2, "category": "GREETING", "language": "English", "query": "good morning", "fast_routed": True, "description": "English Morning"},
    {"id": 3, "category": "GREETING", "language": "English", "query": "who are you", "fast_routed": True, "description": "English Identity"},
    {"id": 4, "category": "GREETING", "language": "Tamil", "query": "வணக்கம்", "fast_routed": True, "description": "Tamil Vanakkam"},
    {"id": 5, "category": "GREETING", "language": "Tanglish", "query": "vanakkam", "fast_routed": True, "description": "Tanglish Vanakkam"},
    {"id": 6, "category": "GREETING", "language": "Tanglish", "query": "epdi iruka", "fast_routed": True, "description": "Tanglish How are you"},
    {"id": 7, "category": "GREETING", "language": "Hindi", "query": "नमस्ते", "fast_routed": True, "description": "Hindi Namaste"},
    {"id": 8, "category": "GREETING", "language": "Hindi", "query": "namaste", "fast_routed": True, "description": "Hinglish Namaste"},
    {"id": 9, "category": "GREETING", "language": "Hindi", "query": "kaise ho", "fast_routed": True, "description": "Hinglish How are you"},
    {"id": 10, "category": "GREETING", "language": "Telugu", "query": "నమస్కారం", "fast_routed": True, "description": "Telugu Namaskaram"},
    {"id": 11, "category": "GREETING", "language": "Telugu", "query": "ela unnav", "fast_routed": True, "description": "Telugish How are you"},
    {"id": 12, "category": "GREETING", "language": "Malayalam", "query": "നമസ്കാരം", "fast_routed": True, "description": "Malayalam Namaskaram"},
    {"id": 13, "category": "GREETING", "language": "Malayalam", "query": "sukhamano", "fast_routed": True, "description": "Manglish How are you"},
    {"id": 14, "category": "GREETING", "language": "Kannada", "query": "ನಮಸ್ಕಾರ", "fast_routed": True, "description": "Kannada Namaskara"},
    {"id": 15, "category": "GREETING", "language": "Kannada", "query": "hegiddira", "fast_routed": True, "description": "Kanglish How are you"},

    # --------------------------------------------------------------------------
    # 2. TIME (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 16, "category": "TIME", "language": "English", "query": "what time is it", "fast_routed": True, "description": "English time"},
    {"id": 17, "category": "TIME", "language": "English", "query": "current time", "fast_routed": True, "description": "English current time"},
    {"id": 18, "category": "TIME", "language": "Tamil", "query": "நேரம் என்ன", "fast_routed": True, "description": "Tamil time"},
    {"id": 19, "category": "TIME", "language": "Tanglish", "query": "enna time", "fast_routed": True, "description": "Tanglish enna time"},
    {"id": 20, "category": "TIME", "language": "Tanglish", "query": "time sollu", "fast_routed": True, "description": "Tanglish time sollu"},
    {"id": 21, "category": "TIME", "language": "Hindi", "query": "समय क्या है", "fast_routed": True, "description": "Hindi samay kya hai"},
    {"id": 22, "category": "TIME", "language": "Hindi", "query": "time kya hai", "fast_routed": True, "description": "Hinglish time kya hai"},
    {"id": 23, "category": "TIME", "language": "Telugu", "query": "సమయం ఎంత", "fast_routed": True, "description": "Telugu samayam entha"},
    {"id": 24, "category": "TIME", "language": "Telugu", "query": "time entha", "fast_routed": True, "description": "Telugish time entha"},
    {"id": 25, "category": "TIME", "language": "Malayalam", "query": "സമയം എത്ര", "fast_routed": True, "description": "Malayalam samayam ethra"},
    {"id": 26, "category": "TIME", "language": "Malayalam", "query": "samayam ethra aayi", "fast_routed": True, "description": "Manglish samayam ethra"},
    {"id": 27, "category": "TIME", "language": "Kannada", "query": "ಸಮಯ ಎಷ್ಟು", "fast_routed": True, "description": "Kannada samaya eshtu"},
    {"id": 28, "category": "TIME", "language": "Kannada", "query": "samaya eshtu", "fast_routed": True, "description": "Kanglish samaya eshtu"},

    # --------------------------------------------------------------------------
    # 3. DATE - TODAY, TOMORROW, YESTERDAY (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 29, "category": "DATE", "language": "English", "query": "what is today's date", "fast_routed": True, "description": "English today date"},
    {"id": 30, "category": "DATE", "language": "Tamil", "query": "இன்றைய தேதி என்ன", "fast_routed": True, "description": "Tamil today date"},
    {"id": 31, "category": "DATE", "language": "Tanglish", "query": "innaiki date enna", "fast_routed": True, "description": "Tanglish today date"},
    {"id": 32, "category": "DATE", "language": "Hindi", "query": "आज की तारीख क्या है", "fast_routed": True, "description": "Hindi today date"},
    {"id": 33, "category": "DATE", "language": "Hindi", "query": "aaj ki date kya hai", "fast_routed": True, "description": "Hinglish today date"},
    {"id": 34, "category": "DATE", "language": "Telugu", "query": "ఈ రోజు తేదీ ఎంత", "fast_routed": True, "description": "Telugu today date"},
    {"id": 35, "category": "DATE", "language": "Telugu", "query": "ee roju date enti", "fast_routed": True, "description": "Telugish today date"},
    {"id": 36, "category": "DATE", "language": "Malayalam", "query": "ഇന്നത്തെ തീയതി എന്താണ്", "fast_routed": True, "description": "Malayalam today date"},
    {"id": 37, "category": "DATE", "language": "Malayalam", "query": "innu ethra theeyathi", "fast_routed": True, "description": "Manglish today date"},
    {"id": 38, "category": "DATE", "language": "Kannada", "query": "ಇವತ್ತಿನ ದಿನಾಂಕ ಏನು", "fast_routed": True, "description": "Kannada today date"},
    {"id": 39, "category": "DATE", "language": "Kannada", "query": "ivathu date enu", "fast_routed": True, "description": "Kanglish today date"},
    {"id": 40, "category": "DATE", "language": "English", "query": "what is tomorrow's date", "fast_routed": True, "description": "English tomorrow"},
    {"id": 41, "category": "DATE", "language": "Tanglish", "query": "nalaiku enna date", "fast_routed": True, "description": "Tanglish tomorrow"},
    {"id": 42, "category": "DATE", "language": "Hindi", "query": "kal ki date kya hai", "fast_routed": True, "description": "Hinglish tomorrow"},
    {"id": 43, "category": "DATE", "language": "Telugu", "query": "repu date enti", "fast_routed": True, "description": "Telugish tomorrow"},
    {"id": 44, "category": "DATE", "language": "English", "query": "what was yesterday's date", "fast_routed": True, "description": "English yesterday"},
    {"id": 45, "category": "DATE", "language": "Tanglish", "query": "nethu enna date", "fast_routed": True, "description": "Tanglish yesterday"},
    {"id": 46, "category": "DATE", "language": "Telugu", "query": "ninna date enti", "fast_routed": True, "description": "Telugish yesterday"},

    # --------------------------------------------------------------------------
    # 4. BATTERY & CHARGING (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 47, "category": "BATTERY", "language": "English", "query": "check battery", "fast_routed": True, "description": "English battery"},
    {"id": 48, "category": "BATTERY", "language": "English", "query": "battery status", "fast_routed": True, "description": "English battery status"},
    {"id": 49, "category": "BATTERY", "language": "Tamil", "query": "பேட்டரி நிலை என்ன", "fast_routed": True, "description": "Tamil battery"},
    {"id": 50, "category": "BATTERY", "language": "Tanglish", "query": "battery evlo", "fast_routed": True, "description": "Tanglish battery evlo"},
    {"id": 51, "category": "BATTERY", "language": "Tanglish", "query": "charge evlo iruku", "fast_routed": True, "description": "Tanglish charge"},
    {"id": 52, "category": "BATTERY", "language": "Hindi", "query": "बैटरी कितनी है", "fast_routed": True, "description": "Hindi battery"},
    {"id": 53, "category": "BATTERY", "language": "Hindi", "query": "battery kitna hai", "fast_routed": True, "description": "Hinglish battery"},
    {"id": 54, "category": "BATTERY", "language": "Telugu", "query": "బ్యాటరీ ఎంత ఉంది", "fast_routed": True, "description": "Telugu battery"},
    {"id": 55, "category": "BATTERY", "language": "Telugu", "query": "battery entha undi", "fast_routed": True, "description": "Telugish battery"},
    {"id": 56, "category": "BATTERY", "language": "Malayalam", "query": "ബാറ്ററി എത്ര ഉണ്ട്", "fast_routed": True, "description": "Malayalam battery"},
    {"id": 57, "category": "BATTERY", "language": "Malayalam", "query": "battery ethra undu", "fast_routed": True, "description": "Manglish battery"},
    {"id": 58, "category": "BATTERY", "language": "Kannada", "query": "ಬ್ಯಾಟರಿ ಎಷ್ಟು ಇದೆ", "fast_routed": True, "description": "Kannada battery"},
    {"id": 59, "category": "BATTERY", "language": "Kannada", "query": "battery eshtu ide", "fast_routed": True, "description": "Kanglish battery"},

    # --------------------------------------------------------------------------
    # 5. RAM / MEMORY (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 60, "category": "RAM", "language": "English", "query": "ram usage", "fast_routed": True, "description": "English RAM"},
    {"id": 61, "category": "RAM", "language": "Tamil", "query": "ரேம் பயன்பாடு என்ன", "fast_routed": True, "description": "Tamil RAM"},
    {"id": 62, "category": "RAM", "language": "Tanglish", "query": "RAM usage sollu", "fast_routed": True, "description": "Tanglish RAM usage sollu"},
    {"id": 63, "category": "RAM", "language": "Tanglish", "query": "ram evlo", "fast_routed": True, "description": "Tanglish RAM evlo"},
    {"id": 64, "category": "RAM", "language": "Hindi", "query": "रैम उपयोग कितना है", "fast_routed": True, "description": "Hindi RAM"},
    {"id": 65, "category": "RAM", "language": "Hindi", "query": "ram kitna use ho raha hai", "fast_routed": True, "description": "Hinglish RAM"},
    {"id": 66, "category": "RAM", "language": "Telugu", "query": "ర్యామ్ వినియోగం ఎంత", "fast_routed": True, "description": "Telugu RAM"},
    {"id": 67, "category": "RAM", "language": "Telugu", "query": "ram entha use avtundi", "fast_routed": True, "description": "Telugish RAM"},
    {"id": 68, "category": "RAM", "language": "Malayalam", "query": "റാം ഉപയോഗം എത്രയാണ്", "fast_routed": True, "description": "Malayalam RAM"},
    {"id": 69, "category": "RAM", "language": "Malayalam", "query": "ram usage ethra", "fast_routed": True, "description": "Manglish RAM"},
    {"id": 70, "category": "RAM", "language": "Kannada", "query": "ರ‍್ಯಾಮ್ ಬಳಕೆ ಎಷ್ಟು", "fast_routed": True, "description": "Kannada RAM"},
    {"id": 71, "category": "RAM", "language": "Kannada", "query": "ram usage eshtu", "fast_routed": True, "description": "Kanglish RAM"},

    # --------------------------------------------------------------------------
    # 6. CPU / PROCESSOR (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 72, "category": "CPU", "language": "English", "query": "cpu usage", "fast_routed": True, "description": "English CPU"},
    {"id": 73, "category": "CPU", "language": "English", "query": "processor load", "fast_routed": True, "description": "English CPU Load"},
    {"id": 74, "category": "CPU", "language": "Tamil", "query": "செயலி பயன்பாடு என்ன", "fast_routed": True, "description": "Tamil CPU"},
    {"id": 75, "category": "CPU", "language": "Tanglish", "query": "cpu evlo use aaguthu", "fast_routed": True, "description": "Tanglish CPU"},
    {"id": 76, "category": "CPU", "language": "Hindi", "query": "सीपीयू उपयोग कितना है", "fast_routed": True, "description": "Hindi CPU"},
    {"id": 77, "category": "CPU", "language": "Hindi", "query": "cpu usage kitna hai", "fast_routed": True, "description": "Hinglish CPU"},
    {"id": 78, "category": "CPU", "language": "Telugu", "query": "సిపియు లోడ్ ఎంత", "fast_routed": True, "description": "Telugu CPU"},
    {"id": 79, "category": "CPU", "language": "Telugu", "query": "cpu load entha", "fast_routed": True, "description": "Telugish CPU"},
    {"id": 80, "category": "CPU", "language": "Malayalam", "query": "സിപിയു ഉപയോഗം എത്ര", "fast_routed": True, "description": "Malayalam CPU"},
    {"id": 81, "category": "CPU", "language": "Malayalam", "query": "cpu usage ethra", "fast_routed": True, "description": "Manglish CPU"},
    {"id": 82, "category": "CPU", "language": "Kannada", "query": "ಸಿಪಿಯು ಬಳಕೆ ಎಷ್ಟು", "fast_routed": True, "description": "Kannada CPU"},
    {"id": 83, "category": "CPU", "language": "Kannada", "query": "cpu usage eshtu", "fast_routed": True, "description": "Kanglish CPU"},

    # --------------------------------------------------------------------------
    # 7. WIFI / NETWORK (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 84, "category": "NETWORK", "language": "English", "query": "wifi status", "fast_routed": True, "description": "English Wi-Fi"},
    {"id": 85, "category": "NETWORK", "language": "Tamil", "query": "வைஃபை நிலை என்ன", "fast_routed": True, "description": "Tamil Wi-Fi"},
    {"id": 86, "category": "NETWORK", "language": "Tanglish", "query": "WiFi status enna", "fast_routed": True, "description": "Tanglish WiFi status enna"},
    {"id": 87, "category": "NETWORK", "language": "Tanglish", "query": "wifi connect aagi iruka", "fast_routed": True, "description": "Tanglish Wi-Fi connect"},
    {"id": 88, "category": "NETWORK", "language": "Hindi", "query": "वाईफाई स्थिति क्या है", "fast_routed": True, "description": "Hindi Wi-Fi"},
    {"id": 89, "category": "NETWORK", "language": "Hindi", "query": "wifi connect hai kya", "fast_routed": True, "description": "Hinglish Wi-Fi"},
    {"id": 90, "category": "NETWORK", "language": "Telugu", "query": "వైఫై స్థితి ఏంటి", "fast_routed": True, "description": "Telugu Wi-Fi"},
    {"id": 91, "category": "NETWORK", "language": "Telugu", "query": "wifi connect ayinda", "fast_routed": True, "description": "Telugish Wi-Fi"},
    {"id": 92, "category": "NETWORK", "language": "Malayalam", "query": "വൈഫൈ അവസ്ഥ എന്താണ്", "fast_routed": True, "description": "Malayalam Wi-Fi"},
    {"id": 93, "category": "NETWORK", "language": "Malayalam", "query": "wifi status entha", "fast_routed": True, "description": "Manglish Wi-Fi"},
    {"id": 94, "category": "NETWORK", "language": "Kannada", "query": "ವೈಫೈ ಸ್ಥಿತಿ ಏನು", "fast_routed": True, "description": "Kannada Wi-Fi"},
    {"id": 95, "category": "NETWORK", "language": "Kannada", "query": "wifi status enu", "fast_routed": True, "description": "Kanglish Wi-Fi"},

    # --------------------------------------------------------------------------
    # 8. SCREENSHOT (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 96, "category": "SCREENSHOT", "language": "English", "query": "take screenshot", "fast_routed": True, "description": "English Screenshot"},
    {"id": 97, "category": "SCREENSHOT", "language": "Tamil", "query": "ஸ்கிரீன்ஷாட் எடு", "fast_routed": True, "description": "Tamil Screenshot"},
    {"id": 98, "category": "SCREENSHOT", "language": "Tanglish", "query": "screenshot edu", "fast_routed": True, "description": "Tanglish Screenshot edu"},
    {"id": 99, "category": "SCREENSHOT", "language": "Hindi", "query": "स्क्रीनशॉट लो", "fast_routed": True, "description": "Hindi Screenshot"},
    {"id": 100, "category": "SCREENSHOT", "language": "Hindi", "query": "screenshot lo", "fast_routed": True, "description": "Hinglish Screenshot"},
    {"id": 101, "category": "SCREENSHOT", "language": "Telugu", "query": "స్క్రీన్‌షాట్ తీయి", "fast_routed": True, "description": "Telugu Screenshot"},
    {"id": 102, "category": "SCREENSHOT", "language": "Telugu", "query": "screenshot theeyi", "fast_routed": True, "description": "Telugish Screenshot"},
    {"id": 103, "category": "SCREENSHOT", "language": "Malayalam", "query": "സ്ക്രീൻഷോട്ട് എടുക്കുക", "fast_routed": True, "description": "Malayalam Screenshot"},
    {"id": 104, "category": "SCREENSHOT", "language": "Malayalam", "query": "screenshot edukku", "fast_routed": True, "description": "Manglish Screenshot"},
    {"id": 105, "category": "SCREENSHOT", "language": "Kannada", "query": "ಸ್ಕ್ರೀನ್‌ಶಾಟ್ ತೆಗೆಯಿರಿ", "fast_routed": True, "description": "Kannada Screenshot"},
    {"id": 106, "category": "SCREENSHOT", "language": "Kannada", "query": "screenshot thegedhukolli", "fast_routed": True, "description": "Kanglish Screenshot"},

    # --------------------------------------------------------------------------
    # 9. OPEN CALCULATOR (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 107, "category": "APP_LAUNCHER", "language": "English", "query": "open calculator", "fast_routed": True, "description": "English Calculator"},
    {"id": 108, "category": "APP_LAUNCHER", "language": "Tamil", "query": "கால்குலேட்டரை திற", "fast_routed": True, "description": "Tamil Calculator"},
    {"id": 109, "category": "APP_LAUNCHER", "language": "Tanglish", "query": "calculator open pannu", "fast_routed": True, "description": "Tanglish Calculator"},
    {"id": 110, "category": "APP_LAUNCHER", "language": "Hindi", "query": "कैलकुलेटर खोलो", "fast_routed": True, "description": "Hindi Calculator"},
    {"id": 111, "category": "APP_LAUNCHER", "language": "Hindi", "query": "calculator open karo", "fast_routed": True, "description": "Hinglish Calculator"},
    {"id": 112, "category": "APP_LAUNCHER", "language": "Telugu", "query": "కాలిక్యులేటర్ ఓపెన్ చేయి", "fast_routed": True, "description": "Telugu Calculator"},
    {"id": 113, "category": "APP_LAUNCHER", "language": "Telugu", "query": "calculator open cheyi", "fast_routed": True, "description": "Telugish Calculator"},
    {"id": 114, "category": "APP_LAUNCHER", "language": "Malayalam", "query": "കാൽക്കുലേറ്റർ തുറക്കുക", "fast_routed": True, "description": "Malayalam Calculator"},
    {"id": 115, "category": "APP_LAUNCHER", "language": "Malayalam", "query": "calculator thurakku", "fast_routed": True, "description": "Manglish Calculator"},
    {"id": 116, "category": "APP_LAUNCHER", "language": "Kannada", "query": "ಕ್ಯಾಲ್ಕುಲೇಟರ್ ತೆರೆಯಿರಿ", "fast_routed": True, "description": "Kannada Calculator"},
    {"id": 117, "category": "APP_LAUNCHER", "language": "Kannada", "query": "calculator open maadi", "fast_routed": True, "description": "Kanglish Calculator"},

    # --------------------------------------------------------------------------
    # 10. OPEN NOTEPAD (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 118, "category": "APP_LAUNCHER", "language": "English", "query": "open notepad", "fast_routed": True, "description": "English Notepad"},
    {"id": 119, "category": "APP_LAUNCHER", "language": "Tamil", "query": "நோட்பேடை திற", "fast_routed": True, "description": "Tamil Notepad"},
    {"id": 120, "category": "APP_LAUNCHER", "language": "Tanglish", "query": "notepad open pannu", "fast_routed": True, "description": "Tanglish Notepad"},
    {"id": 121, "category": "APP_LAUNCHER", "language": "Hindi", "query": "नोटपैड खोलो", "fast_routed": True, "description": "Hindi Notepad"},
    {"id": 122, "category": "APP_LAUNCHER", "language": "Hindi", "query": "notepad kholo", "fast_routed": True, "description": "Hinglish Notepad"},
    {"id": 123, "category": "APP_LAUNCHER", "language": "Telugu", "query": "నోట్‌ప్యాడ్ ఓపెన్ చేయి", "fast_routed": True, "description": "Telugu Notepad"},
    {"id": 124, "category": "APP_LAUNCHER", "language": "Telugu", "query": "notepad open cheyi", "fast_routed": True, "description": "Telugish Notepad"},
    {"id": 125, "category": "APP_LAUNCHER", "language": "Malayalam", "query": "നോട്ട്പാഡ് തുറക്കുക", "fast_routed": True, "description": "Malayalam Notepad"},
    {"id": 126, "category": "APP_LAUNCHER", "language": "Malayalam", "query": "notepad thurakku", "fast_routed": True, "description": "Manglish Notepad"},
    {"id": 127, "category": "APP_LAUNCHER", "language": "Kannada", "query": "ನೋಟ್‌ಪ್ಯಾಡ್ ತೆರೆಯಿರಿ", "fast_routed": True, "description": "Kannada Notepad"},
    {"id": 128, "category": "APP_LAUNCHER", "language": "Kannada", "query": "notepad open maadi", "fast_routed": True, "description": "Kanglish Notepad"},

    # --------------------------------------------------------------------------
    # 11. FIRST-AID EMERGENCY GUIDANCE (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 129, "category": "FIRST_AID", "language": "English", "query": "how do I treat a burn", "fast_routed": True, "description": "English Burns"},
    {"id": 130, "category": "FIRST_AID", "language": "English", "query": "what are the steps for adult cpr", "fast_routed": True, "description": "English CPR"},
    {"id": 131, "category": "FIRST_AID", "language": "English", "query": "choking heimlich maneuver", "fast_routed": True, "description": "English Choking"},
    {"id": 132, "category": "FIRST_AID", "language": "Tamil", "query": "தீக்காயத்திற்கு முதலுதவி என்ன", "fast_routed": True, "description": "Tamil Burns"},
    {"id": 133, "category": "FIRST_AID", "language": "Tanglish", "query": "burn aana enna pannanum", "fast_routed": True, "description": "Tanglish Burns"},
    {"id": 134, "category": "FIRST_AID", "language": "Tanglish", "query": "theekaayam first aid", "fast_routed": True, "description": "Tanglish Theekaayam"},
    {"id": 135, "category": "FIRST_AID", "language": "Tanglish", "query": "paambu kadicha enna pannanum", "fast_routed": True, "description": "Tanglish Snake bite"},
    {"id": 136, "category": "FIRST_AID", "language": "Hindi", "query": "जलने पर क्या करें", "fast_routed": True, "description": "Hindi Burns"},
    {"id": 137, "category": "FIRST_AID", "language": "Hindi", "query": "jalne ka first aid", "fast_routed": True, "description": "Hinglish Burns"},
    {"id": 138, "category": "FIRST_AID", "language": "Telugu", "query": "కాలిన గాయానికి ప్రథమ చికిత్స ఏమిటి", "fast_routed": True, "description": "Telugu Burns"},
    {"id": 139, "category": "FIRST_AID", "language": "Telugu", "query": "kaalina first aid enti", "fast_routed": True, "description": "Telugish Burns"},
    {"id": 140, "category": "FIRST_AID", "language": "Malayalam", "query": "പൊള്ളലേറ്റാൽ എന്ത് ചെയ്യണം", "fast_routed": True, "description": "Malayalam Burns"},
    {"id": 141, "category": "FIRST_AID", "language": "Malayalam", "query": "pollal first aid", "fast_routed": True, "description": "Manglish Burns"},
    {"id": 142, "category": "FIRST_AID", "language": "Kannada", "query": "ಸುಟ್ಟ ಗಾಯಕ್ಕೆ ಪ್ರಥಮ ಚಿಕಿತ್ಸೆ ಏನು", "fast_routed": True, "description": "Kannada Burns"},
    {"id": 143, "category": "FIRST_AID", "language": "Kannada", "query": "sutta gaaya first aid", "fast_routed": True, "description": "Kanglish Burns"},

    # --------------------------------------------------------------------------
    # 12. HEALTH KNOWLEDGE (7 Languages: EN, TA, Tanglish, HI, TE, ML, KN)
    # --------------------------------------------------------------------------
    {"id": 144, "category": "HEALTH_KNOWLEDGE", "language": "English", "query": "how much water should I drink daily", "fast_routed": True, "description": "English Hydration"},
    {"id": 145, "category": "HEALTH_KNOWLEDGE", "language": "English", "query": "headache and migraine remedies", "fast_routed": True, "description": "English Headache"},
    {"id": 146, "category": "HEALTH_KNOWLEDGE", "language": "Tamil", "query": "தினமும் எவ்வளவு தண்ணீர் குடிக்க வேண்டும்", "fast_routed": True, "description": "Tamil Hydration"},
    {"id": 147, "category": "HEALTH_KNOWLEDGE", "language": "Tanglish", "query": "dhinaum evlo thanneer kudikanum", "fast_routed": True, "description": "Tanglish Hydration"},
    {"id": 148, "category": "HEALTH_KNOWLEDGE", "language": "Tanglish", "query": "thalavali poga enna pannanum", "fast_routed": True, "description": "Tanglish Headache"},
    {"id": 149, "category": "HEALTH_KNOWLEDGE", "language": "Hindi", "query": "रोज कितना पानी पीना चाहिए", "fast_routed": True, "description": "Hindi Hydration"},
    {"id": 150, "category": "HEALTH_KNOWLEDGE", "language": "Hindi", "query": "sar dard ke liye kya kare", "fast_routed": True, "description": "Hinglish Headache"},
    {"id": 151, "category": "HEALTH_KNOWLEDGE", "language": "Telugu", "query": "రోజుకు ఎంత నీళ్ళు తాగాలి", "fast_routed": True, "description": "Telugu Hydration"},
    {"id": 152, "category": "HEALTH_KNOWLEDGE", "language": "Telugu", "query": "thalanopi ki emi cheyali", "fast_routed": True, "description": "Telugish Headache"},
    {"id": 153, "category": "HEALTH_KNOWLEDGE", "language": "Malayalam", "query": "ദിവസവും എത്ര വെള്ളം കുടിക്കണം", "fast_routed": True, "description": "Malayalam Hydration"},
    {"id": 154, "category": "HEALTH_KNOWLEDGE", "language": "Malayalam", "query": "thalavedhana maaran enthu cheyyanam", "fast_routed": True, "description": "Manglish Headache"},
    {"id": 155, "category": "HEALTH_KNOWLEDGE", "language": "Kannada", "query": "ದಿನಕ್ಕೆ ಎಷ್ಟು ನೀರು ಕುಡಿಯಬೇಕು", "fast_routed": True, "description": "Kannada Hydration"},
    {"id": 156, "category": "HEALTH_KNOWLEDGE", "language": "Kannada", "query": "thalenovu remedies", "fast_routed": True, "description": "Kanglish Headache"},

    # --------------------------------------------------------------------------
    # 13. BASIC CALCULATOR & MATH / UNIT CONVERSION
    # --------------------------------------------------------------------------
    {"id": 157, "category": "MATH", "language": "English", "query": "what is 25 times 4", "fast_routed": True, "description": "Multiplication English"},
    {"id": 158, "category": "MATH", "language": "English", "query": "100 divided by 5", "fast_routed": True, "description": "Division English"},
    {"id": 159, "category": "MATH", "language": "English", "query": "50 percent of 200", "fast_routed": True, "description": "Percentage English"},
    {"id": 160, "category": "MATH", "language": "English", "query": "square root of 144", "fast_routed": True, "description": "Square root English"},
    {"id": 161, "category": "MATH", "language": "Tanglish", "query": "50 kooda 25 kootu", "fast_routed": True, "description": "Tanglish Addition"},
    {"id": 162, "category": "MATH", "language": "Tanglish", "query": "100 la 20 kali", "fast_routed": True, "description": "Tanglish Subtraction"},
    {"id": 163, "category": "MATH", "language": "Hindi", "query": "25 aur 15 jodo", "fast_routed": True, "description": "Hinglish Addition"},
    {"id": 164, "category": "MATH", "language": "Hindi", "query": "100 se 20 ghatao", "fast_routed": True, "description": "Hinglish Subtraction"},
    {"id": 165, "category": "MATH", "language": "Telugu", "query": "25 mariyu 15 kalapandi", "fast_routed": True, "description": "Telugish Addition"},
    {"id": 166, "category": "MATH", "language": "Malayalam", "query": "25 koottal 15", "fast_routed": True, "description": "Manglish Addition"},
    {"id": 167, "category": "MATH", "language": "Kannada", "query": "25 mathu 15 koodisi", "fast_routed": True, "description": "Kanglish Addition"},
    {"id": 168, "category": "UNIT_CONVERSION", "language": "English", "query": "convert 100 celsius to fahrenheit", "fast_routed": True, "description": "Temperature conversion"},
    {"id": 169, "category": "UNIT_CONVERSION", "language": "English", "query": "convert 10 km to miles", "fast_routed": True, "description": "Distance conversion"},

    # --------------------------------------------------------------------------
    # 14. NOISE & SILENCE DETECTION
    # --------------------------------------------------------------------------
    {"id": 170, "category": "NOISE", "language": "Any", "query": "", "fast_routed": True, "description": "Empty input"},
    {"id": 171, "category": "NOISE", "language": "Any", "query": "...", "fast_routed": True, "description": "Pause dots"},
    {"id": 172, "category": "NOISE", "language": "Any", "query": "[inaudible]", "fast_routed": True, "description": "ASR inaudible tag"},
    {"id": 173, "category": "NOISE", "language": "Any", "query": "uhh", "fast_routed": True, "description": "Hesitation sound"},

    # --------------------------------------------------------------------------
    # 15. UNKNOWN / GENERAL LLM QUERIES (MUST FORWARD TO LLM -> return None)
    # --------------------------------------------------------------------------
    {"id": 174, "category": "GENERAL_LLM", "language": "English", "query": "tell me a story about space exploration", "fast_routed": False, "description": "Creative story EN"},
    {"id": 175, "category": "GENERAL_LLM", "language": "English", "query": "explain quantum computing in simple terms", "fast_routed": False, "description": "Complex science EN"},
    {"id": 176, "category": "GENERAL_LLM", "language": "English", "query": "write a python script for binary search", "fast_routed": False, "description": "Coding request EN"},
    {"id": 177, "category": "GENERAL_LLM", "language": "Tamil", "query": "எனக்கு ஒரு சிறுகதை சொல்", "fast_routed": False, "description": "Creative story TA"},
    {"id": 178, "category": "GENERAL_LLM", "language": "Tamil", "query": "பைதான் நிரலாக்கம் பற்றி விளக்கு", "fast_routed": False, "description": "Coding explanation TA"},
    {"id": 179, "category": "GENERAL_LLM", "language": "Hindi", "query": "मुझे एक कहानी सुनाओ", "fast_routed": False, "description": "Creative story HI"},
    {"id": 180, "category": "GENERAL_LLM", "language": "Hindi", "query": "मशीन लर्निंग क्या है समझाइए", "fast_routed": False, "description": "Tech explanation HI"},
    {"id": 181, "category": "GENERAL_LLM", "language": "Telugu", "query": "నాకు ఒక మంచి కథ చెప్పు", "fast_routed": False, "description": "Creative story TE"},
    {"id": 182, "category": "GENERAL_LLM", "language": "Malayalam", "query": "എനിക്ക് ഒരു കഥ പറഞ്ഞു തരൂ", "fast_routed": False, "description": "Creative story ML"},
    {"id": 183, "category": "GENERAL_LLM", "language": "Kannada", "query": "ನನಗೆ ಒಂದು ಕಥೆ ಹೇಳು", "fast_routed": False, "description": "Creative story KN"},
]


def get_testset() -> List[Dict[str, Any]]:
    """Return the complete list of multilingual judge benchmark test questions."""
    return TEST_QUESTIONS


def export_testset_json(output_path: str = "data/judge_testset.json") -> str:
    """Export the testset as a formatted JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(TEST_QUESTIONS, f, indent=2, ensure_ascii=False)
    return output_path


def export_testset_csv(output_path: str = "data/judge_testset.csv") -> str:
    """Export the testset as a standard CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "language", "query", "fast_routed", "description"])
        writer.writeheader()
        writer.writerows(TEST_QUESTIONS)
    return output_path


def run_benchmark() -> None:
    """Execute all test questions against `route_intent` and print benchmark metrics."""
    print("=" * 88)
    print(" HACKNEX 2026 - JARVIS-EDGE MULTILINGUAL FAST-ROUTER BENCHMARK")
    print("=" * 88)

    fast_latencies = []
    correct_routing = 0
    total = len(TEST_QUESTIONS)
    failures = []

    print(f"\nRunning {total} test questions across 7 languages (English, Tamil, Tanglish, Hindi, Telugu, Malayalam, Kannada)...\n")
    print(f"{'ID':<4} | {'CATEGORY':<16} | {'LANG':<10} | {'FAST?':<6} | {'TIME(µs)':<9} | {'STATUS':<7} | {'QUERY'}")
    print("-" * 88)

    for item in TEST_QUESTIONS:
        q = item["query"]
        expected_fast = item["fast_routed"]

        # Measure latency in microseconds
        t0 = time.perf_counter_ns()
        response = route_intent(q)
        t1 = time.perf_counter_ns()

        duration_us = (t1 - t0) / 1000.0  # microseconds
        fast_latencies.append(duration_us)

        was_handled = response is not None
        is_correct = (was_handled == expected_fast)

        if is_correct:
            correct_routing += 1
            status = "PASS"
        else:
            status = "FAIL"
            failures.append((item, response))

        print(f"{item['id']:<4} | {item['category']:<16} | {item['language']:<10} | {str(was_handled):<6} | {duration_us:8.1f} | {status:<7} | \"{q}\"")

    avg_latency_us = sum(fast_latencies) / len(fast_latencies)
    avg_latency_ms = avg_latency_us / 1000.0
    max_latency_us = max(fast_latencies)
    max_latency_ms = max_latency_us / 1000.0
    accuracy = (correct_routing / total) * 100.0

    print("-" * 88)
    print("BENCHMARK SUMMARY RESULTS:")
    print(f"  Total Questions Tested  : {total}")
    print(f"  Passed Tests            : {correct_routing}")
    print(f"  Failed Tests            : {len(failures)}")
    print(f"  Routing Accuracy        : {accuracy:.1f}% ({correct_routing}/{total})")
    print(f"  Average Router Latency  : {avg_latency_us:.1f} µs ({avg_latency_ms:.4f} ms)")
    print(f"  Max Router Latency      : {max_latency_us:.1f} µs ({max_latency_ms:.4f} ms)")
    print(f"  Target Latency (<5ms)   : {'EXCEEDED (Lightning Fast!)' if max_latency_ms < 5.0 else 'FAILED'}")
    print("=" * 88)

    if failures:
        print("\nFAILURE DETAILS:")
        for item, res in failures:
            print(f"  Item #{item['id']} ({item['category']}/{item['language']}): query='{item['query']}', expected_fast={item['fast_routed']}, actual_response={res}")


if __name__ == "__main__":
    json_path = export_testset_json()
    csv_path = export_testset_csv()
    print(f"Exported testsets to:\n  - {json_path}\n  - {csv_path}\n")
    run_benchmark()
