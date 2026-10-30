"""Offline Markdown Knowledge Base Search Engine.

Indexes markdown documents in `data/knowledge/` into concise section chunks
and provides sub-millisecond keyword and semantic tag matching for instant
multilingual offline guidance across:
- English, Tamil (தமிழ் / Tanglish), Hindi (हिंदी / Hinglish),
  Telugu (తెలుగు / Telugish), Malayalam (മലയാളം / Manglish), Kannada (ಕನ್ನಡ / Kanglish).
"""

import os
import re
from typing import Any, Dict, List, Optional

_KNOWLEDGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge")
)

# Comprehensive multilingual topic keywords mapping directly to sections
_MULTILINGUAL_TOPIC_MAP = {
    # 1. Burns & Scalds
    "burn": "Burns & Scalds Treatment",
    "scald": "Burns & Scalds Treatment",
    "theekaayam": "Burns & Scalds Treatment",
    "theepollal": "Burns & Scalds Treatment",
    "thee": "Burns & Scalds Treatment",
    "jalne": "Burns & Scalds Treatment",
    "jala": "Burns & Scalds Treatment",
    "kaalina": "Burns & Scalds Treatment",
    "kaalithe": "Burns & Scalds Treatment",
    "suttagaaya": "Burns & Scalds Treatment",
    "suttare": "Burns & Scalds Treatment",
    "sutta": "Burns & Scalds Treatment",
    "pollal": "Burns & Scalds Treatment",
    "pollalettal": "Burns & Scalds Treatment",
    "தீக்காய": "Burns & Scalds Treatment",
    "தீக்காயம்": "Burns & Scalds Treatment",
    "தீப்பொള്ളல்": "Burns & Scalds Treatment",
    "தீ காயம்": "Burns & Scalds Treatment",
    "जलने": "Burns & Scalds Treatment",
    "जलन": "Burns & Scalds Treatment",
    "कालिन": "Burns & Scalds Treatment",
    "కాలిన": "Burns & Scalds Treatment",
    "కాలితే": "Burns & Scalds Treatment",
    "പൊള്ളൽ": "Burns & Scalds Treatment",
    "പൊള്ളലേറ്റാൽ": "Burns & Scalds Treatment",
    "ಸುಟ್ಟ": "Burns & Scalds Treatment",
    "ಸುಟ್ಟರೆ": "Burns & Scalds Treatment",
    "ಸುಟ್ಟ ಗಾಯ": "Burns & Scalds Treatment",

    # 2. CPR & Heart Attack
    "cpr": "Cardiopulmonary Resuscitation (CPR)",
    "chest compression": "Cardiopulmonary Resuscitation (CPR)",
    "heart attack": "Cardiopulmonary Resuscitation (CPR)",
    "ithaya": "Cardiopulmonary Resuscitation (CPR)",
    "ithaya thudippu": "Cardiopulmonary Resuscitation (CPR)",
    "dil ka daura": "Cardiopulmonary Resuscitation (CPR)",
    "gunde aagipothe": "Cardiopulmonary Resuscitation (CPR)",
    "hrudayaghatham": "Cardiopulmonary Resuscitation (CPR)",
    "hrudayaghatha": "Cardiopulmonary Resuscitation (CPR)",
    "சிபிஆர்": "Cardiopulmonary Resuscitation (CPR)",
    "இதய துடிப்பு": "Cardiopulmonary Resuscitation (CPR)",
    "இதய": "Cardiopulmonary Resuscitation (CPR)",
    "सीपीआर": "Cardiopulmonary Resuscitation (CPR)",
    "दिल का दौरा": "Cardiopulmonary Resuscitation (CPR)",
    "సిపిఆర్": "Cardiopulmonary Resuscitation (CPR)",
    "గుండెపోటు": "Cardiopulmonary Resuscitation (CPR)",
    "സിപിആർ": "Cardiopulmonary Resuscitation (CPR)",
    "ഹൃദയാഘാതം": "Cardiopulmonary Resuscitation (CPR)",
    "ಸಿಪಿಆರ್": "Cardiopulmonary Resuscitation (CPR)",
    "ಹೃದಯಾಘಾತ": "Cardiopulmonary Resuscitation (CPR)",

    # 3. Choking
    "choking": "Choking Management",
    "choke": "Choking Management",
    "heimlich": "Choking Management",
    "thondai": "Choking Management",
    "adappu": "Choking Management",
    "gale": "Choking Management",
    "gala": "Choking Management",
    "gonthu": "Choking Management",
    "gonthulo": "Choking Management",
    "thonda": "Choking Management",
    "thondayil": "Choking Management",
    "gantalu": "Choking Management",
    "gantallu": "Choking Management",
    "தொண்டை": "Choking Management",
    "அடைப்பு": "Choking Management",
    "गले": "Choking Management",
    "गला": "Choking Management",
    "గొంతు": "Choking Management",
    "గొంతులో": "Choking Management",
    "തൊണ്ട": "Choking Management",
    "തൊണ്ടയിൽ": "Choking Management",
    "ಗಂಟಲು": "Choking Management",
    "ಗಂಟಲಿನಲ್ಲಿ": "Choking Management",

    # 4. Snake Bites
    "snake": "Snake Bites & Venous Envenomation",
    "snakebite": "Snake Bites & Venous Envenomation",
    "paambu": "Snake Bites & Venous Envenomation",
    "saamp": "Snake Bites & Venous Envenomation",
    "saap": "Snake Bites & Venous Envenomation",
    "paamu": "Snake Bites & Venous Envenomation",
    "haavu": "Snake Bites & Venous Envenomation",
    "paampu": "Snake Bites & Venous Envenomation",
    "பாம்பு": "Snake Bites & Venous Envenomation",
    "சாம்பு": "Snake Bites & Venous Envenomation",
    "सांप": "Snake Bites & Venous Envenomation",
    "साँप": "Snake Bites & Venous Envenomation",
    "पामु": "Snake Bites & Venous Envenomation",
    "పాము": "Snake Bites & Venous Envenomation",
    "പാമ്പ്": "Snake Bites & Venous Envenomation",
    "ಹಾವು": "Snake Bites & Venous Envenomation",

    # 5. Severe Bleeding
    "bleeding": "Severe Bleeding & Hemorrhage Control",
    "bleed": "Severe Bleeding & Hemorrhage Control",
    "ratham": "Severe Bleeding & Hemorrhage Control",
    "ratha": "Severe Bleeding & Hemorrhage Control",
    "khoon": "Severe Bleeding & Hemorrhage Control",
    "raktham": "Severe Bleeding & Hemorrhage Control",
    "raktha": "Severe Bleeding & Hemorrhage Control",
    "chora": "Severe Bleeding & Hemorrhage Control",
    "tourniquet": "Severe Bleeding & Hemorrhage Control",
    "ரத்தம்": "Severe Bleeding & Hemorrhage Control",
    "இரத்தம்": "Severe Bleeding & Hemorrhage Control",
    "ரத்தப்போக்கு": "Severe Bleeding & Hemorrhage Control",
    "खून": "Severe Bleeding & Hemorrhage Control",
    "रक्त": "Severe Bleeding & Hemorrhage Control",
    "रक्तस्राव": "Severe Bleeding & Hemorrhage Control",
    "రక్తం": "Severe Bleeding & Hemorrhage Control",
    "రక్తస్రావం": "Severe Bleeding & Hemorrhage Control",
    "രക്തം": "Severe Bleeding & Hemorrhage Control",
    "രക്തസ്രാവം": "Severe Bleeding & Hemorrhage Control",
    "ರಕ್ತ": "Severe Bleeding & Hemorrhage Control",
    "ರಕ್ತಸ್ರಾವ": "Severe Bleeding & Hemorrhage Control",

    # 6. Sprains & Fractures
    "sprain": "Fractures, Sprains & Strains (R.I.C.E.)",
    "fracture": "Fractures, Sprains & Strains (R.I.C.E.)",
    "strain": "Fractures, Sprains & Strains (R.I.C.E.)",
    "elumbu": "Fractures, Sprains & Strains (R.I.C.E.)",
    "sulukku": "Fractures, Sprains & Strains (R.I.C.E.)",
    "haddi": "Fractures, Sprains & Strains (R.I.C.E.)",
    "moch": "Fractures, Sprains & Strains (R.I.C.E.)",
    "emuka": "Fractures, Sprains & Strains (R.I.C.E.)",
    "benuku": "Fractures, Sprains & Strains (R.I.C.E.)",
    "eluka": "Fractures, Sprains & Strains (R.I.C.E.)",
    "ellu": "Fractures, Sprains & Strains (R.I.C.E.)",
    "ulukku": "Fractures, Sprains & Strains (R.I.C.E.)",
    "moole": "Fractures, Sprains & Strains (R.I.C.E.)",
    "moolu": "Fractures, Sprains & Strains (R.I.C.E.)",
    "uluku": "Fractures, Sprains & Strains (R.I.C.E.)",
    "எலும்பு": "Fractures, Sprains & Strains (R.I.C.E.)",
    "சுளுக்கு": "Fractures, Sprains & Strains (R.I.C.E.)",
    "हड्डी": "Fractures, Sprains & Strains (R.I.C.E.)",
    "मोच": "Fractures, Sprains & Strains (R.I.C.E.)",
    "ఎముక": "Fractures, Sprains & Strains (R.I.C.E.)",
    "బెణుకు": "Fractures, Sprains & Strains (R.I.C.E.)",
    "എല്ല്": "Fractures, Sprains & Strains (R.I.C.E.)",
    "ഉളുക്ക്": "Fractures, Sprains & Strains (R.I.C.E.)",
    "ಮೂಳೆ": "Fractures, Sprains & Strains (R.I.C.E.)",
    "ಉಳುಕು": "Fractures, Sprains & Strains (R.I.C.E.)",

    # 7. Seizures
    "seizure": "Seizures & Convulsions",
    "convulsion": "Seizures & Convulsions",
    "fits": "Seizures & Convulsions",
    "valippu": "Seizures & Convulsions",
    "daure": "Seizures & Convulsions",
    "mirgi": "Seizures & Convulsions",
    "murcha": "Seizures & Convulsions",
    "apasmaram": "Seizures & Convulsions",
    "moorche": "Seizures & Convulsions",
    "வலிப்பு": "Seizures & Convulsions",
    "दौरे": "Seizures & Convulsions",
    "मिर्गी": "Seizures & Convulsions",
    "మూర్ఛ": "Seizures & Convulsions",
    "അപസ്മാരം": "Seizures & Convulsions",
    "ಮೂರ್ಛೆ": "Seizures & Convulsions",

    # 8. Anaphylaxis
    "anaphylaxis": "Anaphylaxis (Severe Allergic Reaction)",
    "epipen": "Anaphylaxis (Severe Allergic Reaction)",
    "allergy": "Anaphylaxis (Severe Allergic Reaction)",
    "ovvaamai": "Anaphylaxis (Severe Allergic Reaction)",
    "ஒவ்வாமை": "Anaphylaxis (Severe Allergic Reaction)",
    "அலர்ஜி": "Anaphylaxis (Severe Allergic Reaction)",
    "एलर्जी": "Anaphylaxis (Severe Allergic Reaction)",
    "అలెర్జీ": "Anaphylaxis (Severe Allergic Reaction)",
    "അലർജി": "Anaphylaxis (Severe Allergic Reaction)",
    "ಅಲರ್ಜಿ": "Anaphylaxis (Severe Allergic Reaction)",

    # 9. Hydration
    "hydration": "Daily Hydration Guidelines",
    "water": "Daily Hydration Guidelines",
    "drink": "Daily Hydration Guidelines",
    "thanneer": "Daily Hydration Guidelines",
    "thanni": "Daily Hydration Guidelines",
    "kudineer": "Daily Hydration Guidelines",
    "paani": "Daily Hydration Guidelines",
    "pani": "Daily Hydration Guidelines",
    "jal": "Daily Hydration Guidelines",
    "neellu": "Daily Hydration Guidelines",
    "vellam": "Daily Hydration Guidelines",
    "kudivellam": "Daily Hydration Guidelines",
    "neeru": "Daily Hydration Guidelines",
    "தண்ணீர்": "Daily Hydration Guidelines",
    "குடிநீர்": "Daily Hydration Guidelines",
    "पानी": "Daily Hydration Guidelines",
    "जल": "Daily Hydration Guidelines",
    "నీళ్ళు": "Daily Hydration Guidelines",
    "మంచి నీళ్ళు": "Daily Hydration Guidelines",
    "വെള്ളം": "Daily Hydration Guidelines",
    "കുടിവെള്ളം": "Daily Hydration Guidelines",
    "ನೀರು": "Daily Hydration Guidelines",
    "ಕುಡಿಯುವ ನೀರು": "Daily Hydration Guidelines",

    # 10. Headache
    "headache": "Headache & Migraine Relief",
    "migraine": "Headache & Migraine Relief",
    "thalavali": "Headache & Migraine Relief",
    "mandai vali": "Headache & Migraine Relief",
    "sar dard": "Headache & Migraine Relief",
    "sir dard": "Headache & Migraine Relief",
    "thalanopi": "Headache & Migraine Relief",
    "thalanoppi": "Headache & Migraine Relief",
    "thalavedhana": "Headache & Migraine Relief",
    "thalavedana": "Headache & Migraine Relief",
    "thalenovu": "Headache & Migraine Relief",
    "தலைவலி": "Headache & Migraine Relief",
    "ஒற்றைத் தலைவலி": "Headache & Migraine Relief",
    "सिर दर्द": "Headache & Migraine Relief",
    "माइग्रेन": "Headache & Migraine Relief",
    "తలనొప్పి": "Headache & Migraine Relief",
    "తల నొప్పి": "Headache & Migraine Relief",
    "തലവേദന": "Headache & Migraine Relief",
    "തല വേദന": "Headache & Migraine Relief",
    "മൈഗ്രെയ്ൻ": "Headache & Migraine Relief",
    "ತಲೆನೋವು": "Headache & Migraine Relief",
    "ತಲೆ ನೋವು": "Headache & Migraine Relief",

    # 11. Fever
    "fever": "Fever Management & Comfort",
    "kaichal": "Fever Management & Comfort",
    "kaachal": "Fever Management & Comfort",
    "bukhar": "Fever Management & Comfort",
    "jwaram": "Fever Management & Comfort",
    "jvaram": "Fever Management & Comfort",
    "jwara": "Fever Management & Comfort",
    "jvara": "Fever Management & Comfort",
    "காய்ச்சல்": "Fever Management & Comfort",
    "बुखार": "Fever Management & Comfort",
    "ज्वर": "Fever Management & Comfort",
    "జ్వరం": "Fever Management & Comfort",
    "പനി": "Fever Management & Comfort",
    "ಜ್ವರ": "Fever Management & Comfort",

    # 12. Sleep
    "sleep": "Sleep Hygiene & Quality Rest",
    "thookam": "Sleep Hygiene & Quality Rest",
    "neend": "Sleep Hygiene & Quality Rest",
    "nidra": "Sleep Hygiene & Quality Rest",
    "urakkam": "Sleep Hygiene & Quality Rest",
    "nidre": "Sleep Hygiene & Quality Rest",
    "nidde": "Sleep Hygiene & Quality Rest",
    "தூக்கம்": "Sleep Hygiene & Quality Rest",
    "नींद": "Sleep Hygiene & Quality Rest",
    "నిద్ర": "Sleep Hygiene & Quality Rest",
    "ഉറക്കം": "Sleep Hygiene & Quality Rest",
    "ನಿದ್ರೆ": "Sleep Hygiene & Quality Rest",

    # 13. Ergonomics
    "ergonomics": "Desk Ergonomics & Posture",
    "posture": "Desk Ergonomics & Posture",
    "20-20-20": "Desk Ergonomics & Posture",
    "eye strain": "Desk Ergonomics & Posture",
    "தோரணை": "Desk Ergonomics & Posture",
    "मुद्रा": "Desk Ergonomics & Posture",
    "భంగిమ": "Desk Ergonomics & Posture",
    "ഇരിപ്പ്": "Desk Ergonomics & Posture",
    "ಭಂಗಿ": "Desk Ergonomics & Posture",
}


class KnowledgeSearchEngine:
    """Fast, in-memory indexed offline markdown knowledge retrieval engine."""

    def __init__(self, knowledge_dir: str = _KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self.sections: List[Dict[str, Any]] = []
        self._load_and_index()

    def _load_and_index(self) -> None:
        """Parse and index markdown headings and bullet points into memory."""
        self.sections.clear()
        if not os.path.exists(self.knowledge_dir):
            return

        for fname in sorted(os.listdir(self.knowledge_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(self.knowledge_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                self._parse_markdown_file(fname, content)
            except Exception:
                pass

    def _parse_markdown_file(self, filename: str, content: str) -> None:
        """Split markdown content by Level 2 and Level 3 headers."""
        raw_sections = re.split(r"\n(?=##\s+)", content)
        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean.startswith("##"):
                continue

            lines = sec_clean.splitlines()
            title_line = lines[0].lstrip("#").strip()
            clean_title = re.sub(r"^\d+[\.\)]\s*", "", title_line).strip()

            body = "\n".join(lines[1:]).strip()
            keywords = set(re.findall(r"\b\w{3,}\b", (clean_title + " " + body).lower(), flags=re.UNICODE))

            self.sections.append({
                "source": filename,
                "title": clean_title,
                "raw_title": title_line,
                "body": body,
                "keywords": keywords,
            })

    def search(self, query: str, min_score: float = 0.5) -> Optional[str]:
        """Search indexed offline knowledge and return a clean, concise guidance summary."""
        query_clean = query.strip().lower()

        # 1. Match on Multilingual Topic Dictionary (sorted by longest keyword first to prevent partial match issues)
        sorted_topics = sorted(_MULTILINGUAL_TOPIC_MAP.items(), key=lambda x: len(x[0]), reverse=True)
        for topic_key, mapped_title in sorted_topics:
            if topic_key in query_clean:
                for sec in self.sections:
                    if mapped_title.lower() in sec["title"].lower() or sec["title"].lower() in mapped_title.lower():
                        return self._format_section_response(sec)

        # 2. General Keyword Overlap
        query_words = set(re.findall(r"\b\w{3,}\b", query_clean, flags=re.UNICODE))
        if not query_words:
            return None

        best_section = None
        best_score = 0.0

        for sec in self.sections:
            title_lower = sec["title"].lower()

            if any(word in title_lower for word in query_words):
                title_overlap = sum(1 for w in query_words if w in title_lower)
                score = 2.0 + (title_overlap / len(query_words))
            else:
                overlap = query_words.intersection(sec["keywords"])
                score = len(overlap) / len(query_words)

            if score > best_score:
                best_score = score
                best_section = sec

        if best_section and best_score >= min_score:
            return self._format_section_response(best_section)

        return None

    def _format_section_response(self, section: Dict[str, Any]) -> str:
        """Format raw markdown section text into clean, speech-friendly guidance."""
        lines = [l.strip() for l in section["body"].splitlines() if l.strip()]
        summary_bullets = []
        for l in lines:
            if l.startswith("- ") or l.startswith("1.") or l.startswith("2.") or l.startswith("3."):
                summary_bullets.append(l)
            if len(summary_bullets) >= 4:
                break

        guidance_text = " ".join(summary_bullets) if summary_bullets else lines[0] if lines else ""
        clean_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", guidance_text)
        clean_text = re.sub(r"\*([^*]+)\*", r"\1", clean_text)
        clean_text = re.sub(r"^[0-9\.\-\s]+", "", clean_text)

        return f"[{section['title']}]: {clean_text}"


# Global singleton index
_GLOBAL_ENGINE = KnowledgeSearchEngine()


def search_offline_knowledge(query: str) -> Optional[str]:
    """Retrieve fast offline knowledge snippet matching the query."""
    return _GLOBAL_ENGINE.search(query)
