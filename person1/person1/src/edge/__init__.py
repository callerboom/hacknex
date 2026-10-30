"""Edge package for offline fast-path routing and local operations."""

from src.edge.knowledge_search import search_offline_knowledge
from src.edge.router import route_intent
from src.edge.voice_pipeline import VoicePipeline, execute_voice_command

__all__ = [
    "route_intent",
    "search_offline_knowledge",
    "VoicePipeline",
    "execute_voice_command",
]
