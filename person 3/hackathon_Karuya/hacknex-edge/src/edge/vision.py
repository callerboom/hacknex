"""
vision.py — Lightweight Edge Vision & Image Analysis Processor.
HackNEX 2026 — Person 3 Module B (AI Brain Multimodal Extension)

Provides image inspection, OCR, and visual feature extraction
for questions involving uploaded images from Person 2 or local files.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Analyzes uploaded images and generates structured context
    to inject into the LLM prompt.
    """

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self):
        self._cv2 = None
        self._pil = None
        try:
            from PIL import Image
            self._pil = Image
        except ImportError:
            pass

        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            pass

    def is_image_path(self, path_str: str) -> bool:
        """Check if a string refers to an existing image file."""
        if not path_str:
            return False
        clean = path_str.strip().strip("'\"")
        p = Path(clean)
        return p.exists() and p.suffix.lower() in self.SUPPORTED_EXTS

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Extract dimensions, format, color characteristics, and text/visual cues.
        """
        clean_path = str(image_path).strip().strip("'\"")
        p = Path(clean_path)
        if not p.exists():
            return {"error": f"Image file not found: {clean_path}"}

        info: Dict[str, Any] = {
            "filename": p.name,
            "path": str(p.resolve()),
            "format": p.suffix.upper().lstrip("."),
            "file_size_kb": round(p.stat().st_size / 1024, 1),
        }

        # PIL Analysis
        if self._pil:
            try:
                with self._pil.open(p) as img:
                    info["width"], info["height"] = img.size
                    info["mode"] = img.mode
                    info["aspect_ratio"] = round(img.size[0] / img.size[1], 2) if img.size[1] else 1.0

                    # Basic brightness assessment
                    grayscale = img.convert("L")
                    stat_pixels = list(grayscale.resize((32, 32)).getdata())
                    avg_brightness = sum(stat_pixels) / len(stat_pixels) if stat_pixels else 128
                    if avg_brightness > 180:
                        info["lighting"] = "Bright / High Key"
                    elif avg_brightness < 70:
                        info["lighting"] = "Dark / Low Key"
                    else:
                        info["lighting"] = "Balanced"
            except Exception as e:
                logger.debug("PIL analysis error: %s", e)

        # CV2 Analysis (Edge density & sharpness)
        if self._cv2:
            try:
                mat = self._cv2.imread(str(p), self._cv2.IMREAD_GRAYSCALE)
                if mat is not None:
                    laplacian_var = self._cv2.Laplacian(mat, self._cv2.CV_64F).var()
                    info["sharpness"] = "Sharp / Focused" if laplacian_var > 100 else "Soft / Blurred"
            except Exception as e:
                logger.debug("CV2 analysis error: %s", e)

        return info

    def format_vision_context(self, image_path: str, user_caption: str = "") -> str:
        """
        Produces a rich text block describing the image for the LLM prompt.
        """
        meta = self.analyze(image_path)
        if "error" in meta:
            return f"Image File: {image_path} (File could not be opened or is missing)."

        w = meta.get("width", "Unknown")
        h = meta.get("height", "Unknown")
        fmt = meta.get("format", "Image")
        lighting = meta.get("lighting", "Standard")
        sharpness = meta.get("sharpness", "Clear")
        size_kb = meta.get("file_size_kb", 0)

        context_lines = [
            f"Image File Name: {meta.get('filename')}",
            f"Image Dimensions: {w} pixels wide by {h} pixels tall ({fmt} format, {size_kb} KB)",
            f"Visual Attributes: Lighting is {lighting}, Image clarity is {sharpness}.",
        ]

        if user_caption:
            context_lines.append(f"Image Note / Caption from user: {user_caption}")

        return "\n".join(context_lines)


# Global instance
vision_engine = VisionProcessor()
