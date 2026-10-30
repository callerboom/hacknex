"""
Offline Vision Engine for EdgeVoice OS (HNX26EPS08).
Primary: RapidOCR (ONNX, CPU-only, offline) — reads text from images.
Fallback: PIL metadata (format, dimensions, mode, EXIF).
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("vision")

_ocr_engine = None


def _get_ocr():
    """Lazy-load RapidOCR engine (models auto-download once on first use)."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
            logger.info("[Vision] RapidOCR engine loaded.")
        except Exception as e:
            logger.warning(f"[Vision] OCR unavailable: {e}")
            _ocr_engine = False
    return _ocr_engine or None


def _describe_metadata(path: Path) -> str:
    """PIL-based basic image info."""
    try:
        from PIL import Image as PILImage
        import PIL.ExifTags as ExifTags

        with PILImage.open(path) as img:
            w, h = img.size
            mode = img.mode
            fmt = img.format or "unknown"
            desc = f"Image: {fmt} format, {w}x{h}, mode {mode}"

            exif_data = []
            try:
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        if tag_name in ("DateTimeOriginal", "Make", "Model"):
                            exif_data.append(f"{tag_name}: {value}")
            except Exception:
                pass

            if exif_data:
                desc += ". Metadata: " + ", ".join(exif_data)
            return desc
    except Exception as e:
        return f"Image analysis failed: {e}"


def describe(image_path: str | Path) -> Dict[str, Any]:
    """
    Examine image_path offline and return description / extracted text.
    Tries OCR (text reading) first; falls back to metadata.
    """
    path = Path(image_path)
    if not path.exists():
        return {"text": f"Image not found: {path.name}", "method": "error", "ms": 0.0}

    t0 = time.perf_counter()
    ocr = _get_ocr()

    if ocr is not None:
        try:
            result, _ = ocr(str(path))
            if result:
                lines = []
                for box, text, conf in result:
                    if text and text.strip():
                        lines.append(f"{text.strip()} ({conf:.0%})")
                if lines:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    return {
                        "text": "Text found in image: " + " | ".join(lines),
                        "method": "ocr",
                        "ms": round(elapsed_ms, 2),
                    }
                else:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    return {
                        "text": "No text detected in image. " + _describe_metadata(path),
                        "method": "ocr",
                        "ms": round(elapsed_ms, 2),
                    }
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return {
                    "text": "No text detected in image. " + _describe_metadata(path),
                    "method": "ocr",
                    "ms": round(elapsed_ms, 2),
                }
        except Exception as e:
            logger.warning(f"[Vision] OCR failed: {e} — falling back to metadata")

    # Fallback: metadata only
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "text": _describe_metadata(path),
        "method": "metadata",
        "ms": round(elapsed_ms, 2),
    }


def unload_models() -> None:
    """Release OCR engine to reclaim memory."""
    global _ocr_engine
    _ocr_engine = None
    import gc
    gc.collect()
