"""OSD OCR for prescan. EasyOCR is optional; tests inject text hits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.util import find_spec
from typing import Any

from viana.stages.time_map import ParsedOcr, parse_ocr_texts

OcrReader = Callable[[object], Sequence[tuple[str, float]]]


def filter_ocr_hits(
    hits: Sequence[tuple[str, float]],
    min_confidence: float,
) -> tuple[list[str], float | None]:
    """Keep strings at or above ``min_confidence``; return mean confidence."""
    kept: list[tuple[str, float]] = []
    for text, confidence in hits:
        if confidence >= min_confidence:
            kept.append((text, confidence))
    if not kept:
        return [], None
    texts = [item[0] for item in kept]
    mean = sum(item[1] for item in kept) / len(kept)
    return texts, mean


def parse_osd_hits(
    hits: Sequence[tuple[str, float]],
    min_confidence: float,
) -> tuple[ParsedOcr, float | None]:
    """Map EasyOCR-style (text, prob) hits into structured OSD fields."""
    texts, mean = filter_ocr_hits(hits, min_confidence)
    return parse_ocr_texts(texts), mean


def easyocr_reader(gpu: bool = False) -> OcrReader:
    """Build a full-frame EasyOCR reader (Phase 4 production path)."""
    reader_box: list[Any] = []

    def _read(frame: object) -> list[tuple[str, float]]:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("easyocr is required for live OSD OCR") from exc
        if not reader_box:
            reader_box.append(easyocr.Reader(["en"], gpu=gpu, verbose=False))
        reader = reader_box[0]
        results = reader.readtext(frame)
        hits: list[tuple[str, float]] = []
        for item in results:
            if len(item) < 3:
                continue
            hits.append((str(item[1]), float(item[2])))
        return hits

    return _read


def optional_easyocr_reader(gpu: bool = False) -> OcrReader:
    """Use EasyOCR when installed; otherwise return no hits so prescan still runs."""
    if find_spec("easyocr") is None:
        return lambda _frame: []
    return easyocr_reader(gpu=gpu)
