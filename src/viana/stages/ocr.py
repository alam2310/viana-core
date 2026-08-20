"""OSD OCR for prescan. EasyOCR is optional; tests inject text hits."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, TypeGuard

from viana.stages.time_map import (
    ParsedOcr,
    extract_ocr_time,
    is_plausible_ocr_date,
    parse_location_texts,
    parse_metadata_texts,
)

OcrReader = Callable[[object], Sequence[tuple[str, float]]]


def _as_bgr_array(frame: object) -> Any:
    """OpenCV BGR ndarray; ``Any`` because numpy is optional at type-check time."""
    return frame


@dataclass(frozen=True)
class OsdRoi:
    """Fractional corner crop for on-screen metadata."""

    name: str
    y_start: float
    y_end: float
    x_start: float
    x_end: float
    allowlist: str | None = None


# Top-left: date/time/day. Bottom-left: camera location label.
# Fast path uses a tighter metadata crop at 2× so CRAFT runs on fewer pixels
# (S08). Wide 4× corners remain the accuracy fallback when time/date are missing.
# S21: full-width top/bottom bands run only when required fields are still empty.
_LOCATION_ROI = OsdRoi(
    "location",
    0.86,
    1.0,
    0.0,
    0.32,
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
)
DEFAULT_OSD_ROIS: tuple[OsdRoi, OsdRoi] = (
    OsdRoi("metadata", 0.0, 0.07, 0.0, 0.45, None),
    _LOCATION_ROI,
)
WIDE_OSD_ROIS: tuple[OsdRoi, OsdRoi] = (
    OsdRoi("metadata", 0.0, 0.06, 0.0, 0.58, None),
    _LOCATION_ROI,
)
BAND_OSD_ROIS: tuple[OsdRoi, OsdRoi] = (
    OsdRoi("band_top", 0.0, 0.14, 0.0, 1.0, None),
    OsdRoi("band_bottom", 0.86, 1.0, 0.0, 1.0, None),
)
OSD_SCORE_ROIS: tuple[OsdRoi, ...] = (
    DEFAULT_OSD_ROIS[0],
    OsdRoi("score_top_center", 0.0, 0.14, 0.25, 0.75, None),
    OsdRoi("score_bottom_left", 0.86, 1.0, 0.0, 0.50, None),
)

OCR_ROI_SCALE = 2.0
OCR_FALLBACK_SCALE = 4.0
OCR_BAND_SCALE = 2.0


class CornerOsdReader:
    """EasyOCR reader that crops corner ROIs instead of scanning the full frame."""

    corner_osd = True

    def __init__(self, gpu: bool = False) -> None:
        self._gpu = gpu
        self._reader: Any | None = None

    def _ensure_reader(self) -> Any:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
        return self._reader

    def __call__(self, frame: object) -> list[tuple[str, float]]:
        metadata_hits, location_hits = read_corner_osd_hits_with_fallback(
            frame, self._ensure_reader()
        )
        return metadata_hits + location_hits

    def parse(self, frame: object, min_confidence: float) -> tuple[ParsedOcr, float | None]:
        """OCR corner ROIs and return structured metadata."""
        return parse_frame_corner_osd(frame, self._ensure_reader(), min_confidence)


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


def _usable_location(text: str | None) -> TypeGuard[str]:
    """Reject OCR leftovers that are too short or only a clock fragment."""
    if text is None:
        return False
    cleaned = text.strip()
    if len(cleaned) < 4:
        return False
    compact = cleaned.replace(" ", "")
    if compact.isdigit():
        return False
    if re.fullmatch(r"[A-Za-z]{1,3}\s+\d{4,}", cleaned):
        return False
    compact_alnum = re.sub(r"[^\w]", "", cleaned)
    digits = sum(ch.isdigit() for ch in compact_alnum)
    letters = sum(ch.isalpha() for ch in compact_alnum)
    if digits >= 4 and digits > letters:
        return False
    return True


def _preferred_location(*candidates: str | None) -> str | None:
    usable = [item for item in candidates if _usable_location(item)]
    if not usable:
        return next((item for item in candidates if item), None)
    return max(
        usable,
        key=lambda item: (
            extract_ocr_time([item]) is None,
            any(token in item.upper() for token in ("BANKI", "BARA", "BYPASS", "NH")),
            "-" in item or "_" in item,
            sum(ch.isalpha() for ch in item),
            len(item),
        ),
    )


def finalize_parsed_ocr(parsed: ParsedOcr) -> ParsedOcr:
    """Move a clock token out of location when time/date parse missed it."""
    time = parsed.time or extract_ocr_time([parsed.location or ""])
    location = parsed.location
    if location:
        stripped = parse_location_texts([location])
        location = stripped if _usable_location(stripped) else None
    return ParsedOcr(time=time, date=parsed.date, location=location)


def parse_osd_hits(
    hits: Sequence[tuple[str, float]],
    min_confidence: float,
) -> tuple[ParsedOcr, float | None]:
    """Map EasyOCR-style (text, prob) hits into structured OSD fields."""
    texts, mean = filter_ocr_hits(hits, min_confidence)
    meta = parse_metadata_texts(texts)
    location = _preferred_location(parse_location_texts(texts), meta.location)
    return finalize_parsed_ocr(
        ParsedOcr(
            time=meta.time,
            date=meta.date,
            location=location if _usable_location(location) else None,
        )
    ), mean


def parse_corner_osd_hits(
    metadata_hits: Sequence[tuple[str, float]],
    location_hits: Sequence[tuple[str, float]],
    min_confidence: float,
) -> tuple[ParsedOcr, float | None]:
    """Parse metadata and location ROIs separately, then merge confidence."""
    meta_texts, meta_conf = filter_ocr_hits(metadata_hits, min_confidence)
    loc_texts, loc_conf = filter_ocr_hits(location_hits, min_confidence)
    meta = parse_metadata_texts(meta_texts)
    loc_meta = parse_metadata_texts(loc_texts)
    location = _preferred_location(
        parse_location_texts(loc_texts),
        parse_location_texts(meta_texts),
    )
    confs = [value for value in (meta_conf, loc_conf) if value is not None]
    mean_conf = sum(confs) / len(confs) if confs else None
    return finalize_parsed_ocr(
        ParsedOcr(
            time=meta.time or loc_meta.time,
            date=meta.date or loc_meta.date,
            location=location if _usable_location(location) else None,
        )
    ), mean_conf


def _merge_parsed_ocr(primary: ParsedOcr, fallback: ParsedOcr) -> ParsedOcr:
    """Fill missing OSD fields; keep a plausible date over a 70xx OCR year."""
    date = primary.date
    if not is_plausible_ocr_date(date) and is_plausible_ocr_date(fallback.date):
        date = fallback.date
    elif not date:
        date = fallback.date
    return ParsedOcr(
        time=primary.time or fallback.time,
        date=date,
        location=_preferred_location(primary.location, fallback.location),
    )


def _mean_confidences(*values: float | None) -> float | None:
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _osd_fields_complete(parsed: ParsedOcr) -> bool:
    return bool(parsed.time and parsed.date and _usable_location(parsed.location))


def parse_frame_corner_osd(
    frame: object,
    easyocr_reader: Any,
    min_confidence: float,
) -> tuple[ParsedOcr, float | None]:
    """Fast 2× corners, wide 4× if time/date miss, then S21 band fallback."""
    metadata_hits, location_hits = read_corner_osd_hits(
        frame,
        easyocr_reader,
        rois=DEFAULT_OSD_ROIS,
        scale=OCR_ROI_SCALE,
    )
    parsed, conf = parse_corner_osd_hits(metadata_hits, location_hits, min_confidence)
    parsed = finalize_parsed_ocr(parsed)
    if _osd_fields_complete(parsed):
        return parsed, conf
    if not (parsed.time and parsed.date):
        wide_meta, wide_loc = read_corner_osd_hits(
            frame,
            easyocr_reader,
            rois=WIDE_OSD_ROIS,
            scale=OCR_FALLBACK_SCALE,
        )
        parsed_wide, conf_wide = parse_corner_osd_hits(wide_meta, wide_loc, min_confidence)
        parsed = finalize_parsed_ocr(_merge_parsed_ocr(parsed_wide, parsed))
        conf = _mean_confidences(conf, conf_wide)
        if _osd_fields_complete(parsed):
            return parsed, conf
    if _osd_fields_complete(parsed):
        return parsed, conf
    band_hits = read_band_osd_hits(frame, easyocr_reader, scale=OCR_BAND_SCALE)
    parsed_band, conf_band = parse_osd_hits(band_hits, min_confidence)
    merged = finalize_parsed_ocr(_merge_parsed_ocr(parsed, parsed_band))
    return merged, _mean_confidences(conf, conf_band)


def read_corner_osd_hits_with_fallback(
    frame: object,
    easyocr_reader: Any,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return fast-path hits, then wide and band hits only when fields are missing."""
    metadata_hits, location_hits = read_corner_osd_hits(
        frame,
        easyocr_reader,
        rois=DEFAULT_OSD_ROIS,
        scale=OCR_ROI_SCALE,
    )
    parsed, _conf = parse_corner_osd_hits(metadata_hits, location_hits, 0.0)
    if _osd_fields_complete(parsed):
        return metadata_hits, location_hits
    if not (parsed.time and parsed.date):
        wide_meta, wide_loc = read_corner_osd_hits(
            frame,
            easyocr_reader,
            rois=WIDE_OSD_ROIS,
            scale=OCR_FALLBACK_SCALE,
        )
        parsed_wide, _wide_conf = parse_corner_osd_hits(wide_meta, wide_loc, 0.0)
        parsed = _merge_parsed_ocr(parsed_wide, parsed)
        metadata_hits, location_hits = wide_meta, wide_loc
        if _osd_fields_complete(parsed):
            return metadata_hits, location_hits
    if _osd_fields_complete(parsed):
        return metadata_hits, location_hits
    band_hits = read_band_osd_hits(frame, easyocr_reader, scale=OCR_BAND_SCALE)
    return metadata_hits + band_hits, location_hits + band_hits


def _roi_bounds(width: int, height: int, roi: OsdRoi) -> tuple[int, int, int, int]:
    y1 = max(0, min(height, int(height * roi.y_start)))
    y2 = max(0, min(height, int(height * roi.y_end)))
    x1 = max(0, min(width, int(width * roi.x_start)))
    x2 = max(0, min(width, int(width * roi.x_end)))
    return y1, y2, x1, x2


def prepare_roi_for_ocr(crop_bgr: object, scale: float = OCR_ROI_SCALE) -> object:
    """Upscale a BGR crop and return an RGB array for EasyOCR."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for OSD ROI OCR") from exc
    scaled = crop_bgr
    if scale != 1.0:
        scaled = cv2.resize(
            crop_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    return cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)


def crop_has_mixed_text_polarity(crop_bgr: object) -> bool:
    """True when OSD strokes are both bright-on-dark and dark-on-light."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if gray.size < 80:
        return False
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = float(np.percentile(magnitude, 85))
    edges = magnitude > max(threshold, 12.0)
    if int(np.count_nonzero(edges)) < 20:
        return False
    samples = gray[edges]
    dark_frac = float(np.mean(samples < 80))
    bright_frac = float(np.mean(samples > 175))
    return dark_frac > 0.12 and bright_frac > 0.12


def polarity_invariant_rgb(rgb: object) -> object:
    """Stroke-emphasize mixed white/black glyphs so EasyOCR sees one polarity."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    inverted = cv2.bitwise_not(np.asarray(normalized, dtype="uint8"))
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)


def _collect_readtext_hits(
    easyocr_reader: Any,
    rgb: object,
    read_kwargs: dict[str, Any],
) -> list[tuple[str, float]]:
    results = easyocr_reader.readtext(rgb, **read_kwargs)
    hits: list[tuple[str, float]] = []
    for item in results:
        if len(item) < 2:
            continue
        confidence = float(item[2]) if len(item) >= 3 else 1.0
        hits.append((str(item[1]), confidence))
    return hits


def _read_roi_hits(
    bgr: Any,
    easyocr_reader: Any,
    roi: OsdRoi,
    scale: float,
) -> list[tuple[str, float]]:
    height, width = bgr.shape[:2]
    y1, y2, x1, x2 = _roi_bounds(width, height, roi)
    if y2 <= y1 or x2 <= x1:
        return []
    crop = bgr[y1:y2, x1:x2]
    rgb = prepare_roi_for_ocr(crop, scale)
    read_kwargs: dict[str, Any] = {"paragraph": True}
    if roi.allowlist is not None:
        read_kwargs["allowlist"] = roi.allowlist
    hits = _collect_readtext_hits(easyocr_reader, rgb, read_kwargs)
    mixed = crop_has_mixed_text_polarity(crop)
    if mixed:
        try:
            import cv2

            inverted = cv2.bitwise_not(rgb)
            hits.extend(_collect_readtext_hits(easyocr_reader, inverted, read_kwargs))
        except ImportError:
            pass
        if roi.name == "location":
            extra = polarity_invariant_rgb(rgb)
            hits.extend(_collect_readtext_hits(easyocr_reader, extra, read_kwargs))
    return hits


def read_corner_osd_hits(
    frame: object,
    easyocr_reader: Any,
    *,
    rois: Sequence[OsdRoi] = DEFAULT_OSD_ROIS,
    scale: float = OCR_ROI_SCALE,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """OCR the metadata and location corner ROIs on a BGR frame."""
    bgr = _as_bgr_array(frame)
    metadata_hits: list[tuple[str, float]] = []
    location_hits: list[tuple[str, float]] = []
    for roi in rois:
        hits = _read_roi_hits(bgr, easyocr_reader, roi, scale)
        if roi.name == "metadata":
            metadata_hits.extend(hits)
        elif roi.name == "location":
            location_hits.extend(hits)
        else:
            metadata_hits.extend(hits)
            location_hits.extend(hits)
    return metadata_hits, location_hits


def read_band_osd_hits(
    frame: object,
    easyocr_reader: Any,
    *,
    rois: Sequence[OsdRoi] = BAND_OSD_ROIS,
    scale: float = OCR_BAND_SCALE,
) -> list[tuple[str, float]]:
    """OCR full-width top/bottom bands used when corner ROIs miss fields (S21)."""
    bgr = _as_bgr_array(frame)
    hits: list[tuple[str, float]] = []
    for roi in rois:
        hits.extend(_read_roi_hits(bgr, easyocr_reader, roi, scale))
    return hits


def easyocr_reader(gpu: bool = False) -> CornerOsdReader:
    """Build a corner-ROI EasyOCR reader (Phase 4 production path)."""
    return CornerOsdReader(gpu=gpu)


def optional_easyocr_reader(gpu: bool = False) -> OcrReader | CornerOsdReader:
    """Use EasyOCR when installed; otherwise return no hits so prescan still runs."""
    if find_spec("easyocr") is None:
        return lambda _frame: []
    return easyocr_reader(gpu=gpu)


def is_corner_osd_reader(reader: OcrReader | CornerOsdReader | None) -> TypeGuard[CornerOsdReader]:
    """True when ``reader`` reads metadata/location corner ROIs."""
    return isinstance(reader, CornerOsdReader)
