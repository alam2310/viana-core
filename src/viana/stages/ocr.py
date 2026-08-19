"""OSD OCR for prescan. EasyOCR is optional; tests inject text hits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, TypeGuard

from viana.stages.time_map import ParsedOcr, parse_location_texts, parse_metadata_texts

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


# Top band: date/time/day. Bottom-left: camera location label.
# Fast path uses a tighter metadata crop at 2× so CRAFT runs on fewer pixels
# (S08). Wide 4× ROIs remain the accuracy fallback when time/date are missing.
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

OCR_ROI_SCALE = 2.0
OCR_FALLBACK_SCALE = 4.0


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


def parse_osd_hits(
    hits: Sequence[tuple[str, float]],
    min_confidence: float,
) -> tuple[ParsedOcr, float | None]:
    """Map EasyOCR-style (text, prob) hits into structured OSD fields."""
    texts, mean = filter_ocr_hits(hits, min_confidence)
    meta = parse_metadata_texts(texts)
    location = parse_location_texts(texts)
    return ParsedOcr(
        time=meta.time,
        date=meta.date,
        location=location or meta.location,
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
    location = parse_location_texts(loc_texts)
    confs = [value for value in (meta_conf, loc_conf) if value is not None]
    mean_conf = sum(confs) / len(confs) if confs else None
    return ParsedOcr(
        time=meta.time,
        date=meta.date,
        location=location,
    ), mean_conf


def _merge_parsed_ocr(primary: ParsedOcr, fallback: ParsedOcr) -> ParsedOcr:
    """Fill missing OSD fields from a slower/wider second pass."""
    return ParsedOcr(
        time=primary.time or fallback.time,
        date=primary.date or fallback.date,
        location=primary.location or fallback.location,
    )


def parse_frame_corner_osd(
    frame: object,
    easyocr_reader: Any,
    min_confidence: float,
) -> tuple[ParsedOcr, float | None]:
    """Fast 2× tight ROI, then wide 4× fallback if time or date is missing."""
    metadata_hits, location_hits = read_corner_osd_hits(
        frame,
        easyocr_reader,
        rois=DEFAULT_OSD_ROIS,
        scale=OCR_ROI_SCALE,
    )
    parsed, conf = parse_corner_osd_hits(metadata_hits, location_hits, min_confidence)
    if parsed.time and parsed.date:
        return parsed, conf
    wide_meta, wide_loc = read_corner_osd_hits(
        frame,
        easyocr_reader,
        rois=WIDE_OSD_ROIS,
        scale=OCR_FALLBACK_SCALE,
    )
    parsed_wide, conf_wide = parse_corner_osd_hits(wide_meta, wide_loc, min_confidence)
    merged = _merge_parsed_ocr(parsed_wide, parsed)
    confs = [value for value in (conf, conf_wide) if value is not None]
    mean_conf = sum(confs) / len(confs) if confs else None
    return merged, mean_conf


def read_corner_osd_hits_with_fallback(
    frame: object,
    easyocr_reader: Any,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return fast-path hits, or wide-ROI hits when time/date do not parse."""
    metadata_hits, location_hits = read_corner_osd_hits(
        frame,
        easyocr_reader,
        rois=DEFAULT_OSD_ROIS,
        scale=OCR_ROI_SCALE,
    )
    parsed, _conf = parse_corner_osd_hits(metadata_hits, location_hits, 0.0)
    if parsed.time and parsed.date:
        return metadata_hits, location_hits
    return read_corner_osd_hits(
        frame,
        easyocr_reader,
        rois=WIDE_OSD_ROIS,
        scale=OCR_FALLBACK_SCALE,
    )


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


def read_corner_osd_hits(
    frame: object,
    easyocr_reader: Any,
    *,
    rois: Sequence[OsdRoi] = DEFAULT_OSD_ROIS,
    scale: float = OCR_ROI_SCALE,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """OCR the metadata and location corner ROIs on a BGR frame."""
    bgr = _as_bgr_array(frame)
    height, width = bgr.shape[:2]
    metadata_hits: list[tuple[str, float]] = []
    location_hits: list[tuple[str, float]] = []
    for roi in rois:
        y1, y2, x1, x2 = _roi_bounds(width, height, roi)
        if y2 <= y1 or x2 <= x1:
            continue
        crop = bgr[y1:y2, x1:x2]
        rgb = prepare_roi_for_ocr(crop, scale)
        read_kwargs: dict[str, Any] = {"paragraph": True}
        if roi.allowlist is not None:
            read_kwargs["allowlist"] = roi.allowlist
        results = easyocr_reader.readtext(rgb, **read_kwargs)
        hits: list[tuple[str, float]] = []
        for item in results:
            if len(item) < 2:
                continue
            confidence = float(item[2]) if len(item) >= 3 else 1.0
            hits.append((str(item[1]), confidence))
        if roi.name == "metadata":
            metadata_hits.extend(hits)
        elif roi.name == "location":
            location_hits.extend(hits)
    return metadata_hits, location_hits


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
