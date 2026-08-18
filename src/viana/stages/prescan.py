"""Sample a video, OCR OSD, propose lines, and write a preview JPEG."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from viana.config.defaults import load_engine_defaults
from viana.config.job import PROJECT_ID_PATTERN
from viana.io.paths import prescan_dir
from viana.io.profiles import CalibrationProfile, list_profiles
from viana.stages.lines import ProposedLines, propose_lines
from viana.stages.ocr import OcrReader, parse_osd_hits
from viana.stages.time_map import ParsedOcr

# 1×1 JPEG so CI can write a preview without OpenCV/Pillow.
_PLACEHOLDER_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb0043000101010101010101010101"
    "010101010101010101010101010101010101010101010101010101010101010101010101"
    "01010101010101ffc0000b080001000101011100ffc40014100100000000000000000000"
    "00000000000000ffda00080001000100003f00fbffd9"
)


class VideoMeta(BaseModel):
    """Frame size and timing from the source video."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: float = Field(gt=0)
    duration_sec: float = Field(ge=0)
    frame_count: int = Field(ge=0)


class PrescanOcr(BaseModel):
    """OSD fields returned to the UI (nulls when OCR misses)."""

    model_config = ConfigDict(extra="forbid")

    time: str | None = None
    date: str | None = None
    location: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class PrescanResponse(BaseModel):
    """CLI/API payload — ``prescan_response.schema.json``."""

    model_config = ConfigDict(extra="forbid")

    prescan_id: str
    video_meta: VideoMeta
    ocr: PrescanOcr
    proposed_lines: ProposedLines | None = None
    preview_url: str | None = None
    profiles: list[CalibrationProfile] = Field(default_factory=list)


@dataclass(frozen=True)
class SampledVideo:
    """One preview frame plus container metadata (frame may be None in tests)."""

    meta: VideoMeta
    frame_offset_sec: float
    frame: object | None = None
    preview_jpeg: bytes | None = None


VideoSampler = Callable[[Path, float], SampledVideo]


def new_prescan_id() -> str:
    """Allocate a disk-safe prescan identifier."""
    return f"prescan_{uuid4().hex[:12]}"


def preview_jpeg_path(output_dir: Path, prescan_id: str) -> Path:
    """Return ``{output_dir}/prescan/{prescan_id}_preview.jpg``."""
    return prescan_dir(output_dir) / f"{prescan_id}_preview.jpg"


def write_preview_jpeg(path: Path, jpeg_bytes: bytes | None) -> None:
    """Write a JPEG preview (placeholder when encoder output is missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg_bytes if jpeg_bytes else _PLACEHOLDER_JPEG)


def _encode_bgr_jpeg(frame: object) -> bytes | None:
    try:
        import cv2
    except ImportError:
        return None
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return bytes(buffer)


def sample_video_cv2(source: Path, frame_offset_sec: float) -> SampledVideo:
    """Open ``source`` with OpenCV and grab a frame at ``frame_offset_sec``."""
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to sample video") from exc
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        if width < 1 or height < 1:
            raise ValueError(f"Invalid frame size {width}x{height} in {source}")
        offset = max(0.0, frame_offset_sec)
        if duration_sec > 0:
            offset = min(offset, max(0.0, duration_sec - (1.0 / fps)))
        capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read a frame from {source}")
        meta = VideoMeta(
            width=width,
            height=height,
            fps=fps,
            duration_sec=duration_sec,
            frame_count=max(frame_count, 0),
        )
        return SampledVideo(
            meta=meta,
            frame_offset_sec=offset,
            frame=frame,
            preview_jpeg=_encode_bgr_jpeg(frame),
        )
    finally:
        capture.release()


def _ocr_from_sample(
    sampled: SampledVideo,
    min_confidence: float,
    ocr_reader: OcrReader | None,
) -> tuple[ParsedOcr, float | None]:
    if ocr_reader is None:
        return ParsedOcr(), None
    if sampled.frame is None:
        hits: Sequence[tuple[str, float]] = ocr_reader(sampled)
    else:
        hits = ocr_reader(sampled.frame)
    return parse_osd_hits(hits, min_confidence)


def run_prescan(
    source: Path,
    project_id: str,
    *,
    frame_offset_sec: float = 0.0,
    output_dir: Path,
    sampler: VideoSampler | None = None,
    ocr_reader: OcrReader | None = None,
    prescan_id: str | None = None,
) -> PrescanResponse:
    """Produce a PrescanResponse and write ``{prescan_id}_preview.jpg``."""
    if not PROJECT_ID_PATTERN.match(project_id):
        raise ValueError("project_id must match [a-z0-9][a-z0-9_-]*")
    if frame_offset_sec < 0:
        raise ValueError("frame_offset must be >= 0")
    probe = sampler or sample_video_cv2
    sampled = probe(source, frame_offset_sec)
    defaults = load_engine_defaults()
    parsed, ocr_conf = _ocr_from_sample(sampled, defaults.ocr.min_confidence, ocr_reader)
    profiles = list_profiles(output_dir)
    lines = propose_lines(sampled.meta.width, sampled.meta.height, profiles)
    ident = prescan_id or new_prescan_id()
    preview = preview_jpeg_path(output_dir, ident)
    write_preview_jpeg(preview, sampled.preview_jpeg)
    return PrescanResponse(
        prescan_id=ident,
        video_meta=sampled.meta,
        ocr=PrescanOcr(
            time=parsed.time,
            date=parsed.date,
            location=parsed.location,
            confidence=ocr_conf,
        ),
        proposed_lines=lines,
        preview_url=str(preview),
        profiles=profiles,
    )
