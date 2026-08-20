"""Moving-count process loop: events CSV, checkpoints, telemetry. No 15-min bins."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from uuid import uuid4

from viana.config.classes import ClassTaxonomy, VehicleClass, load_class_taxonomy
from viana.config.defaults import EngineDefaults, load_engine_defaults
from viana.config.job import JobConfig
from viana.domain.boxes import Detection
from viana.io.checkpoint import Checkpoint, load_checkpoint, save_checkpoint, utc_now_iso
from viana.io.csv_schema import RawCrossingEventRow
from viana.io.events import EventsCsvWriter
from viana.io.paths import artifact_paths
from viana.io.run_result import RunResult, RunResultArtifacts, completed_now, save_run_result
from viana.io.telemetry import TelemetryMessage, emit_telemetry_stderr
from viana.stages.crossing import Crossing
from viana.stages.cv_core import FrameCVEngine
from viana.stages.ocr import OcrReader, optional_easyocr_reader, parse_osd_hits
from viana.stages.prescan import VideoMeta
from viana.stages.render import FfmpegRenderer, FrameRenderer, NullRenderer
from viana.stages.time_map import TimeMap, save_time_map, time_map_from_metadata
from viana.stages.track import build_tracker
from viana.stages.ultralytics_detect import UltralyticsDualDetector
from viana.stages.video import VideoFrame, iter_cv2_frames

FrameDetector = Callable[[VideoFrame], tuple[list[Detection], list[Detection]]]
TelemetryEmit = Callable[[TelemetryMessage], None]
FrameFeed = tuple[VideoMeta, Iterable[VideoFrame]]


class CheckpointExistsError(ValueError):
    """Raised when ``viana run`` would silently resume an incomplete checkpoint."""


class MissingCheckpointError(ValueError):
    """Raised when ``viana resume`` has no checkpoint on disk."""


def _class_or_unknown(taxonomy: ClassTaxonomy, class_id: int) -> VehicleClass | None:
    try:
        return taxonomy.by_id(class_id)
    except KeyError:
        return None


def crossing_to_event(
    job: JobConfig,
    taxonomy: ClassTaxonomy,
    video_file: str,
    crossing: Crossing,
    time_map: TimeMap,
) -> RawCrossingEventRow:
    """Map a unique crossing to an events-CSV row (schema columns only)."""
    vehicle = _class_or_unknown(taxonomy, crossing.class_id)
    raw = _class_or_unknown(taxonomy, crossing.raw_class_id)
    wall, source, ocr_conf = time_map.resolve(crossing.video_pts_ms)
    date = job.metadata.user_start_date
    location = job.metadata.location
    if time_map.anchors:
        date = time_map.anchors[-1].date or date
        location = time_map.anchors[-1].location or location
    return RawCrossingEventRow(
        event_id=uuid4(),
        job_id=job.job_id,
        video_file=video_file,
        track_id=crossing.track_id,
        frame_index=crossing.frame_index,
        video_pts_ms=crossing.video_pts_ms,
        class_name=vehicle.name if vehicle else f"class_{crossing.class_id}",
        direction=crossing.direction,
        confidence=crossing.confidence,
        wall_time=wall,
        wall_time_source=source,
        ocr_confidence=ocr_conf,
        date=date,
        location=location,
        class_id=crossing.class_id,
        raw_class_id=crossing.raw_class_id,
        raw_class_name=raw.name if raw else None,
        category=vehicle.category if vehicle else None,
        class_type=vehicle.class_type if vehicle else None,
        sub_class=vehicle.sub_class if vehicle else None,
        norm_area=crossing.norm_area,
        anchor_x=crossing.anchor_x,
        anchor_y=crossing.anchor_y,
    )


def _wipe_run_artifacts(paths: dict[str, Path]) -> None:
    for key in ("events", "checkpoint", "processed_video", "time_map", "run_result", "manifest"):
        target = paths[key]
        if target.is_file():
            target.unlink()


def _default_detector(job: JobConfig, defaults: EngineDefaults) -> FrameDetector:
    model = UltralyticsDualDetector(
        defaults.models.vehicle,
        defaults.models.pedestrian,
        device=job.gpu_device,
        detection=defaults.detection,
    )
    return model.detect


def _default_frames(source: Path, start_index: int) -> FrameFeed:
    return iter_cv2_frames(source, start_index=start_index)


def _open_renderer(
    job: JobConfig,
    paths: dict[str, Path],
    meta: VideoMeta,
    *,
    resume: bool,
    start_index: int,
    class_names: dict[int, str] | None = None,
) -> FrameRenderer:
    if not job.task_parameters.render_video or resume or start_index > 0:
        return NullRenderer()
    renderer = FfmpegRenderer(paths["processed_video"], meta.width, meta.height, meta.fps)
    renderer.set_lines(job.task_parameters.horizon_line, job.task_parameters.counting_line)
    renderer.set_class_names(class_names or {})
    return renderer


def _save_progress_checkpoint(
    paths: dict[str, Path],
    job: JobConfig,
    video_stem: str,
    *,
    current_frame: int,
    total_frames: int,
    counted: set[int],
    events_rows: int,
) -> None:
    save_checkpoint(
        paths["checkpoint"],
        Checkpoint(
            job_id=job.job_id,
            project_id=job.project_id,
            source_video_path=job.source_video_path,
            video_stem=video_stem,
            current_frame=current_frame,
            total_frames=max(total_frames, 1),
            saved_at=utc_now_iso(),
            counted_track_ids=sorted(counted),
            events_rows_written=events_rows,
        ),
    )


def run_moving_count(
    job: JobConfig,
    *,
    resume: bool,
    frames: FrameFeed | None = None,
    detector: FrameDetector | None = None,
    renderer: FrameRenderer | None = None,
    emit: TelemetryEmit | None = None,
    ocr_reader: OcrReader | None = None,
    taxonomy: ClassTaxonomy | None = None,
    defaults: EngineDefaults | None = None,
) -> RunResult:
    """Process a video into events CSV + checkpoint/time map. No 15-min aggregation."""
    emit = emit or emit_telemetry_stderr
    defaults = (defaults or load_engine_defaults()).apply_task_overrides(job.task_parameters)
    taxonomy = taxonomy or load_class_taxonomy()
    video_stem = job.source_video_path.stem
    paths = artifact_paths(job.output_dir, video_stem)
    if frames is None and not job.source_video_path.is_file():
        raise FileNotFoundError(f"Video not found: {job.source_video_path}")
    job.output_dir.mkdir(parents=True, exist_ok=True)

    if job.start_fresh:
        _wipe_run_artifacts(paths)

    checkpoint: Checkpoint | None = None
    start_index = 0
    ckpt_path = paths["checkpoint"]
    if ckpt_path.is_file():
        checkpoint = load_checkpoint(ckpt_path)
        if checkpoint.is_complete() and not job.start_fresh:
            if resume:
                artifacts = RunResultArtifacts(
                    events=str(paths["events"]) if paths["events"].is_file() else None,
                    time_map=str(paths["time_map"]) if paths["time_map"].is_file() else None,
                    processed_video=(
                        str(paths["processed_video"])
                        if paths["processed_video"].is_file()
                        else None
                    ),
                )
                result = completed_now(
                    job.job_id, job.source_video_path, video_stem, artifacts, status="COMPLETED"
                )
                save_run_result(paths["run_result"], result)
                return result
            raise CheckpointExistsError(
                "checkpoint already complete; set start_fresh=true to re-run"
            )
        if not resume and not job.start_fresh:
            raise CheckpointExistsError("checkpoint exists; use `viana resume` or set start_fresh")
        if resume:
            start_index = checkpoint.current_frame

    if resume and checkpoint is None:
        raise MissingCheckpointError(f"Checkpoint not found: {ckpt_path}")

    supplied_frames = frames is not None
    try:
        meta, frame_iter = (
            frames if frames is not None else _default_frames(job.source_video_path, start_index)
        )
    except FileNotFoundError:
        raise
    job.validate_geometry(meta.width, meta.height)
    total_frames = max(meta.frame_count, start_index + 1)
    detect = detector or _default_detector(job, defaults)
    ocr = ocr_reader if ocr_reader is not None else optional_easyocr_reader()
    writer_renderer: FrameRenderer = (
        renderer
        if renderer is not None
        else _open_renderer(
            job,
            paths,
            meta,
            resume=resume,
            start_index=start_index,
            class_names=taxonomy.id_to_name(),
        )
    )

    engine = FrameCVEngine(
        horizon=job.task_parameters.horizon_line,
        counting_line=job.task_parameters.counting_line,
        frame_height=meta.height,
        detection=defaults.detection,
        classification=defaults.classification,
        tracker=build_tracker(frame_rate=meta.fps),
    )
    if checkpoint is not None and resume:
        engine.crossings.counted_track_ids = set(checkpoint.counted_track_ids)

    time_map = time_map_from_metadata(job.job_id, video_stem, job.metadata)
    events_rows = checkpoint.events_rows_written if checkpoint is not None and resume else 0
    append_events = resume and paths["events"].is_file()
    last_ocr_pts = 0.0
    t0 = time.perf_counter()
    processed = start_index
    last_index = start_index - 1

    emit(
        TelemetryMessage(
            job_id=job.job_id,
            status="PROCESSING",
            telemetry_type="LOG",
            data={"message": "process_start", "resume": resume, "start_index": start_index},
        )
    )

    try:
        with EventsCsvWriter(paths["events"], append=append_events) as csv_writer:
            iterator: Iterator[VideoFrame] = iter(frame_iter)
            for frame in iterator:
                if frame.index < start_index:
                    continue
                last_index = frame.index
                if frame.index == 0 or (
                    resume and time_map.anchors == [] and frame.index == start_index
                ):
                    if frame.image is not None:
                        parsed, conf = parse_osd_hits(ocr(frame.image), defaults.ocr.min_confidence)
                        time_map = time_map_from_metadata(
                            job.job_id,
                            video_stem,
                            job.metadata,
                            ocr=parsed,
                            video_pts_ms=frame.pts_ms,
                            ocr_confidence=conf,
                        )
                    last_ocr_pts = frame.pts_ms
                elif (
                    frame.image is not None
                    and (frame.pts_ms - last_ocr_pts)
                    >= defaults.ocr.recalibration_interval_sec * 1000.0
                ):
                    parsed, conf = parse_osd_hits(ocr(frame.image), defaults.ocr.min_confidence)
                    extra = time_map_from_metadata(
                        job.job_id,
                        video_stem,
                        job.metadata,
                        ocr=parsed,
                        video_pts_ms=frame.pts_ms,
                        ocr_confidence=conf,
                    )
                    time_map.anchors.extend(extra.anchors)
                    last_ocr_pts = frame.pts_ms

                vehicles, pedestrians = detect(frame)
                cv_result = engine.process_models(
                    vehicles,
                    pedestrians,
                    frame_index=frame.index,
                    video_pts_ms=frame.pts_ms,
                )
                for crossing in cv_result.crossings:
                    row = crossing_to_event(
                        job, taxonomy, job.source_video_path.name, crossing, time_map
                    )
                    csv_writer.write_row(row)
                    events_rows += 1
                    emit(
                        TelemetryMessage(
                            job_id=job.job_id,
                            status="PROCESSING",
                            telemetry_type="MOVING_EVENT",
                            data={
                                "track_id": crossing.track_id,
                                "class_name": row.class_name,
                                "direction": crossing.direction,
                                "frame_index": crossing.frame_index,
                                "fps": meta.fps,
                                "video_pts_ms": crossing.video_pts_ms,
                                "event_timestamp": row.wall_time,
                                "event_timestamp_source": row.wall_time_source,
                                "event_timestamp_confidence": row.ocr_confidence,
                            },
                        )
                    )
                writer_renderer.write(frame, cv_result)
                processed = frame.index + 1
                progress_every = (
                    defaults.pipeline.telemetry_detail_progress_frames
                    if job.task_parameters.telemetry_detail
                    else defaults.pipeline.telemetry_progress_frames
                )
                if processed % progress_every == 0 or processed == total_frames:
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    fps_val = round(processed / elapsed, 2)
                    remaining = max(0, total_frames - processed)
                    # Wall-clock ETA: remaining *frames* / processing fps (not video fps).
                    eta_sec = round(remaining / fps_val, 1) if fps_val > 0 else None
                    emit(
                        TelemetryMessage(
                            job_id=job.job_id,
                            status="PROCESSING",
                            telemetry_type="PROGRESS",
                            data={
                                "current_frame": processed,
                                "total_frames": total_frames,
                                "processing_fps": fps_val,
                                "crossing_count": events_rows,
                                **({"eta_sec": eta_sec} if eta_sec is not None else {}),
                            },
                        )
                    )
                if processed % defaults.pipeline.checkpoint_interval_frames == 0:
                    _save_progress_checkpoint(
                        paths,
                        job,
                        video_stem,
                        current_frame=processed,
                        total_frames=total_frames,
                        counted=engine.crossings.counted_track_ids,
                        events_rows=events_rows,
                    )
            if last_index >= 0:
                observed = last_index + 1
                if supplied_frames:
                    total_frames = max(total_frames, observed)
                else:
                    # Decoder EOF is the true length when MPEG-PS headers inflate nb_frames.
                    total_frames = observed
            processed = max(processed, last_index + 1)
            _save_progress_checkpoint(
                paths,
                job,
                video_stem,
                current_frame=processed,
                total_frames=max(total_frames, processed, 1),
                counted=engine.crossings.counted_track_ids,
                events_rows=events_rows,
            )
        save_time_map(paths["time_map"], time_map)
        writer_renderer.close()
        artifacts = RunResultArtifacts(
            events=str(paths["events"]),
            time_map=str(paths["time_map"]),
            processed_video=(
                str(paths["processed_video"])
                if job.task_parameters.render_video and paths["processed_video"].is_file()
                else None
            ),
        )
        result = completed_now(job.job_id, job.source_video_path, video_stem, artifacts)
        save_run_result(paths["run_result"], result)
        emit(
            TelemetryMessage(
                job_id=job.job_id,
                status="COMPLETED",
                telemetry_type="LOG",
                data={"message": "process_complete", "events_rows": events_rows},
            )
        )
        return result
    except KeyboardInterrupt:
        writer_renderer.close()
        _save_progress_checkpoint(
            paths,
            job,
            video_stem,
            current_frame=processed,
            total_frames=max(total_frames, 1),
            counted=engine.crossings.counted_track_ids,
            events_rows=events_rows,
        )
        artifacts = RunResultArtifacts(
            events=str(paths["events"]) if paths["events"].is_file() else None
        )
        result = completed_now(
            job.job_id,
            job.source_video_path,
            video_stem,
            artifacts,
            status="CANCELLED",
            error_message="interrupted",
        )
        save_run_result(paths["run_result"], result)
        emit(
            TelemetryMessage(
                job_id=job.job_id,
                status="CANCELLED",
                telemetry_type="LOG",
                data={"message": "interrupted"},
            )
        )
        return result
    except Exception as exc:
        writer_renderer.close()
        _save_progress_checkpoint(
            paths,
            job,
            video_stem,
            current_frame=processed,
            total_frames=max(total_frames, 1),
            counted=engine.crossings.counted_track_ids,
            events_rows=events_rows,
        )
        artifacts = RunResultArtifacts(
            events=str(paths["events"]) if paths["events"].is_file() else None
        )
        result = completed_now(
            job.job_id,
            job.source_video_path,
            video_stem,
            artifacts,
            status="FAILED",
            error_message=str(exc),
        )
        save_run_result(paths["run_result"], result)
        emit(
            TelemetryMessage(
                job_id=job.job_id,
                status="FAILED",
                telemetry_type="LOG",
                data={"message": str(exc)},
            )
        )
        return result
