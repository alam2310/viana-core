# Engine design intent (`src/viana/`)

## Purpose

CLI-first computer vision pipeline for ViAna Moving Count: detect, classify, track, count line crossings, emit CSV artifacts.

## Invariants

- The frame loop **must not** compute 15-minute aggregates inline.
- Each crossing is appended to `{stem}_events.csv` exactly once per track crossing.
- `counted_track_ids` (or equivalent) prevents duplicate crossing events.
- Checkpoint files are written only for resumable states; resume is **explicit** (`viana resume` or API resume).
- Class names and aggregation flags come from `configs/classes.yaml` only.

## Preconditions

- `horizon_line` and `counting_line` are present and within `video_meta` dimensions.
- Model weights exist at paths in `configs/engine_defaults.yaml`.
- `source_video_path` is readable inside the container.

## Rationale

- **Subprocess-friendly CLI** — orchestrator spawns `python -m viana run` without embedding CV in HTTP handlers.
- **Separate aggregation** — allows OCR/time-map fixes and re-binning without re-inference (ADR 001).

## Pattern reference

| Task | Example |
|------|---------|
| New CLI stage | `src/viana/cli.py` command stub |
| Job config model | `src/viana/config/job.py` |
| Artifact paths | `src/viana/io/paths.py` |
