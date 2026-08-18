# Project Status (Living Document)

**Last updated:** 2026-08-19  
**Current focus:** E2E (mocks off) complete; Phase 9 parity recorded — **do not delete `legacy/`**  
**API blocker:** none. Live `:8000/health` is Phase 6.  
**Phase 0 closed:** 2026-08-18 — see `docs/PHASE_0_SIGNOFF.md`  
**Canonical plan:** `docs/PROJECT_PLAN.md`

> AI agents: update this file when you complete a phase, endpoint, or milestone. Do not rely on chat memory.

---

## Overall progress (v2 platform)

| Track | Phase | Status | Owner surface |
|-------|-------|--------|---------------|
| Platform | 0 — Monorepo scaffold | ✅ **Closed** | repo root |
| Engine | 1 — Contracts & config | ✅ **Complete** | `src/viana/` |
| Engine | 2 — I/O & CSV | ✅ **Complete** | `src/viana/` |
| Engine | 3 — CV core | ✅ **Complete** (modules; GPU loop is Phase 5) | `src/viana/` |
| Engine | 4 — Prescan & lines | ✅ **Complete** | `src/viana/` |
| Engine | 5 — Process & render | ✅ **Complete** (YOLO/FFmpeg optional at runtime) | `src/viana/` |
| API | 6 — Orchestrator | ✅ Workers spawn `python -m viana` | `src/orchestrator/` |
| UI | 7 — Foundation | ✅ Scaffold complete | `apps/web/` |
| UI | 8 — Workflows | ✅ Live client (`NEXT_PUBLIC_USE_MOCKS=false`) | `apps/web/` |
| QA | 9 — Parity & hardening | 🟡 Recorded, **not signed off** | `tests/viana/fixtures/PARITY_NOTES.md` |

---

## Phase 0 completion checklist ✅

- [x] `src/viana/` package + CLI stubs
- [x] `src/orchestrator/` FastAPI stub (`GET /health`)
- [x] `packages/contracts/` schemas + TS types + fixtures
- [x] `configs/classes.yaml`, `configs/engine_defaults.yaml`
- [x] `legacy/` consolidated (discardable)
- [x] `docs/ui/*` skeleton guides
- [x] `docs/adr/` 001, 002
- [x] Governance docs (`AGENTS.md`, this file, `docs/governance/*`)
- [x] Models: `models/v1/`, `models/pretrained/`, `models/README.md`
- [x] Engine artifact schemas: `checkpoint`, `job_status`, `run_result`
- [x] Dockerfile installs `pip install -e ".[dev]"`
- [x] UI: Next.js 15 scaffold in `apps/web/` (Phase 7)
- [x] Formal sign-off: `docs/PHASE_0_SIGNOFF.md`

---

## Phase 1 — first tasks

1. [x] Full `JobConfig` Pydantic validation ↔ JSON schema sync
2. [x] `classes.yaml` / `engine_defaults.yaml` loaders + tests
3. [x] Wire CSV column validation to `events_*.schema.json`
4. [x] Optional: `.github/workflows/ci.yml` already runs `pytest tests/viana/`

**Engine Phase 1 (2026-08-18):** `JobConfig` + `load_job_config`; `viana run`/`resume` validate JSON then exit 2; CSV column helpers in `viana.io.csv_schema`.

**Engine Phase 2 (2026-08-18):** `{stem}_events.csv` writer (`viana.io.events`), `viana aggregate` clock 15-min grid (`viana.stages.aggregate`), checkpoint read/write (`viana.io.checkpoint`).

**Engine Phase 3 (2026-08-18):** detect merge/NMS, IoU tracker, heuristic classify, once-per-track crossing, time map (`viana.stages.cv_core`). `viana run` still exit-2 until Phase 5 opens video + weights.

**Engine Phase 5 (2026-08-19):** `viana run` / `viana resume` open the video, run `FrameCVEngine`, append `{stem}_events.csv`, write checkpoints/time_map/run_result, emit telemetry JSON on stderr. No 15-min bins in the frame loop. Live YOLO + FFmpeg need container extras; tests inject frames/detectors.

---

## API implementation matrix (for UI parallel work)

UI **must** use `packages/contracts/fixtures/` until an endpoint is marked ✅.

| Endpoint | Implemented | Schema | Fixture |
|----------|-------------|--------|---------|
| `GET /health` | ✅ | — | — |
| `POST /utils/prescan` | ✅ shells `viana prescan`; rewrites `preview_url` | `prescan_response.schema.json` | `prescan_response.json` |
| `POST /jobs` | ✅ 409 on incomplete checkpoint; 422 if client sends `job_id`/`gpu_device` | `job_submit.schema.json`, `job_submit_response.schema.json` | `job_submit_response.json` |
| `GET /jobs` | ✅ | `job_status.schema.json` (array) | — |
| `GET /jobs/{id}` | ✅ | `job_status.schema.json` | `job_status_paused.json` |
| `POST /jobs/{id}/resume` | ✅ shells `viana resume` | — | — |
| `POST /jobs/{id}/start-fresh` | ✅ shells `viana run` with `start_fresh` | — | — |
| `DELETE /jobs/{id}` | ✅ | — | — |
| `POST /jobs/{id}/aggregate` | ✅ shells `viana aggregate` | — | — |
| `WS /ws/jobs` | ✅ stderr NDJSON → telemetry.schema.json | `telemetry.schema.json` | `telemetry_progress.json` |
| `GET/POST /projects/{id}/profiles` | ✅ disk under `{output_dir}/profiles/` | `calibration_profile.schema.json` | `calibration_profile.json` |

**Engine disk artifacts** (not HTTP): `checkpoint.schema.json`, `run_result.schema.json`, `time_map.schema.json`, `calibration_profile.schema.json` — fixtures `checkpoint_resume.json`, `time_map.json`, `calibration_profile.json`, `run_result.json`.

---

## CLI implementation matrix

| Command | Implemented | Notes |
|---------|-------------|-------|
| `viana prescan` | ✅ | Preview JPEG + PrescanResponse JSON on stdout |
| `viana run` | ✅ | Events CSV + checkpoint + run_result; YOLO/OpenCV at runtime |
| `viana resume` | ✅ | Explicit checkpoint continue; no silent resume |
| `viana aggregate` | ✅ | Events CSV → `{stem}_15min.csv`; `--partial` for incomplete runs |

---

## Locked decisions

See `docs/PROJECT_PLAN.md` and `docs/adr/`.

| Topic | Decision |
|-------|----------|
| Job ownership | Backend assigns `job_id`, `gpu_device` |
| GPU | 1 GPU / job; 2 concurrent videos |
| Output path | `{output.parent_dir}/{project_id}/` |
| Artifacts | `{stem}_events`, `{stem}_15min`, `{stem}_processed` |
| Resume | Explicit only |
| Detection conf default | 0.75 |
| Scope | ViAna Moving Count only (v0.1) |

---

## E2E (UI mocks off) — 2026-08-19

- `apps/web/.env.local`: `NEXT_PUBLIC_USE_MOCKS=false`, `NEXT_PUBLIC_API_URL=http://localhost:8000`
- Container `viana_core` publishes **8000**; compose command runs uvicorn (installs editable package if missing, then pins `numpy<2`).
- `GET http://localhost:8000/health` → `{"status":"ok","phase":6}`. CORS allows the Next.js origin.
- Scripted live flow: health → prescan (`/data/raw/test_video.mp4`) → submit (no `job_id`/`gpu_device`) → WS/GET progress → COMPLETED → `POST …/aggregate` → 409 on silent re-submit → `start_fresh` → DELETE cancel → CANCELLED.
- Pause/resume: short clips finish before `checkpoint_interval_frames=500`, so **PAUSED** is rare; cancel is eventually `CANCELLED` (poll; not always instant).
- 15-min CSV needs wall-clock metadata (OCR or `user_start_time`/`user_start_date`). `test_video` OCR was empty → header-only `_15min.csv`. `parity_golden` with user metadata → 28 aggregate rows (`--partial` because 300/331 frames).
- **YOLO + FFmpeg:** confirmed in container after NumPy pin. Processed MP4 written. Do not treat a NumPy 2 + OpenCV 4.10 combo as success.

## Parity gate (before deleting `legacy/`)

- [x] Golden clip — `tests/viana/fixtures/PARITY_NOTES.md` (`hiv000001` 180s in-frame match + `parity_golden.mp4`)
- [x] Legacy vs v2 counts — `legacy/PARITY.md` (matched lines + conf 0.25: 125 vs 160; vehicles 115 vs 113; **not** ±2% overall)
- [x] Real project extract via UI HTTP client (not CLI-only); browser click-through not automated
- [ ] Human sign-off on deltas — **required before deleting `legacy/`**

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-19 | Overlay class: rolling majority off-line (no distant lock); HEVC NVENC cq 42 |
| 2026-08-19 | Overlay: class name on boxes; geometric lines pin x=0 / x=width-1. Parity v2@0.25: 160 vs legacy 125 (vehicles 113 vs 115; Pedestrian 47 vs 10). Still not ±2%; do not delete `legacy/` |
| 2026-08-19 | Parity re-run: `hiv000001` 180s window with **matching in-frame lines**; legacy 125 vs v2@0.75 72 (Van/MiniBus equal). Still not ±2%; do not delete `legacy/` |
| 2026-08-19 | E2E: UI mocks off; CORS; preview canvas; 409/aggregate/cancel; live Phase 6 health. Phase 9: golden clip + legacy vs v2 counts recorded; **do not delete legacy/** |
| 2026-08-19 | Phase 6: orchestrator workers spawn `python -m viana`; job queue (2 GPUs); 409 silent-resume; WS telemetry |
| 2026-08-19 | Phase 5: viana run/resume process loop, telemetry, FFmpeg render, explicit resume |
| 2026-08-19 | Phase 4: viana prescan OCR + line proposal + profiles + preview JPEG |
| 2026-08-18 | Phase 8 UI workflows (mocked): prescan modal, calibration canvas, queue, paused resume/fresh |
| 2026-08-18 | Phase 3: detect/classify/track/crossing/time_map (CPU-testable; no GPU loop) |
| 2026-08-18 | Phase 7 UI scaffold: Next.js 15, fixture api-client, host `/api/container/*`, dashboard |
| 2026-08-18 | Phase 2: events CSV writer, viana aggregate (clock 15-min, zero-fill), checkpoint I/O |
| 2026-08-18 | Phase 1 complete: JobConfig schema sync, CSV column contracts, CLI config validation |
| 2026-08-18 | Orchestrator job/prescan/profile/WS routes scaffolded as 501 stubs; GPU workers blocked on Phase 5 CLI |
| 2026-08-18 | Phase 1: classes.yaml / engine_defaults.yaml Pydantic loaders + tests |
| 2026-08-18 | AgentReady round 2: requirements.txt, pattern refs, legacy docstrings |
| 2026-08-18 | AgentReady remediation: CI, lint, OpenAPI, threat model, design docs |
| 2026-08-18 | Phase 0 formally closed; hygiene pass (schemas, fixtures, Dockerfile, sign-off doc) |
| 2026-08-18 | Repo cleanup: historical docs → `legacy/`; active specs → `docs/` |
| 2026-08-18 | Phase 0 complete; governance docs; UI agent context in `apps/web/AGENTS.md` |
| 2026-08-18 | v2 plan approved (event-sourced, backend jobs, monorepo) |
