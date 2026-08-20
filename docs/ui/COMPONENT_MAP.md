# Component Map (Step 1 redesign → Step 4 implementation)

**Spec:** [`REDESIGN.md`](REDESIGN.md)  
**Status:** ✅ Implemented Step 4 (2026-08-19)

## Feature modules

| Module | Path | Responsibility |
|--------|------|----------------|
| Container | `features/container/container-panel.tsx` | Docker health, start (compact in project bar) |
| Project | `features/project/project-bar.tsx` | `project_id`, browsable `output_dir`, task type picker |
| Intake | `features/intake/intake-panel.tsx`, `path-browser.tsx` | Host path browser, file/folder/multi-select |
| Queue | `features/queue/job-queue-table.tsx`, `job-status.ts` | Job table, FIFO, row actions, status labels |
| Prescan | `features/prescan/prescan-review-modal.tsx` | Review modal: side-by-side canvas, OCR, scrubber, summary step |
| Calibration | `features/calibration/calibration-canvas.tsx` | HTML5 canvas, line drag |
| Telemetry | `features/telemetry/telemetry-panel.tsx`, `telemetry-formatters.ts` | Structured progress, crossing feed, activity log |
| Monitor | `features/monitor/monitor-sidebar.tsx` | Live sidebar: progress + Live Crossings (WS). **Partial MP4 preview PARKED** — see `live-processed-video.tsx` + STABILIZATION_BACKLOG S20/S24 |
| Dashboard | `features/dashboard/dashboard.tsx` | Layout: project bar + intake + queue + monitor drawer |

## Removed (Phase 8)

| Component | Replaced by |
|-----------|-------------|
| `queue-panel.tsx` | `job-queue-table.tsx` |
| `prescan-modal.tsx` | `prescan-review-modal.tsx` |
| localStorage `pendingPaths` / `drafts` | Backend `GET /jobs` + prescan lifecycle |

## Shadcn / UI primitives

| Primitive | Use |
|-----------|-----|
| Table | Job queue, crossing feed |
| Dialog (fixed overlay) | Prescan review (wide), path browser |
| Progress | Row progress + ETA strip |
| Button | Review, Monitor, Retry, Cancel |
| Select | Task type |
| Badge-like spans | Status operator labels |

## API dependencies

| Endpoint | Step | Consumer |
|----------|------|----------|
| `POST /jobs/intake` | 2 | `intake-panel.tsx` |
| `PATCH /jobs/{id}/prescan` | 2 | `prescan-review-modal.tsx` |
| `POST /jobs/{id}/prescan/retry` | 3 | `job-queue-table.tsx` |
| `GET /jobs` | 2 | `dashboard.tsx` |
| `GET /jobs/{id}/prescan/preview` | 3 | `prescan-review-modal.tsx` — **Re-scan OCR** only (stabilization S05) |
| `GET /artifacts/{id}/source.mp4` | stab S02 | `calibration-canvas.tsx` — frame scrub via video seek |
| `/api/proxy/source` | stab S03 | Same-origin proxy for source MP4 (`api-client.sourceVideoUrl`) |
| `/api/proxy/preview` | 4 | `calibration-canvas.tsx` — prescan JPEG |
| `GET /api/fs/browse` | 4 | `path-browser.tsx` (Next.js route) |
| `GET /artifacts/.../partial.mp4` | 3 | **PARKED** — code in `live-processed-video.tsx` (not mounted). Live Monitor shows crossings only (S24) |
| `WS /ws/jobs` | 3 | `dashboard.tsx` → telemetry formatters |

## localStorage (UI prefs only)

| Key | Purpose |
|-----|---------|
| `viana.project_id` | Active project slug |
| `viana.output_dir` | Output directory override |
| `viana.task_type` | Task picker (Moving only enabled v0.1) |
| `viana.telemetry_detail` | Default for next prescan confirm |

Queue **Submitted**, **Video length**, and **Run time** use API `created_at`, `video_duration_sec`, and `processing_duration_sec` (not localStorage).

**Queue timing units (S19 / F016):**

| Column | Source | Unit |
|--------|--------|------|
| Video length | `JobStatus.video_duration_sec` | seconds → `HH:MM:SS` |
| Run time | `JobStatus.processing_duration_sec` | seconds → `HH:MM:SS` |
| Time remaining | `progress.eta_sec` | seconds; formula `(total_frames − current_frame) / processing_fps` |

Do not derive video length from `total_frames / processing_fps` (that is GPU ETA, not clip length).
