# Component Map (Step 1 redesign → Step 3 implementation)

**Spec:** [`REDESIGN.md`](REDESIGN.md)

## Feature modules

| Module | Path | Responsibility |
|--------|------|----------------|
| Container | `features/container/` | Docker health, start |
| Project | `features/project/` | `project_id`, browsable `output_dir`, task type picker |
| Intake | `features/intake/` | Host path browser, file/folder/multi-select |
| Queue | `features/queue/` | Job table, FIFO, row actions, status labels |
| Prescan | `features/prescan/` | Review modal: canvas, OCR, scrubber, summary step |
| Calibration | `features/calibration/` | HTML5 canvas, line drag, profiles |
| Telemetry | `features/telemetry/` | WS hook, progress strip, crossing feed, activity log, raw JSON toggle |
| Monitor | `features/monitor/` | Live sidebar: partial MP4 + telemetry stack |
| Dashboard | `app/page.tsx` | Layout: project bar + intake + queue + monitor drawer |

## New / changed vs Phase 8

| Component | Change |
|-----------|--------|
| `queue-panel.tsx` | → **JobQueueTable** with full status lifecycle |
| `prescan-modal.tsx` | Side-by-side layout + review summary step |
| `telemetry-panel.tsx` | Replace raw JSON with structured views |
| `dashboard.tsx` | Remove localStorage drafts; backend job sync only |
| `path-browser.tsx` | **New** — host API filesystem picker |
| `monitor-sidebar.tsx` | **New** — video + telemetry |
| `project-bar.tsx` | **New** — project_id + output_dir + task type |

## Shadcn / UI primitives

| Primitive | Use |
|-----------|-----|
| Table | Job queue, crossing feed (virtualized wrapper) |
| Sheet / Dialog | Prescan review (wide), monitor sidebar |
| Progress | Row progress + ETA strip |
| Toast | Errors, container path warning |
| Button | Review, Monitor, Retry, Cancel, Artifacts |
| Dropdown | Task type |
| Badge | Status operator labels |

## API dependencies (Step 2)

| Endpoint | Purpose |
|----------|---------|
| `POST /jobs/intake` | Create job(s) from path(s) |
| `PATCH /jobs/{id}/prescan` | Confirm reviewed calibration → `READY` |
| `POST /jobs/{id}/prescan/retry` | `PRESCAN_FAILED` → retry |
| `GET /jobs` | Queue table sync |
| `GET /api/fs/browse` | Host path browser (Step 3 host route) |
| `GET /artifacts/.../partial.mp4` | Live monitor (range requests) |
| `WS /ws/jobs` | Progress, crossings, logs |
