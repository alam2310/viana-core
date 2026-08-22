# Shared Contracts (`@viana/contracts`)

**Single source of truth** for API and CSV shapes between UI, orchestrator, and engine.

## Layout

```
schemas/           JSON Schema (authoritative)
typescript/        Hand-maintained TS types (must match schemas)
fixtures/          Mock JSON for UI dev when API not ready
```

## Change workflow

1. Edit `schemas/*.json`
2. Update `typescript/index.ts`
3. Update `src/viana/config/job.py` (Pydantic)
4. Update `docs/api_contracts.md`
5. Update `docs/PROJECT_STATUS.md` API matrix when endpoint is implemented

See `docs/governance/AI_SDLC.md`.

## Schemas

| File | Purpose |
|------|---------|
| `job_submit.schema.json` | UI → POST /jobs |
| `job_intake.schema.json` | POST /jobs/intake |
| `job_intake_response.schema.json` | POST /jobs/intake response |
| `job_prescan_confirm.schema.json` | PATCH /jobs/{id}/prescan |
| `prescan_request.schema.json` | POST /utils/prescan request |
| `job_config.schema.json` | Engine CLI JSON (`viana run` / `viana resume`) |
| `job_submit_response.schema.json` | POST /jobs response |
| `job_status.schema.json` | GET /jobs/{id} response |
| `prescan_response.schema.json` | POST /utils/prescan response |
| `events_raw.schema.json` | `{stem}_events.csv` columns (debug) |
| `events_report.schema.json` | `{stem}_events_report.csv` columns |
| `events_15min.schema.json` | `{stem}_15min.csv` columns |
| `calibration_profile.schema.json` | Project profiles |
| `telemetry.schema.json` | WebSocket messages |
| `checkpoint.schema.json` | `_meta/{stem}/checkpoint.json` (legacy flat `{stem}.checkpoint.json` still read) |
| `run_result.schema.json` | `_meta/{stem}/run_result.json` |
| `time_map.schema.json` | `_meta/{stem}/time_map.json` (OCR / user wall-clock anchors) |
| `classes.schema.json` | `configs/classes.yaml` (YOLO id → reporting hierarchy) |
| `engine_defaults.schema.json` | `configs/engine_defaults.yaml` (thresholds, model paths) |

## Fixtures (UI mocks)

| File | Use |
|------|-----|
| `prescan_response.json` | Prescan modal + canvas |
| `calibration_profile.json` | Project calibration profile |
| `job_submit_response.json` | POST /jobs success mock |
| `job_intake_response.json` | POST /jobs/intake mock |
| `job_status_paused.json` | Resume / start fresh UI |
| `job_status_awaiting_review.json` | Prescan review queue row |
| `checkpoint_resume.json` | Engine resume state reference |
| `job_config.json` | Engine CLI JobConfig (not an HTTP mock) |
| `time_map.json` | Engine time-map anchors |
| `run_result.json` | Engine `{stem}.run_result.json` |
| `telemetry_progress.json` | Dashboard progress |
| `telemetry_sample.json` | General WS shape reference |

## UI import (after Phase 7 scaffold)

```json
// apps/web/tsconfig.json
"paths": {
  "@viana/contracts": ["../../packages/contracts/typescript"]
}
```
