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
| `job_submit_response.schema.json` | POST /jobs response |
| `prescan_response.schema.json` | POST /utils/prescan response |
| `events_raw.schema.json` | `{stem}_events.csv` columns |
| `events_15min.schema.json` | `{stem}_15min.csv` columns |
| `calibration_profile.schema.json` | Project profiles |
| `telemetry.schema.json` | WebSocket messages |

## Fixtures (UI mocks)

| File | Use |
|------|-----|
| `prescan_response.json` | Prescan modal + canvas |
| `job_status_paused.json` | Resume / start fresh UI |
| `telemetry_progress.json` | Dashboard progress |
| `telemetry_sample.json` | General WS shape reference |

## UI import (after Phase 7 scaffold)

```json
// apps/web/tsconfig.json
"paths": {
  "@viana/contracts": ["../../packages/contracts/typescript"]
}
```
