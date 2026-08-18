# Orchestrator Agent — src/orchestrator

**Read first:** `/AGENTS.md` → `docs/api_contracts.md` → `docs/PROJECT_STATUS.md`

## Owned paths

- `src/orchestrator/` — FastAPI app, routes, workers, WebSocket

## Responsibilities

- Job queue (max 2 concurrent)
- Assign `job_id`, `gpu_device`, `output_dir`
- Spawn `python -m viana` via **subprocess**
- Bridge telemetry stdout → WebSocket
- Job state machine (see `docs/ui/STATE_MACHINE.md`)

## Do not

- Run YOLO inference in route handlers
- Accept `job_id` / `gpu_device` from client on submit
- Auto-resume when checkpoint exists (return 409 unless explicit resume/fresh)

## Contracts

Implement exactly what is in `packages/contracts/schemas/`. Update `docs/PROJECT_STATUS.md` API matrix when each route works.

## Dev server

```bash
make api-dev   # uvicorn on :8000
```
