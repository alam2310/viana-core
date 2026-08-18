# Orchestrator design intent (`src/orchestrator/`)

## Purpose

FastAPI job queue and HTTP/WebSocket façade over the `viana` CLI. Owns job lifecycle and GPU assignment.

## Invariants

- **No CV in routes** — workers spawn `python -m viana` via subprocess.
- **No client `job_id` / `gpu_device`** on `POST /jobs`.
- **No silent resume** — if a checkpoint exists, plain submit returns **409** unless explicit resume/start-fresh.
- WebSocket telemetry payloads match `packages/contracts/schemas/telemetry.schema.json`.
- Max **two** concurrent GPU workers.

## Preconditions

- Engine package installed (`pip install -e .`).
- `VIANA_OUTPUT_PARENT` configured for artifact root.
- NVIDIA runtime available when scheduling GPU jobs.

## Rationale

- **Thin API layer** — keeps failure domains separate; engine crashes do not require API redeploy.
- **Structured logs** — `structlog` JSON in production for job_id-scoped log correlation.

## Pattern reference

| Task | Example |
|------|---------|
| New route module | `src/orchestrator/routes/health.py` |
| App wiring | `src/orchestrator/app.py` |
| Logging setup | `src/orchestrator/logging_config.py` |
