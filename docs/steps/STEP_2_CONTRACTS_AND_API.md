# Step 2 — Contracts & API foundation

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | Contract + API |
| **Blocks** | Step 3 |
| **Blocked by** | Step 1 ✅ |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 2.

---

## Objective

Schema-first contract changes and **HTTP API surface** for the backend-owned prescan lifecycle. No heavy engine/prescan worker logic here — that is **Step 3**.

---

## Work items

| ID | Item | Owner | Status |
|----|------|-------|--------|
| P1 | `JobStatus` enum — add `PRESCAN_*`, `AWAITING_REVIEW`, `READY`; remove `PENDING` | Contract | ✅ |
| P2 | `JobStatusResponse` — `proposed_*` + confirmed OCR/lines | Contract | ✅ |
| P5 | `output_dir` on intake / job config | Contract | ✅ |
| P6 | `telemetry.schema.json` status enum sync | Contract | ✅ |
| G1 | Separate proposed vs confirmed fields on job record | Contract + storage | ✅ |
| G4 | Server validation: mandatory metadata + `HH:MM:SS` + `DD-MM-YYYY` | API | ✅ |
| G14 | Orchestrator state machine for new statuses (stub transitions OK) | API | ✅ |
| G15 | Persist prescan proposal + confirmed calibration on job | API + storage | ✅ |
| G16 | `POST /jobs/intake` — create job(s) from path(s) | API | ✅ |
| G17 | `PATCH /jobs/{id}/prescan` — confirm review → `READY` | API | ✅ |
| G20 | `output_dir` override per project | Contract + API | ✅ |
| G2 | `task_type` on prescan request (schema only) | Contract | ✅ |

**Step 3 (not here):** G7, G8, G9, G12, G13, G19 — engine + workers + media serving.

**Step 4 (UI):** G18 filesystem browser, G22 telemetry formatters.

**Step 6 (G21 closed in 6.7):** extra bind-mounts via compose + `VIANA_INTAKE_ROOTS` / `VIANA_PATH_MAPS`. G5, G10 remain later if still needed.

---

## Contract workflow

`schemas/` → `typescript/` → `fixtures/` → `job.py` → `api_contracts.md` → `openapi.yaml`

### Proposals log

| ID | Field / endpoint | Schema file | Status |
|----|------------------|-------------|--------|
| P1 | `JobStatus` extension | `job_status.schema.json` | ✅ |
| P2 | Prescan fields on response | `job_status.schema.json` | ✅ |
| P3 | `POST /jobs/intake` | `job_intake.schema.json` + `api_contracts.md` | ✅ |
| P4 | `PATCH /jobs/{id}/prescan` | `job_prescan_confirm.schema.json` + `api_contracts.md` | ✅ |
| P5 | `output_dir` | `job_submit.schema.json` / `job_config` | ✅ |
| P6 | Telemetry status enum | `telemetry.schema.json` | ✅ |

---

## Surfaces

| Surface | Path |
|---------|------|
| Contracts | `packages/contracts/` |
| Pydantic | `src/viana/config/job.py` |
| API routes | `src/orchestrator/routes/jobs.py` |
| Job storage / models | `src/orchestrator/models.py` |
| Tests | `tests/orchestrator/test_job_routes.py` |

**Do not edit:** `apps/web/` (Step 4), deep prescan engine (Step 3).

---

## Exit criteria

- [x] P1–P6 schemas + TS + fixtures merged
- [x] Intake + prescan confirm routes return contract-shaped responses
- [x] Job record persists proposed + confirmed fields
- [x] Metadata validation enforced on confirm
- [x] Tests pass for new routes
- [x] `AGENT_PROGRESS.md` Step 2 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Step 2 complete: schemas, intake/confirm routes, GPU gate on READY, 19 orchestrator tests |
| 2026-08-19 | Split from monolithic Step 2; contracts + API only |
| 2026-08-19 | Work items from Step 1 `DISCOVERY.md` §5 |
