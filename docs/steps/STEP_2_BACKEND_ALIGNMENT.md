# Step 2 — Backend alignment (conditional)

| Field | Value |
|-------|-------|
| **Status** | ⬜ Not started — **required** (Step 1 filed work items) |
| **Chat** | **New** — Backend (Contract and/or Engine/API) |
| **Blocks** | Step 3 |
| **Blocked by** | — (Step 1 ✅) |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 2.

---

## Objective

Close gaps between Step 1 UX and current prescan/backend **before** UI implementation.

---

## Work items (from Step 1)

| ID | Item | Owner | Status |
|----|------|-------|--------|
| G4 | Server validation: mandatory metadata + `HH:MM:SS` + `DD-MM-YYYY` | API | ⬜ |
| G7 | Auto-skip dark/blocked opening frames in prescan sampler | Engine | ⬜ |
| G8 | Live frame preview on scrub (prescan re-run or frame endpoint) | API + Engine | ⬜ |
| G9 | ETA + crossing count in job status / telemetry | API + contract | ⬜ |
| G12 | Auto-aggregate on job `COMPLETED` | Orchestrator | ⬜ |
| G13 | Orchestrator prescan worker queue for bulk intake | API + pool | ⬜ |
| G14 | `JobStatus`: add `PRESCAN_PENDING`, `PRESCAN_RUNNING`, `PRESCAN_FAILED`, `AWAITING_REVIEW`, `READY`; remove `PENDING` | Contract + orchestrator | ⬜ |
| G15 | Persist proposed + confirmed prescan/calibration on job record | Contract + storage | ⬜ |
| G16 | Intake API — `POST /jobs/intake` from path(s) | API | ⬜ |
| G17 | Confirm API — `PATCH /jobs/{id}/prescan` → `READY` | API | ⬜ |
| G19 | Partial `_processed.mp4` HTTP serving (range) | API | ⬜ |
| G20 | `output_dir` override per project on job config | Contract + API | ⬜ |
| G1 | Separate `proposed_*` vs confirmed fields on job | Contract | ⬜ |
| G2 | `task_type` on prescan request (extensibility) | Contract + engine | ⬜ (low priority v0.1) |

**Deferred:** G5 NP/Junction engine, G10 background-match profiles, G21 container paths → **Step 5.7**.

**UI-only (Step 3):** G18 host filesystem browser, G22 telemetry formatters.

---

## Contract changes

Follow `docs/governance/CONTRACT_SYNC.md`:

`schemas/` → `typescript/` → `fixtures/` → `job.py` → `api_contracts.md` → `openapi.yaml`

### Proposals log

| ID | Field / endpoint | Schema file | Rationale | Status |
|----|------------------|-------------|-----------|--------|
| P1 | `JobStatus` enum extension | `job_status.schema.json` | Prescan lifecycle | ⬜ |
| P2 | `JobStatusResponse` prescan fields | `job_status.schema.json` | Proposed + confirmed OCR/lines | ⬜ |
| P3 | `POST /jobs/intake` | `api_contracts.md` | Batch path intake | ⬜ |
| P4 | `PATCH /jobs/{id}/prescan` | `api_contracts.md` | Review confirm | ⬜ |
| P5 | `output_dir` on submit/intake | `job_submit.schema.json` | Project output override | ⬜ |
| P6 | `telemetry.schema.json` status enum sync | `telemetry.schema.json` | Match new JobStatus | ⬜ |

---

## Prescan implementation changes

| Surface | Path |
|---------|------|
| Engine prescan | `src/viana/stages/prescan.py`, `src/viana/cli.py` |
| API routes | `src/orchestrator/routes/jobs.py`, prescan utils |
| Worker pool | `src/orchestrator/workers/pool.py` — prescan queue + GPU gate on `READY` |
| Tests | `tests/viana/test_prescan.py`, `tests/orchestrator/` |
| Fixtures | `packages/contracts/fixtures/` |

---

## Exit criteria

- [ ] All work items ✅ above (except deferred)
- [ ] Tests pass for touched paths
- [ ] `AGENT_PROGRESS.md` Step 2 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Work items copied from Step 1 `DISCOVERY.md` §5 |
| 2026-08-19 | Step 1 complete — Step 2 unblocked |
