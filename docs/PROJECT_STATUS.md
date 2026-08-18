# Project Status (Living Document)

**Last updated:** 2026-08-18  
**Current focus:** Phase 3 — CV core (engine); Phase 7 UI scaffold complete  
**API blocker:** `viana run` pipeline not implemented (JobConfig validation only). Orchestrator HTTP routes are **501 stubs only** — no GPU workers until Phase 5.  
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
| Engine | 3 — CV core | ⬜ Not started | `src/viana/` |
| Engine | 4 — Prescan & lines | ⬜ Not started | `src/viana/` |
| Engine | 5 — Process & render | ⬜ Not started | `src/viana/` |
| API | 6 — Orchestrator | ⬜ Scaffold only (501 stubs) | `src/orchestrator/` |
| UI | 7 — Foundation | ✅ Scaffold complete | `apps/web/` |
| UI | 8 — Workflows | ⬜ Not started | `apps/web/` |
| QA | 9 — Parity & hardening | ⬜ Not started | `tests/`, `legacy/PARITY.md` |

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

**Engine Phase 2 (2026-08-18):** `{stem}_events.csv` writer (`viana.io.events`), `viana aggregate` clock 15-min grid (`viana.stages.aggregate`), checkpoint read/write (`viana.io.checkpoint`). Next: Phase 3 detect/classify/track/crossing.

---

## API implementation matrix (for UI parallel work)

UI **must** use `packages/contracts/fixtures/` until an endpoint is marked ✅.

Routes exist under `src/orchestrator/routes/` but return **501** (except `GET /health`). Do not flip ✅ until workers spawn `python -m viana`.

| Endpoint | Implemented | Schema | Fixture |
|----------|-------------|--------|---------|
| `GET /health` | ✅ stub | — | — |
| `POST /utils/prescan` | ❌ 501 | `prescan_response.schema.json` | `prescan_response.json` |
| `POST /jobs` | ❌ 501 (422 if client sends `job_id`/`gpu_device`) | `job_submit.schema.json`, `job_submit_response.schema.json` | `job_submit_response.json` |
| `GET /jobs` | ❌ 501 | `job_status.schema.json` (array) | — |
| `GET /jobs/{id}` | ❌ 501 | `job_status.schema.json` | `job_status_paused.json` |
| `POST /jobs/{id}/resume` | ❌ 501 | — | — |
| `POST /jobs/{id}/start-fresh` | ❌ 501 | — | — |
| `DELETE /jobs/{id}` | ❌ 501 | — | — |
| `POST /jobs/{id}/aggregate` | ❌ 501 | — | — |
| `WS /ws/jobs` | ❌ stub LOG then close | `telemetry.schema.json` | `telemetry_progress.json` |
| `GET/POST /projects/{id}/profiles` | ❌ 501 | `calibration_profile.schema.json` | — |

**Engine disk artifacts** (not HTTP): `checkpoint.schema.json`, `run_result.schema.json` — fixture `checkpoint_resume.json`.

---

## CLI implementation matrix

| Command | Implemented | Notes |
|---------|-------------|-------|
| `viana prescan` | ❌ stub | Phase 4; exit 2 |
| `viana run` | 🔄 JobConfig validation | Valid JSON required; pipeline Phase 3–5 |
| `viana resume` | 🔄 JobConfig validation | Requires `resume=true`; pipeline Phase 5 |
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

## Parity gate (before deleting `legacy/`)

- [ ] Golden clip — `tests/viana/fixtures/PARITY_NOTES.md`
- [ ] Legacy vs v2 counts — `legacy/PARITY.md`
- [ ] Real project videos via UI

---

## Changelog

| Date | Change |
|------|--------|
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
