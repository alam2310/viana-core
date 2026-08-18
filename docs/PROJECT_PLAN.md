# Project Plan — ViAna v2 Platform

**Status tracker:** `docs/PROJECT_STATUS.md` (update when phases complete)  
**Architecture:** `docs/ARCHITECTURE.md`  
**API contracts:** `docs/api_contracts.md` + `packages/contracts/schemas/`

Consolidated v2 plan. Historical research: `legacy/blueprint.md`.

---

## 1. Objective

- **Event-sourced** — `{stem}_events.csv` primary; `{stem}_15min.csv` via separate aggregation.
- **CLI-first** — `python -m viana`; same config as API.
- **Backend-managed jobs** — UI does not send `job_id` / `gpu_device`.
- **Monorepo** — engine, API, UI, shared contracts.

**Scope v0.1:** ViAna Moving Count only.

---

## 2. Locked decisions

| Topic | Decision |
|-------|----------|
| Legacy | `legacy/` discardable; parity: `legacy/inference/inference_engine.py` |
| GPU | Max 2 jobs; one GPU each; both models on assigned GPU |
| Output | `{parent_dir}/{project_id}/{stem}_*` |
| 15min CSV | Clock windows; zero-fill; `aggregate: true` classes only |
| Geometry | Mandatory pixel coords within frame bounds |
| OCR | Full-frame v1; recalibrate + user fallback |
| Resume | Explicit only |
| Prescan | OCR + auto line proposal; user edits on canvas |
| Orchestrator | Subprocess engine invocation |
| Profiles | `{parent_dir}/{project_id}/profiles/` |

---

## 3. Implementation phases

### Phase 0 — Monorepo scaffold ✅ (closed 2026-08-18 — `docs/PHASE_0_SIGNOFF.md`)

### Phase 1 — Contracts & config
- JobConfig validation, classes/defaults loaders, schema sync, tests

### Phase 2 — Engine I/O & CSV
- events.csv, aggregate.py, checkpoint.py

### Phase 3 — CV core ✅ (2026-08-18)
- detect, classify, track, crossing, time_map (port from legacy)

### Phase 4 — Prescan ✅ (2026-08-19)
- OCR (EasyOCR optional), line proposal, profiles, `viana prescan` preview JPEG

### Phase 5 — Process & render ✅ (2026-08-19)
- Main loop, telemetry, FFmpeg overlay, explicit resume (`viana run` / `viana resume`)

### Phase 6 — Orchestrator
- FastAPI routes, worker pool, WebSocket

### Phase 7 — UI foundation ✅ (2026-08-18)
- Next.js scaffold, container manager, API client, mock mode

### Phase 8 — UI workflows ✅ (2026-08-18, mocked)
- Prescan, canvas, queue, paused UX

### Phase 9 — Parity & hardening
- Golden clip; delete `legacy/` after sign-off

---

## 4. Parallel work

| Agent | Start now | Blocked for E2E |
|-------|-----------|-----------------|
| Engine | Phase 1 | — |
| UI | Phase 7 with fixtures | Phase 6 API |
| API | Phase 6 design | Engine `viana run` |

---

## 5. References

- `docs/adr/`, `legacy/PARITY.md`, `docs/ui/`, `packages/contracts/`
