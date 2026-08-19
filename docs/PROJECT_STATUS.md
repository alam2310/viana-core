# Project Status (Living Document)

**Last updated:** 2026-08-19  
**Current focus:** **Step 5** — E2E verification (see `docs/steps/TRACKER.md`)  
**Post-v0.1 plan:** `docs/steps/PLAN.md` · **Agent checklist:** `docs/steps/AGENT_PROGRESS.md`  
**API blocker:** none (S07 corner ROI OCR fixed 2026-08-19 — Step 5 unblocked).  
**Phase 0 closed:** 2026-08-18 — see `docs/PHASE_0_SIGNOFF.md`  
**Phase 9 parity:** signed off 2026-08-19 — `legacy/` removed  
**Canonical plan:** `docs/PROJECT_PLAN.md` (Phases 0–9); **Steps 1–6** for remaining work

> AI agents: update this file when you complete a phase, endpoint, or milestone. Do not rely on chat memory.

---

## Overall progress (v2 platform)

| Track | Phase | Status | Owner surface |
|-------|-------|--------|---------------|
| Platform | 0 — Monorepo scaffold | ✅ **Closed** | repo root |
| Engine | 1–5 | ✅ **Complete** | `src/viana/` |
| API | 6 — Orchestrator | ✅ Workers spawn `python -m viana` | `src/orchestrator/` |
| UI | 7 — Foundation | ✅ Scaffold complete | `apps/web/` |
| UI | 8 — Workflows | ✅ Live client (`NEXT_PUBLIC_USE_MOCKS=false`) | `apps/web/` |
| QA | 9 — Parity | ✅ **Signed off** (geometry D / b5) | `tests/viana/fixtures/PARITY_NOTES.md` |

---

## Post-v0.1 Steps (active)

**Tracker:** [`docs/steps/TRACKER.md`](steps/TRACKER.md) · **Plan:** [`docs/steps/PLAN.md`](steps/PLAN.md)

| Step | Name | Status |
|------|------|--------|
| 1 | UX discovery & design | ✅ Complete |
| 2 | Contracts & API foundation | ✅ Complete |
| 3 | Engine prescan & orchestrator | ✅ Complete |
| 4 | UI implementation | ✅ Complete |
| 5 | E2E verification (`_15min.csv`) | ⬜ Not started |
| 6 | Hardening backlog | ⬜ Not started |

**Goals (Steps 1–5):** Backend prescan lifecycle → UI redesign → verify `{stem}_15min.csv`.

**Parked items** in [`docs/steps/STEP_6_HARDENING.md`](steps/STEP_6_HARDENING.md).

---

## Next (legacy list — see Steps above)

1. **Step 5** — Verify 15-min grid on test clip (`verification/5_15min_results.md`).

## Parked (revisit later)

See **Step 6** in [`docs/steps/STEP_6_HARDENING.md`](steps/STEP_6_HARDENING.md):

| Item | Notes |
|------|--------|
| Pause / resume / PAUSED UX | Needs job &gt; 500-frame checkpoint |
| Browser click-through | Manual/Playwright UI pass; HTTP E2E done |
| Extra camera clip beyond `hiv000001` | Overlay go on D sufficient for v0.1 |
| GPU tests in CI | No GPU in GitHub Actions |
| Bake `trackers` + `numpy<2` into image | Compose pip-installs on start |
| Faster DELETE → CANCELLED | Cancel is eventual via poll |

---

## E2E (UI mocks off) — 2026-08-19

- `apps/web/.env.local`: `NEXT_PUBLIC_USE_MOCKS=false`, `NEXT_PUBLIC_API_URL=http://localhost:8000`
- Container `viana_core`: `docker compose build && up`; publishes **8000**
- Scripted live flow: health → prescan → submit → progress → COMPLETED → aggregate → 409 → start-fresh → cancel
- **15-min CSV:** engine supports `viana aggregate`; needs wall-clock from prescan UI (not a legacy parity gap)
- **YOLO + FFmpeg:** HEVC NVENC cq 42 processed MP4; NumPy &lt; 2 + trackers `--no-deps`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-19 | Stabilization S01–S02: preview registry disk fallback verified; `GET /artifacts/{id}/source.mp4` with HTTP Range for prescan-phase jobs |
| 2026-08-19 | Step 4 complete: API-driven queue UI — intake browser, prescan review, live monitor, structured telemetry; localStorage drafts removed |
| 2026-08-19 | Step 3 complete: prescan worker queue, dark-frame skip, `GET /artifacts/{id}/partial.mp4`, auto-aggregate, ETA/crossings on progress |
| 2026-08-19 | Step 2 complete: `JobStatus` prescan lifecycle, `POST /jobs/intake`, `PATCH /jobs/{id}/prescan`, metadata validation |
| 2026-08-19 | Six-step plan: Step 2 contracts/API, Step 3 engine/workers, UI → Step 4 |
| 2026-08-19 | Step 1 complete: `DISCOVERY.md`, `REDESIGN.md`, flows |
| 2026-08-19 | Post-v0.1 Steps plan + tracker under `docs/steps/` |
| 2026-08-19 | Phase 9 signed off; `legacy/` removed; `training/uvh/` + `docs/ops/ENVIRONMENT_SETUP.md` + `docs/archive/ITVA_RESEARCH_LOG.md`; compose `build: .` |
| 2026-08-19 | Overlay human go (geom D / b5); HEVC cq 42; rolling majority off-line |
| 2026-08-19 | E2E mocks off; Phase 6 workers; parity recorded on `hiv000001` |
| 2026-08-19 | Phase 5–6: process loop, orchestrator, telemetry |
| 2026-08-18 | Phase 0–4, 7–8 foundation |

(Full history: git log.)
