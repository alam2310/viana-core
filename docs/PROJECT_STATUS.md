# Project Status (Living Document)

**Last updated:** 2026-08-19  
**Current focus:** Prescan UI redesign → wall-clock → `{stem}_15min.csv`  
**API blocker:** none. Live `:8000/health` is Phase 6.  
**Phase 0 closed:** 2026-08-18 — see `docs/PHASE_0_SIGNOFF.md`  
**Phase 9 parity:** signed off 2026-08-19 — `legacy/` removed  
**Canonical plan:** `docs/PROJECT_PLAN.md`

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

## Next (active)

1. **Prescan UI redesign** — OCR / proposed time / user fallback visible and editable.
2. **Wall-clock on submit** — persist metadata so `viana aggregate` writes real `{stem}_15min.csv`.
3. Re-run aggregate on a clip with user start time to verify 15-min grid.

## Parked (revisit later)

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
| 2026-08-19 | Phase 9 signed off; `legacy/` removed; `training/uvh/` + `docs/ops/ENVIRONMENT_SETUP.md` + `docs/archive/ITVA_RESEARCH_LOG.md`; compose `build: .` |
| 2026-08-19 | Overlay human go (geom D / b5); HEVC cq 42; rolling majority off-line |
| 2026-08-19 | E2E mocks off; Phase 6 workers; parity recorded on `hiv000001` |
| 2026-08-19 | Phase 5–6: process loop, orchestrator, telemetry |
| 2026-08-18 | Phase 0–4, 7–8 foundation |

(Full history: git log.)
