# Project Status (Living Document)

**Last updated:** 2026-08-22
**Current focus:** **Step 6** — hardening + remaining stabilization polish (**S32**; see `docs/steps/TRACKER.md`)
**Post-v0.1 plan:** `docs/steps/PLAN.md` · **Agent checklist:** `docs/steps/AGENT_PROGRESS.md`  
**API blocker:** none (S07 corner ROI OCR fixed 2026-08-19 — Step 5 complete).
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
| 5 | E2E verification (`_15min.csv`) | ✅ Complete |
| 6 | Hardening backlog | 🔄 In progress |

**Goals (Steps 1–5):** Backend prescan lifecycle → UI redesign → verify `{stem}_15min.csv`.

**Parked / remaining Step 6** in [`docs/steps/STEP_6_HARDENING.md`](steps/STEP_6_HARDENING.md). Open Seq: [`STABILIZATION_BACKLOG.md`](steps/STABILIZATION_BACKLOG.md) (**S32**).

**Idea dump (manual review only):** [`docs/steps/IDEA_DUMP.md`](steps/IDEA_DUMP.md) — not a work queue; agents must not self-assign. Promoted & done: I001→**6.8**, I003→**6.9**, I002→**6.10**, I006→**6.11**. Dump only: I004 (demoted), I005 (soft subs).

---

## Next (Step 6 + open Seq)

1. **S32** — CSV schema trim (`_events` / `_15min`).
2. **6.2** pause/resume UX (checkpoint path; S30 502 triage done; S29 keep layout preserves incomplete checkpoints).
3. **6.5** extra clip → **6.4** Playwright → **6.6** GPU CI.

## Still open on Step 6

| Item | Notes |
|------|--------|
| 6.2 Pause / resume / PAUSED UX | After S30; needs checkpoint path |
| 6.4 Browser / Playwright | After S31/6.11 (both done) |
| 6.5 Extra camera clip | Beyond `hiv000001` |
| 6.6 GPU tests in CI | No GPU in GitHub Actions |

**Parked (not Step 6 until re-promoted):** S20/S24 live-edge player; idea dump I004 / I005.

---

## E2E (UI mocks off) — 2026-08-19

- `apps/web/.env.local`: `NEXT_PUBLIC_USE_MOCKS=false`, `NEXT_PUBLIC_API_URL=http://localhost:8000`
- Container `viana_core`: `docker compose build && up`; publishes **8000**
- Scripted live flow: health → prescan → submit → progress → COMPLETED → aggregate → 409 → start-fresh → cancel
- **15-min CSV:** engine supports `viana aggregate`; needs wall-clock from prescan UI (not a legacy parity gap)
- **YOLO + FFmpeg:** H.264 NVENC (cq 32, preset p4) processed MP4 for browser live monitor; async annotate/write thread; HEVC fallback only if H.264 encoders missing; NumPy &lt; 2 + `trackers==2.6.0 --no-deps` baked into the image (rebuild required)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-22 | **S29 (F024):** ADR 003 keep layout — flat deliverables; `_meta/{stem}/` sidecars; legacy checkpoint resolve for 6.2; COMPLETED deletes prescan JPEG only |
| 2026-08-22 | **S30 (F025):** start-fresh/resume mutate + `GET /jobs` healthy; 502 = proxy while engine down + unhandled `refreshJobs`; UI banner + GET retry + compose `nofile` |
| 2026-08-22 | **S31 + 6.11 complete** — prescan Confirm/Confirming…; `render_video` toggle (default true); `test_video` false skips `_processed.mp4` |
| 2026-08-22 | **S33 (F028) fixed:** Pedestrian included in `_15min.csv` (`aggregate: true`); schema/docs say vehicles + pedestrians |
| 2026-08-22 | Doc sync: Next/focus → Step 6 + open Seq S29–S33; removed stale “Next: Step 5” |
| 2026-08-21 | Idea dump: **I006 → Step 6.11** — prescan-review `render_video` toggle (existing API/engine field) |
| 2026-08-21 | **Render perf/size:** async FFmpeg writer (copy-on-enqueue + drain on close); H.264 NVENC cq 32 / p4; libx264 CRF 34 / veryfast (replaces cq 28 / p7) |
| 2026-08-21 | **6.7 complete (S09 / F006)** — API normalizes host intake paths onto container mounts or 400s unreadable paths; compose passes `VIANA_HOST_*` maps; extra volume via `VIANA_EXTRA_INTAKE_ROOT` + `VIANA_PATH_MAPS` |
| 2026-08-21 | **6.3 complete** — DELETE marks `CANCELLED` immediately and frees the GPU so the next READY job can drain (S27 fail path unchanged) |
| 2026-08-21 | Stabilization **S27:** FAILED GPU jobs release the slot and auto-start the next FIFO READY job (`pool.py` drain) |
| 2026-08-21 | Stabilization **S25/S26:** Job Queue labels `Queued (PS)` vs `Queued (GPU)`; actions Review → Restart (Overwrite) → Stop; Open output on completed only |
| 2026-08-21 | Stabilization S28: missed crossings when Car↔Jeep flicker drops the box for 1–2 frames across the counting line; retain previous anchors up to 15 frames (`crossing.py`) |
| 2026-08-21 | **S23 / 6.9 (I003):** process loop no longer runs EasyOCR; wall-clock interpolates confirmed prescan/user metadata; `hiv000001_inframe` 203.2s @ 13.45 fps → 179.3s @ 15.26 fps |
| 2026-08-21 | **S10:** no-profile line proposal uses road-band slope clustering + parallel counting offset; `hiv000001_inframe` near geometry C/D; profile override unchanged |
| 2026-08-21 | **6.8 / 6.10:** Live Monitor removed; Live Crossings in job details while processing; count from `progress.crossing_count` |
| 2026-08-21 | Idea dump review: **I001 → Step 6.8**, **I003 → Step 6.9**; I002 stays dump (P3, check `crossing_count`); I004 demoted |
| 2026-08-21 | **S22:** close worker/engine FDs (pipes, VideoCapture, ffmpeg process groups) so multi-file intake does not hit `[Errno 24]` / API 502 |
| 2026-08-21 | Added `docs/steps/IDEA_DUMP.md` — parked ideas for later human review; not a work queue |
| 2026-08-20 | **S21:** adaptive OSD OCR (bands, clock salvage, mixed-polarity location, `7074` year repair) — UI retest OK; `hiv000001_inframe` S07 fields unchanged |
| 2026-08-20 | **S24:** Live Monitor parks in-progress `_processed.mp4` preview (code retained, not mounted); Live Crossings show WS events immediately |
| 2026-08-20 | Step 6.1 follow-up: bake EasyOCR English weights (CRAFT + `english_g2`) into the image so first prescan after rebuild does not stall on GitHub |
| 2026-08-20 | Step 6.1: bake `numpy>=1.26,<2` and `trackers==2.6.0 --no-deps` into the Docker image; compose no longer pip-installs on start |
| 2026-08-20 | Stabilization S20 fixed: live monitor blank was Chromium rejecting HEVC `_processed.mp4` (Range/proxy OK); encoder prefers H.264 for browser; S13 fragmented MP4 retained |
| 2026-08-19 | Stabilization S10 partial: refined no-profile line fallback with dominant-slope parallel fitting and extra tests; further tuning needed on sample camera views before closure |
| 2026-08-19 | Stabilization S10 fixed: no-profile prescan line proposal now uses deterministic frame-guided edge fitting with bounds-safe clamping and cue-based confidence; profile override precedence unchanged |
| 2026-08-19 | Stabilization S15 fixed: 15-min CSV contract/output aligned to `date` + `HH:MM` windows; UI parser groups by `date+window+class` to avoid cross-day merge |
| 2026-08-19 | Stabilization S13–S14 fixed: in-progress processed MP4 fragmented for streaming during PROCESSING; MOVING_EVENT now carries timestamp/source/confidence + video_pts_ms |
| 2026-08-19 | Step 5 complete: QA verified intake→prescan→confirm READY→PROCESSING→COMPLETED→aggregate and validated `_15min.csv` evidence + path-mapping negative-path repro |
| 2026-08-19 | Stabilization S11–S12: JobStatus `created_at`, `video_duration_sec`, `processing_duration_sec` on GET /jobs and GET /jobs/{id} |
| 2026-08-19 | Stabilization S08: prescan CLI 6.7s → 4.6s on `hiv000001_inframe.mp4` (OSD frame probe + faster corner OCR); S07 metadata unchanged |
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
