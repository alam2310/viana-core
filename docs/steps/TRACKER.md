# Step tracker (living)

**Last updated:** 2026-08-21  
**Current Step:** **6** — Hardening backlog  
**Canonical plan:** [`PLAN.md`](PLAN.md)  
**Agent checklist:** [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md)  
**Idea dump (manual review only):** [`IDEA_DUMP.md`](IDEA_DUMP.md)

> Update this file when a Step changes status. Agents: do not rely on chat memory.

### Idea dump — not a work queue

[`IDEA_DUMP.md`](IDEA_DUMP.md) is a **parking lot** for ideas captured outside active Steps. Use it for **human review after current work**, then promote chosen items into Step 6 / stabilization / a new Step.

**Agents: do not pick work from `IDEA_DUMP.md` unless the user explicitly names an idea ID.** Do not treat it as the current Step, a blocker, or a source of Seq/F IDs.

---

## Status overview

| Step | Name | Status | Owner chat | Started | Completed |
|------|------|--------|------------|---------|-----------|
| **1** | UX discovery & design | ✅ Complete | UX discovery | 2026-08-19 | 2026-08-19 |
| **2** | Contracts & API foundation | ✅ Complete | Contract + API | 2026-08-19 | 2026-08-19 |
| **3** | Engine prescan & orchestrator | ✅ Complete | Engine + workers | 2026-08-19 | 2026-08-19 |
| **4** | UI implementation | ✅ Complete | UI v2 | 2026-08-19 | 2026-08-19 |
| **5** | E2E verification | ✅ Complete | UI v2 or QA | 2026-08-19 | 2026-08-19 |
| **6** | Hardening backlog | 🔄 In progress | Per item | 2026-08-20 | — |

**Status legend:** ⬜ Not started · 🔄 In progress · ✅ Complete · ⏸ Skipped · ❌ Cancelled

---

## Step 1 — UX discovery & design ✅

| Deliverable | Status | Path |
|-------------|--------|------|
| 1.1 Discovery Q&A | ✅ | `docs/ui/DISCOVERY.md` |
| 1.2 Task-type matrix | ✅ | `DISCOVERY.md` §3 |
| 1.3 Redesign spec | ✅ | `docs/ui/REDESIGN.md` |
| 1.4 Flows / component map | ✅ | `USER_FLOWS.md`, `COMPONENT_MAP.md` |
| 1.5 Backend gap list | ✅ | `STEP_2_*` + `STEP_3_*` work items |

---

## Step 2 — Contracts & API foundation ✅

| Item | Status |
|------|--------|
| P1–P6 schemas + TS + fixtures | ✅ |
| `POST /jobs/intake` (G16) | ✅ |
| `PATCH /jobs/{id}/prescan` (G17) | ✅ |
| Proposed + confirmed fields (G1, G15) | ✅ |
| Metadata validation (G4) | ✅ |
| `output_dir` (G20) | ✅ |
| JobStatus state machine stubs (G14) | ✅ |

Detail: [`STEP_2_CONTRACTS_AND_API.md`](STEP_2_CONTRACTS_AND_API.md)

---

## Step 3 — Engine prescan & orchestrator

| Item | Status |
|------|--------|
| Prescan worker queue (G13) | ✅ |
| Dark-frame skip (G7) | ✅ |
| Frame preview (G8) | ✅ |
| GPU gate — `READY` only | ✅ |
| Auto-aggregate (G12) | ✅ |
| Partial MP4 serving (G19) | ✅ |
| ETA + crossings (G9) | ✅ |

Detail: [`STEP_3_ENGINE_AND_ORCHESTRATOR.md`](STEP_3_ENGINE_AND_ORCHESTRATOR.md)

---

## Step 4 — UI implementation

| Sub-step | Status | Surface |
|----------|--------|---------|
| 4.1 Intake browser + queue table | ✅ | `features/project/`, `features/intake/`, `features/queue/` |
| 4.2 Prescan review modal | ✅ | `features/prescan/prescan-review-modal.tsx` |
| 4.3 Live monitor + telemetry | ✅ | `features/monitor/`, `features/telemetry/` |
| 4.4 Completed artifacts | ✅ | `features/queue/job-queue-table.tsx` |
| 4.5 Polish + docs | ✅ | `apps/web/`, `docs/ui/COMPONENT_MAP.md` |
| **4.stab** Stabilization path | 🔄 S10/S21 fixed; S22–S23 open | [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) |

### Stabilization execution path (follow in order)

| Seq | Work | Lane | Status |
|-----|------|------|--------|
| S01 | F004 — verify preview JPEG after restart | B | fixed |
| S02 | F003 — `GET /artifacts/{id}/source.mp4` | B | fixed |
| S03 | F003 — Next.js source proxy | A | fixed |
| S04 | F003 — prescan scrub via video seek (not prescan API) | A | fixed |
| S05 | F003 — Re-scan OCR only + docs | A/B | fixed |
| S06 | F005 — EasyOCR triage | C | fixed |
| S07 | F001 — corner ROI OCR (**Step 5 blocker**) | C | fixed |
| S08 | F002 — prescan latency | C | fixed |
| S10 | F007 — horizon/counting line proposal | C | **fixed** |
| S11 | F008 — `created_at` + sortable submitted time in API | B/D | fixed |
| S12 | F009 — `video_duration_sec` + `processing_duration_sec` in API | B/D | fixed |
| S13 | F010 — streamable growing `_processed.mp4` during processing | B/C | fixed |
| S14 | F011 — emit `MOVING_EVENT` without `telemetry_detail` gate + timestamp | C | fixed |
| S15 | F012 — 15-min CSV: add `date`, HH:MM window columns | B/D | fixed |
| S19 | F016 — queue video length / ETA inflation + MPEG-PS probe | A/B/C | fixed |
| S21 | F017 — adaptive OSD OCR when text is outside corner ROIs | C | **fixed** |
| S20 | F010 follow-on — browser live monitor play of in-progress MP4 (H.264) | A/B | **parked** → S24 |
| S24 | Park live-monitor partial MP4 UI; crossings immediate | A | parked |
| ~~S09~~ | F006 — intake path validation | B | **deferred → 6.7** |

---

## Step 5 — E2E verification

**Gate status:** Unblocked (`S07` fixed). Continue S10–S15 polish in parallel with Step 5.

| Check | Status |
|-------|--------|
| Intake → confirm → READY → COMPLETED | ✅ |
| `_15min.csv` verified | ✅ |
| Evidence `verification/5_15min_results.md` | ✅ |

---

## Step 6 — Hardening

| Item | Work | Status |
|------|------|--------|
| 6.1 | Docker image bake (`trackers==2.6.0 --no-deps` + `numpy<2`) | ✅ |
| 6.2 | Pause / resume UX | ⬜ |
| 6.3 | Faster cancel | ⬜ |
| 6.4 | Playwright | ⬜ |
| 6.5 | Extra camera clip | ⬜ |
| 6.6 | GPU CI | ⬜ |
| 6.7 | Container host path access + API intake path validation (S09 / F006) | ⬜ |
| 6.8 | Job details: drop Live Monitor widget/action; Live Crossings in details; row click opens details (**I001**) | ⬜ |
| 6.9 | Disable in-process OSD OCR; wall-clock/location from confirmed prescan only (**I003**) | ⬜ |
| 6.10 | Bind live crossing total to existing `crossing_count` (JobStatus / WS PROGRESS), not session WS list length (**I002**) | ⬜ |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-21 | **S10 (F007) fixed** — road-band horizon/counting proposal on real clips; profile override kept; `test_prescan.py` 29 passed |
| 2026-08-21 | Promoted **I002 → 6.10** — UI uses existing `crossing_count` (confirmed on JobStatus / WS PROGRESS). I004 remains demoted. |
| 2026-08-21 | Promoted idea dump **I001 → 6.8** (job details / drop Live Monitor) and **I003 → 6.9** (no process-loop OSD OCR). I002 stays dump (P3, API check). I004 demoted. |
| 2026-08-21 | Added [`IDEA_DUMP.md`](IDEA_DUMP.md) — manual-review parking lot; agents must not self-assign from it |
| 2026-08-21 | S21 follow-up: `test_video.mp4` plus-as-colon clock (`08:38+31`) and unhyphenated location join; prior inframe/shimoga/night reads unchanged |
| 2026-08-20 | S21 (F017) fixed — adaptive OSD bands + clock/date salvage (spaced/`"` colons, year `7074`→`2024`, mixed-polarity location); UI retest OK; no `hiv000001_inframe` S07/S08 regression |
| 2026-08-20 | S19 (F016) fixed — MPEG-PS/DVR header duration corrected via ffprobe packet count; queue video length/ETA units documented |
| 2026-08-20 | Step 6.1 follow-up: bake EasyOCR English weights into the image; prescan timeout fails the job instead of hanging |
| 2026-08-20 | Step 6.1 complete: bake `numpy>=1.26,<2` and `trackers==2.6.0 --no-deps` into the image; compose starts uvicorn only |
| 2026-08-19 | S10 moved to in_progress (partial): dominant-slope/parallel-band fallback refinement landed with tests, but sample-view placement still needs tuning before close |
| 2026-08-19 | S10 fixed — frame-guided prescan line proposal now uses deterministic edge/line cues when no profile matches; bounds-safe fallback retained; tests added for confidence uplift, determinism, and invalid-frame fallback |
| 2026-08-19 | S15 fixed — `_15min.csv` contract/output aligned (`date` + `HH:MM` windows), parser grouped by `date+window+class`, tests updated |
| 2026-08-19 | Step 5 QA evidence captured: intake→prescan→confirm(READY)→PROCESSING→COMPLETED→aggregate; `_15min.csv` verified with row/header samples; path-mapping negative path repro logged |
| 2026-08-19 | S11–S12 (F008/F009) fixed — JobStatus timing fields on GET /jobs and GET /jobs/{id} |
| 2026-08-19 | S08 (F002) fixed — prescan CLI 6.7s → 4.6s on `hiv000001_inframe.mp4`; S07 OCR fields unchanged |
| 2026-08-19 | S09 deferred to 6.7; S10 (F007) line proposal added to stabilization path |
| 2026-08-19 | Stabilization execution path S01–S09 (merged findings + scrub plan) |
| 2026-08-19 | Stabilization workflow + backlog; Step 5 blocked until prescan fixes |
| 2026-08-19 | Step 4 complete: API-driven queue, intake browser, prescan review, live monitor, structured telemetry |
| 2026-08-19 | Step 3 complete: prescan worker queue, dark-frame skip, auto-aggregate, partial MP4, ETA/crossings |
| 2026-08-19 | Step 2 complete: JobStatus lifecycle, intake + prescan confirm APIs |
| 2026-08-19 | Six-step plan: split backend into Step 2 (contracts/API) + Step 3 (engine/workers) |
| 2026-08-19 | Step 1 complete |
| 2026-08-19 | Steps 1–5 tracker created |
