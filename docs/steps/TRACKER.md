# Step tracker (living)

**Last updated:** 2026-08-19  
**Current Step:** **5** — E2E verification (`_15min.csv`)  
**Canonical plan:** [`PLAN.md`](PLAN.md)  
**Agent checklist:** [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md)

> Update this file when a Step changes status. Agents: do not rely on chat memory.

---

## Status overview

| Step | Name | Status | Owner chat | Started | Completed |
|------|------|--------|------------|---------|-----------|
| **1** | UX discovery & design | ✅ Complete | UX discovery | 2026-08-19 | 2026-08-19 |
| **2** | Contracts & API foundation | ✅ Complete | Contract + API | 2026-08-19 | 2026-08-19 |
| **3** | Engine prescan & orchestrator | ✅ Complete | Engine + workers | 2026-08-19 | 2026-08-19 |
| **4** | UI implementation | ✅ Complete | UI v2 | 2026-08-19 | 2026-08-19 |
| **5** | E2E verification | 🔄 In progress | UI v2 or QA | 2026-08-19 | — |
| **6** | Hardening backlog | ⬜ Not started | Per item | — | — |

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
| **4.stab** Stabilization path | 🔄 8 fixed / 6 open | [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) S01–S08, S10–S15 (S09 → 6.7) |

### Stabilization execution path (follow in order)

| Seq | Work | Lane | Status |
|-----|------|------|--------|
| S01 | F004 — verify preview JPEG after restart | B | fixed |
| S02 | F003 — `GET /artifacts/{id}/source.mp4` | B | fixed |
| S03 | F003 — Next.js source proxy | A | fixed |
| S04 | F003 — prescan scrub via video seek (not prescan API) | A | fixed |
| S05 | F003 — Re-scan OCR only + docs | A/B | fixed |
| S06 | F005 — EasyOCR triage | C | open |
| S07 | F001 — corner ROI OCR (**Step 5 blocker**) | C | fixed |
| S08 | F002 — prescan latency | C | fixed |
| S10 | F007 — horizon/counting line proposal | C | open |
| S11 | F008 — `created_at` + sortable submitted time in API | B/D | open |
| S12 | F009 — `video_duration_sec` + `processing_duration_sec` in API | B/D | open |
| S13 | F010 — streamable growing `_processed.mp4` during processing | B/C | open |
| S14 | F011 — emit `MOVING_EVENT` without `telemetry_detail` gate + timestamp | C | open |
| S15 | F012 — 15-min CSV: add `date`, HH:MM window columns | B/D | open |
| ~~S09~~ | F006 — intake path validation | B | **deferred → 6.7** |

---

## Step 5 — E2E verification

**Gate status:** Unblocked (`S07` fixed). Continue S10–S15 polish in parallel with Step 5.

| Check | Status |
|-------|--------|
| Intake → confirm → READY → COMPLETED | ⬜ |
| `_15min.csv` verified | ⬜ |
| Evidence `verification/5_15min_results.md` | ⬜ |

---

## Step 6 — Hardening

| Item | Work | Status |
|------|------|--------|
| 6.1 | Docker image bake | ⬜ |
| 6.2 | Pause / resume UX | ⬜ |
| 6.3 | Faster cancel | ⬜ |
| 6.4 | Playwright | ⬜ |
| 6.5 | Extra camera clip | ⬜ |
| 6.6 | GPU CI | ⬜ |
| 6.7 | Container host path access + API intake path validation (S09 / F006) | ⬜ |

---

## Changelog

| Date | Change |
|------|--------|
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
