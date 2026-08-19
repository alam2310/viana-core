# Step tracker (living)

**Last updated:** 2026-08-19  
**Current Step:** **2** — Contracts & API foundation  
**Canonical plan:** [`PLAN.md`](PLAN.md)  
**Agent checklist:** [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md)

> Update this file when a Step changes status. Agents: do not rely on chat memory.

---

## Status overview

| Step | Name | Status | Owner chat | Started | Completed |
|------|------|--------|------------|---------|-----------|
| **1** | UX discovery & design | ✅ Complete | UX discovery | 2026-08-19 | 2026-08-19 |
| **2** | Contracts & API foundation | ⬜ Not started | New — Contract + API | — | — |
| **3** | Engine prescan & orchestrator | ⬜ Not started | New — Engine + workers | — | — |
| **4** | UI implementation | ⬜ Not started | New — UI v2 | — | — |
| **5** | E2E verification | ⬜ Not started | UI v2 or QA | — | — |
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

## Step 2 — Contracts & API foundation

| Item | Status |
|------|--------|
| P1–P6 schemas + TS + fixtures | ⬜ |
| `POST /jobs/intake` (G16) | ⬜ |
| `PATCH /jobs/{id}/prescan` (G17) | ⬜ |
| Proposed + confirmed fields (G1, G15) | ⬜ |
| Metadata validation (G4) | ⬜ |
| `output_dir` (G20) | ⬜ |
| JobStatus state machine stubs (G14) | ⬜ |

Detail: [`STEP_2_CONTRACTS_AND_API.md`](STEP_2_CONTRACTS_AND_API.md)

---

## Step 3 — Engine prescan & orchestrator

| Item | Status |
|------|--------|
| Prescan worker queue (G13) | ⬜ |
| Dark-frame skip (G7) | ⬜ |
| Frame preview (G8) | ⬜ |
| GPU gate — `READY` only | ⬜ |
| Auto-aggregate (G12) | ⬜ |
| Partial MP4 serving (G19) | ⬜ |
| ETA + crossings (G9) | ⬜ |

Detail: [`STEP_3_ENGINE_AND_ORCHESTRATOR.md`](STEP_3_ENGINE_AND_ORCHESTRATOR.md)

---

## Step 4 — UI implementation

| Sub-step | Status | Surface |
|----------|--------|---------|
| 4.1 Intake browser + queue table | ⬜ | `apps/web/` |
| 4.2 Prescan review modal | ⬜ | `features/prescan/` |
| 4.3 Live monitor + telemetry | ⬜ | `features/telemetry/` |
| 4.4 Completed artifacts | ⬜ | dashboard |
| 4.5 Polish + docs | ⬜ | `apps/web/`, `docs/ui/` |

---

## Step 5 — E2E verification

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
| 6.7 | Container host path access | ⬜ |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-19 | Six-step plan: split backend into Step 2 (contracts/API) + Step 3 (engine/workers) |
| 2026-08-19 | Step 1 complete |
| 2026-08-19 | Steps 1–5 tracker created |
