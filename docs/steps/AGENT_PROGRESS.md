# Agent progress checklist (Steps 1–6)

**Read this when starting or finishing any post-v0.1 Step.** Do not rely on chat memory.

| Doc | When |
|-----|------|
| [`TRACKER.md`](TRACKER.md) | Every session |
| This file | Step start / complete / skip |
| [`STEP_N_*.md`](.) | Step scope + exit criteria |
| [`IDEA_DUMP.md`](IDEA_DUMP.md) | **Do not read for work.** Parking lot for later human review. Ignore unless the user names an idea ID. |

**Naming:** **Phase 0–9** = build (complete). **Step 1–6** = post-v0.1 (active).

---

## On starting any Step

1. Read [`TRACKER.md`](TRACKER.md) — confirm current Step and prerequisites.
2. Read `STEP_N_*.md` + kickoff in [`KICKOFF_PROMPTS.md`](KICKOFF_PROMPTS.md).
3. **`TRACKER.md`:** status → 🔄 In progress; **Started** date; **Current Step** at top.
4. **`PROJECT_STATUS.md`:** update **Current focus**.
5. Append to Step file **Log**.

---

## On completing Step 1 — UX discovery & design

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark Step 1 ✅ | `TRACKER.md` |
| 2 | Check deliverables 1.1–1.5 | `TRACKER.md` § Step 1 |
| 3 | `DISCOVERY.md` §6 signed off | `docs/ui/DISCOVERY.md` |
| 4 | Set **Current Step** → **2** | `TRACKER.md` |
| 5 | Changelog + focus | `PROJECT_STATUS.md` |
| 6 | Log | `STEP_1_UX_DESIGN.md` |
| 7 | Work items in Step 2 + Step 3 files | `STEP_2_*`, `STEP_3_*` |

**No code** in Step 1.

---

## On completing Step 2 — Contracts & API

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark Step 2 ✅ | `TRACKER.md` |
| 2 | P1–P6 + G16/G17/G1/G4/G14/G15/G20 ✅ | `STEP_2_CONTRACTS_AND_API.md` |
| 3 | Set **Current Step** → **3** | `TRACKER.md` |
| 4 | Changelog if schemas/routes changed | `PROJECT_STATUS.md` |
| 5 | Log | `STEP_2_CONTRACTS_AND_API.md` |
| 6 | Tests pass | `tests/orchestrator/` |

**Surfaces:** `packages/contracts/`, `src/orchestrator/routes/`, `job.py`. Not engine prescan depth (Step 3).

---

## On completing Step 3 — Engine & orchestrator

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark Step 3 ✅ | `TRACKER.md` |
| 2 | G7/G8/G9/G12/G13/G19 + worker gates ✅ | `STEP_3_ENGINE_AND_ORCHESTRATOR.md` |
| 3 | Set **Current Step** → **4** | `TRACKER.md` |
| 4 | Changelog | `PROJECT_STATUS.md` |
| 5 | Log | `STEP_3_ENGINE_AND_ORCHESTRATOR.md` |
| 6 | Tests pass | `tests/viana/test_prescan.py`, orchestrator |

**Surfaces:** `src/viana/stages/prescan.py`, `workers/pool.py`. Not `apps/web/`.

---

## Stabilization (Step 4 → Step 5 gate; polish continues in Step 6)

Step **5 is complete**. Remaining open Seq rows are **polish** during Step 6.

1. Follow **Execution path** in [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md#execution-path) (respect **Depends**). SoT for Seq status is the backlog, not chat memory.
2. Log new issues: add **Seq** + **ID** rows per [`STABILIZATION.md`](STABILIZATION.md).
3. Historical gate: Step 5 required **S07** (F001) `fixed` or `deferred` — done.
4. Assign next open Seq to a lane chat (A UI / B API / C engine / D contract).

---

## On completing Step 4 — UI implementation

Update tracker after **each** sub-step 4.1–4.5.

| # | Action | File(s) |
|---|--------|---------|
| 1 | Sub-steps 4.1–4.5 ✅ | `TRACKER.md` § Step 4 |
| 2 | Mark Step 4 ✅ only if stabilization blockers clear (see `STABILIZATION_BACKLOG.md`) | `TRACKER.md` |
| 3 | Set **Current Step** → **5** | `TRACKER.md` |
| 4 | Changelog | `PROJECT_STATUS.md` |
| 5 | Sync component map | `docs/ui/COMPONENT_MAP.md` |

**Surface:** `apps/web/` only.

---

## On completing Step 5 — E2E verification

| # | Action | File(s) |
|---|--------|---------|
| 1 | Checks ✅ | `TRACKER.md` § Step 5 |
| 2 | Mark Step 5 ✅ | `TRACKER.md` |
| 3 | Set **Current Step** → **6** | `TRACKER.md` |
| 4 | Changelog — 15-min CSV result | `PROJECT_STATUS.md` |
| 5 | Evidence | `verification/5_15min_results.md` |

---

## On completing Step 6 — Hardening (per item)

Per item **6.1–6.11** in `TRACKER.md`. Mark Step 6 ✅ when all done or deferred.

**Done through 2026-08-22:** 6.1, 6.3, 6.7–**6.11** (I001–I003, I006). **Still open:** 6.2, 6.4, 6.5, 6.6.

---

## Quick reference — files agents touch

| File | Step |
|------|------|
| `TRACKER.md`, `PROJECT_STATUS.md` | All |
| `packages/contracts/*` | 2 |
| `src/orchestrator/routes/`, models | 2–3 |
| `src/viana/stages/prescan.py` | 3 |
| `apps/web/*` | 4 |
| `docs/steps/STABILIZATION_BACKLOG.md` | 4 stabilization (append defects) |
| `docs/steps/IDEA_DUMP.md` | **Never** (unless user names an idea ID to promote/implement) |
| `verification/5_15min_results.md` | 5 |
| `docs/ui/REDESIGN.md` | 1 (write), 4 (read) |

---

## Commit message examples

```
Complete Step 2: JobStatus lifecycle and intake API
Complete Step 3: prescan worker queue and auto-aggregate
Step 4.2: prescan review modal with confirm summary
Stabilization S04: prescan scrub via video seek (F003)
Stabilization S07: corner ROI OCR (F001 blocker)
Complete Step 5: verify 15min CSV on hiv000001_inframe
Step 6.1: bake trackers into Docker image
```
