# Stabilization workflow (Step 4 → Step 5 gate; Step 6 polish)

**Purpose:** Record bugs and optimizations found during Step 4 acceptance **without** starting Step 5 or reopening full Steps 2–3. After Step 5 completed, the same backlog continues for **Step 6 polish**.

**Living backlog (SoT for Seq status):** [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md)  
**Execution path:** full table in backlog (S01–S08, S10–S33; S09 → Step 6.7)  
**Tracker mirror:** [`TRACKER.md`](TRACKER.md) § 4.stab — keep in sync when closing Seq  
**Triage owner:** Planning chat (human + coordinator agent) — assigns lane and kickoff prompt.

**Status (2026-08-22):** Step 5 ✅. Seq polish complete (S29–S33). Parked: S20/S24.

---

## When this applies

- **Historical:** Step 4 UI built but prescan / queue / confirm not ready for E2E; Step 5 blocked on backlog **blocker** rows.
- **Now:** New defects during Step 6 still go on the backlog as polish (`Blocker: no`) unless they reopen a product gate.

---

## Agent rules (all implementation chats)

### 1. Log before you fix (or when you cannot fix in-lane)

When you find a defect during Step 4 stabilization:

1. **Add a row** to [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) before or immediately after attempting a fix.
2. Assign the next **Seq** (`SNN`) in the execution path table and link to an **ID** (`FNNN`) — reuse an existing ID if the work is a follow-on phase (e.g. F003 S02–S05).
3. Set **Lane** (A/B/C/D) — see below. If unsure, set `TBD` and note in row detail.
4. Set **Blocker** = `yes` if Step 5 cannot pass until fixed; `no` for polish.
5. Set **Depends** when work requires a prior Seq (e.g. UI proxy depends on API endpoint).
6. Do **not** mark Step 5 started or update `TRACKER.md` Step 5 until blocker rows are cleared.
7. Pick up work from the **first open Seq** you are in-lane for, unless coordinator assigns a specific row.

### 2. Stay in your lane

| Lane | Owns | Paths | Chat |
|------|------|-------|------|
| **A** | UI only | `apps/web/`, `docs/ui/` | Step 4 |
| **B** | API / orchestrator | `src/orchestrator/` (routes, pool, models) | Step 3 patch |
| **C** | Engine prescan | `src/viana/stages/prescan.py`, `cli.py`, prescan tests | Step 3 patch |
| **D** | Contract | `packages/contracts/`, `api_contracts.md`, `job.py` | Step 2 patch |

- **Lane A** agents must not change engine or orchestrator except trivial client bugs.
- **Lane B/C** agents must not edit `apps/web/`.
- **Lane D** is schema-first per `docs/governance/CONTRACT_SYNC.md`; then B/C/A consume.

### 3. Update the backlog when done

When a fix lands:

- Set **Status** → `fixed`
- Fill **Fix commit / PR** and **Verified by**
- If the issue was wrongly triaged, update **Lane** and add a one-line note

### 4. Do not expand scope

- Stabilization fixes **only** items on the backlog or explicitly assigned in chat.
- New product features → new Step 4 sub-step or defer to Step 6.
- Do not reopen full Step 3 “implement everything again.”

### 5. Tracker state during stabilization

**Historical (Step 4 → Step 5 gate):**

| Field | Value |
|-------|-------|
| `TRACKER.md` **Current Step** | `4 (stabilization)` |
| Step 4 | 🔄 In progress *or* ✅ with open blockers noted |
| Step 5 | ⬜ Blocked — see `STABILIZATION_BACKLOG.md` |

**Now (2026-08-22+):**

| Field | Value |
|-------|-------|
| `TRACKER.md` **Current Step** | **6** |
| Steps 1–5 | ✅ |
| Open Seq | none (polish done) — parked S20/S24; SoT [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) |

---

## Backlog row template

Add to **Execution path** in `STABILIZATION_BACKLOG.md`:

```markdown
| S0NN | F0NN | B | no | S0M | short title | open |
```

Add repro / expected / files to **Row detail**. For multi-phase fixes, keep one **ID** (e.g. F003) across several **Seq** rows.

---

## Coordinator workflow (this / planning chat)

Human returns here to:

1. Review **Execution path** in `STABILIZATION_BACKLOG.md` (S01→SNN)
2. Assign the next open **Seq** to a lane chat (respect **Depends**)
3. Get copy-paste **patch prompt** for UI / API / engine chat
4. Keep `TRACKER.md` § 4.stab mirror in sync when closing Seq
5. *(Historical)* Confirm **S07** fixed → unblock Step 5 — **done**

Coordinator does **not** implement fixes; it triages and prompts.

---

## Step 5 entry criteria (reminder — met)

All rows with **Blocker** = `yes` must be `fixed` or `deferred` (with user approval). **S07 fixed; Step 5 complete.**

Minimum flow that must work (verified):

- Intake → prescan → `AWAITING_REVIEW` with usable `proposed_*`
- Review → confirm → `READY`
- Run through `COMPLETED` + `_15min.csv` — see `verification/5_15min_results.md`

---

## Related docs

- [`STEP_4_UI_IMPLEMENTATION.md`](STEP_4_UI_IMPLEMENTATION.md)
- [`STEP_3_ENGINE_AND_ORCHESTRATOR.md`](STEP_3_ENGINE_AND_ORCHESTRATOR.md)
- [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md)
