# Stabilization workflow (Step 4 → Step 5 gate)

**Purpose:** Record bugs and optimizations found during Step 4 acceptance **without** starting Step 5 or reopening full Steps 2–3.

**Living backlog:** [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md)  
**Execution path:** [`STABILIZATION_BACKLOG.md` § Execution path](STABILIZATION_BACKLOG.md#execution-path) — **S01→S09**, single ordered sequence  
**Triage owner:** Planning chat (human + coordinator agent) — assigns lane and kickoff prompt.

---

## When this applies

- Step 4 UI is built (or nearly built) but prescan / queue / confirm flow is not ready for E2E.
- Issues may touch `apps/web/`, `src/orchestrator/`, or `src/viana/stages/prescan.py`.
- **Step 5 is blocked** until all **blocker** items in the backlog are ✅ or explicitly deferred.

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

| Field | Value |
|-------|-------|
| `TRACKER.md` **Current Step** | `4 (stabilization)` |
| Step 4 | 🔄 In progress *or* ✅ with open blockers noted |
| Step 5 | ⬜ Blocked — see `STABILIZATION_BACKLOG.md` |

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
3. Get copy-paste **patch prompt** for Step 3 / Step 4 / Step 2 chat
4. Confirm **S07** (F001 blocker) `fixed` or `deferred` → unblock Step 5

Coordinator does **not** implement fixes; it triages and prompts.

---

## Step 5 entry criteria (reminder)

All rows with **Blocker** = `yes` must be `fixed` or `deferred` (with user approval).

Minimum flow that must work:

- Intake → prescan → `AWAITING_REVIEW` with usable `proposed_*`
- Review → confirm → `READY`
- Optional: short run to `PROCESSING` without prescan regressions

---

## Related docs

- [`STEP_4_UI_IMPLEMENTATION.md`](STEP_4_UI_IMPLEMENTATION.md)
- [`STEP_3_ENGINE_AND_ORCHESTRATOR.md`](STEP_3_ENGINE_AND_ORCHESTRATOR.md)
- [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md)
