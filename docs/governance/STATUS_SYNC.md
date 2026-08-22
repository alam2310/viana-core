# Status document sync (mandatory)

## Why drift keeps happening

ViAna tracks post-v0.1 work in **three coupled surfaces**:

| File | Role |
|------|------|
| [`STABILIZATION_BACKLOG.md`](../steps/STABILIZATION_BACKLOG.md) | **SoT** for Seq `SNN` status (`open` / `fixed` / `parked`) |
| [`TRACKER.md`](../steps/TRACKER.md) | Step 6.x checklist + **mirror** of the Seq table (§ 4.stab) |
| [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) | Human/agent landing page — **summary only** (open Seq list, current focus) |

Drift occurs when an implementation chat:

1. Updates **only** the backlog (or only TRACKER) when closing a Seq.
2. Adds **S34+** rows to the backlog but never extends the TRACKER mirror table.
3. Leaves stale one-liners (`4.stab`, **Next**, changelog “open Seq now **S32**”) from an earlier sync.
4. Uses the **planning/governance chat** to sync docs while **feature chats** keep landing code without the doc checklist.

There is **one write path** for Seq status — see below.

---

## Source of truth

See [`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md):

- **Seq status** → `STABILIZATION_BACKLOG.md` execution path table (authoritative).
- **Step 6.x items** → `TRACKER.md` § Step 6 (authoritative for 6.1–6.13).
- **Current focus / open work summary** → `PROJECT_STATUS.md` (must match backlog + tracker; not a third status source).

`TRACKER.md` § “Stabilization execution path” is a **mirror**. It must match the backlog row-for-row.

---

## Mandatory checklist — closing or opening a Seq

When a Seq moves to `fixed`, `open`, `parked`, or `deferred`, the **same commit** (or the commit the user approves after testing) must include **all** of:

| # | File | Action |
|---|------|--------|
| 1 | `STABILIZATION_BACKLOG.md` | Update execution path row + summary counts + row detail; changelog line |
| 2 | `TRACKER.md` | Update § 4.stab one-liner; mirror table row(s); add new Seq rows if SNN is new; changelog |
| 3 | `PROJECT_STATUS.md` | Update **Current focus**, **Open Seq**, **Next** bullets; changelog if milestone |
| 4 | `STEP_6_HARDENING.md` | Log line if the Seq maps to a Step item (e.g. S09 → 6.7) |
| 5 | — | Run `make check-status-sync` — **must pass** before commit |

**Order:** edit backlog first → copy status to TRACKER mirror → update PROJECT_STATUS summary.

---

## Mandatory checklist — closing a Step 6.x item

Same commit must update:

1. `TRACKER.md` § Step 6 row → ✅  
2. `STEP_6_HARDENING.md` log  
3. `PROJECT_STATUS.md` if it changes “Still open on Step 6”  
4. `make check-status-sync` passes  

If promoted from idea dump: also update `IDEA_DUMP.md` promoted table (not required for sync script).

---

## Adding a new Seq (SNN)

1. Append row to **backlog** execution path (SoT).  
2. Append matching row to **TRACKER** mirror table (do not rely on “sync later”).  
3. Update **4.stab** one-liner and **PROJECT_STATUS** open list.  
4. Run `make check-status-sync`.

Never add a Seq in TRACKER alone.

---

## Automation

```bash
make check-status-sync
```

Script: [`scripts/check_status_sync.py`](../../scripts/check_status_sync.py)

Fails if:

- Any Seq status in TRACKER mirror ≠ backlog  
- Backlog lists open Seq not mentioned in PROJECT_STATUS **Open Seq** line  
- Summary counts in backlog ≠ derived from the execution table  

Run locally before commit; CI may run the same check on docs changes.

---

## Commit policy (feature chats)

Implementation agents:

1. Test locally.  
2. Apply this checklist in the **same patch** as the fix.  
3. Run `make check-status-sync`.  
4. **Do not commit** until the user confirms — but **do** include doc updates in the uncommitted patch so drift does not land on `main` without docs.

Planning/governance chats may land doc-only sync commits when fixing drift.
