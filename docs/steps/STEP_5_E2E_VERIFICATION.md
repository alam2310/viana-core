# Step 5 — E2E verification (15-min CSV)

| Field | Value |
|-------|-------|
| **Status** | ⬜ Not started — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | UI v2 continuation or **QA** |
| **Blocks** | Step 6 (optional) |
| **Blocked by** | Step 4 ✅ |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 5.

---

## Objective

Prove end-to-end: intake → prescan → confirm → `READY` → process → auto-aggregate → valid `{stem}_15min.csv`.

---

## Test plan

### 5.1 — Happy path

| # | Action | Expected |
|---|--------|----------|
| 1 | Intake test clip | Job `PRESCAN_PENDING` / `PRESCAN_RUNNING` |
| 2 | Review + confirm metadata (`DD-MM-YYYY`, `HH:MM:SS`) | `READY` |
| 3 | Worker runs → `COMPLETED` | `_events.csv`, `time_map.json` |
| 4 | Auto-aggregate | `_15min.csv` non-empty |

### 5.2 — Negative path

Confirm without wall-clock → document empty `_15min.csv` + operator message.

### 5.3 — Regression

Orchestrator + prescan tests green; prior HTTP E2E script still passes.

---

## Evidence

`docs/steps/verification/5_15min_results.md`

---

## Exit criteria

- [ ] 5.1–5.3 documented
- [ ] `AGENT_PROGRESS.md` Step 5 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Renumbered from Step 4 |
