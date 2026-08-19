# Step 4 — E2E verification (15-min CSV)

| Field | Value |
|-------|-------|
| **Status** | ⬜ Not started — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | UI v2 continuation or new **QA** |
| **Blocks** | Step 5 (optional) |
| **Blocked by** | Step 3 |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 4.

---

## Objective

Prove: prescan metadata → job run → `time_map.json` → aggregate → valid `{stem}_15min.csv`.

---

## Test plan

### 4.1 — Happy path

| # | Action | Expected |
|---|--------|----------|
| 1 | Prescan test clip | OCR or manual time/date |
| 2 | Submit with `metadata` | Job COMPLETED |
| 3 | Check output dir | `_events.csv`, `.time_map.json` |
| 4 | Aggregate | `_15min.csv` written |
| 5 | Inspect CSV | Non-empty rows; 15-min windows |

### 4.2 — Negative path

Submit without wall-clock → aggregate → document empty/partial behavior + UI empty-state.

### 4.3 — Regression

Existing E2E script and `pytest tests/viana/test_aggregate.py` pass.

---

## If `_15min.csv` is empty

| Symptom | Owner |
|---------|-------|
| Missing `metadata` on job | UI / API |
| No `time_map.json` | Engine |
| Aggregate, no rows | Engine |

---

## Evidence

Write `docs/steps/verification/4_15min_results.md` with clip path, metadata used, CSV snippet.

---

## Exit criteria

- [ ] 4.1–4.3 documented in tracker
- [ ] `AGENT_PROGRESS.md` Step 4 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Step created (numbered Steps 1–5) |
