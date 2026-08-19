# Step 3 — Engine prescan & orchestrator workers

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | **New** — Engine + API (workers) |
| **Blocks** | Step 4 |
| **Blocked by** | Step 2 ✅ |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 3.

---

## Objective

Implement **prescan engine behavior** and **orchestrator workers** on top of Step 2 contracts/API: bulk prescan queue, GPU gate (`READY` only), auto-aggregate, frame preview, live media.

---

## Work items

| ID | Item | Owner | Status |
|----|------|-------|--------|
| G7 | Auto-skip dark/blocked opening frames in prescan sampler | Engine | ✅ |
| G8 | Live frame preview on scrub (prescan re-run or frame endpoint) | Engine + API | ✅ |
| G9 | ETA + crossing count on job status / telemetry | API + WS | ✅ |
| G12 | Auto-aggregate on job `COMPLETED` | Orchestrator | ✅ |
| G13 | Prescan worker queue — bulk intake prescan | API + `workers/pool.py` | ✅ |
| G19 | Partial `_processed.mp4` HTTP serving (range requests) | API | ✅ |
| — | Wire `POST /utils/prescan` → job lifecycle (`PRESCAN_RUNNING` → `AWAITING_REVIEW`) | API | ✅ |
| — | GPU workers pick up only `READY` jobs | `workers/pool.py` | ✅ |
| — | Retry prescan: `PRESCAN_FAILED` → `PRESCAN_PENDING` | Orchestrator | ✅ |

---

## Surfaces

| Surface | Path |
|---------|------|
| Engine prescan | `src/viana/stages/prescan.py`, `src/viana/cli.py` |
| Worker pool | `src/orchestrator/workers/pool.py` |
| Prescan / media routes | `src/orchestrator/routes/` |
| Tests | `tests/viana/test_prescan.py`, `tests/orchestrator/` |

**Do not edit:** `apps/web/` (Step 4).

**Contract changes:** only if Step 3 discovers a gap — file in `STABILIZATION_BACKLOG.md` lane D; Step 2 patch.

**Stabilization:** append prescan/orchestrator defects to [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) per [`STABILIZATION.md`](STABILIZATION.md).

---

## Exit criteria

- [x] Intake job → prescan runs → `AWAITING_REVIEW` with proposals persisted
- [x] Confirm → `READY` → worker starts on GPU
- [x] Bulk folder intake prescan queues correctly
- [x] Auto-aggregate fires on `COMPLETED`
- [x] Partial MP4 servable for live monitor (Step 4 consumes)
- [x] Tests pass
- [x] `AGENT_PROGRESS.md` Step 3 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | S08: faster opening-frame pick (probe t=2s, 4s scan, one VideoCapture) + 2× tight OSD OCR with 4× fallback; CLI lazy-imports process/aggregate; `hiv000001_inframe` CLI 6.7s → 4.6s, S07 fields unchanged |
| 2026-08-19 | Step 3 complete: prescan queue, G7–G9/G12/G13/G19, `LineSegment.assert_within_frame` fix |
| 2026-08-19 | Split from Step 2; engine + orchestrator workers |
