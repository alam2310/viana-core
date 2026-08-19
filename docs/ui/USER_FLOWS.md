# User Flows (ViAna — post-Step 1 redesign)

> **v0.1:** `ViAna_Moving` only. See [`REDESIGN.md`](REDESIGN.md).

## Flow 1 — First launch

1. User opens UI
2. `GET /api/container/status` → if down, offer **Start container**
3. `GET http://localhost:8000/health` → confirm API ready

## Flow 2 — Project setup

1. User sets `project_id` (slug, e.g. `nh48`)
2. User sets **output directory** (browse; default `{parent}/{project_id}`)
3. Task type: **ViAna_Moving** (NP/Junction visible but disabled)

## Flow 3 — Single video intake

1. User clicks **Browse file** → host filesystem picker → selects one video
2. Backend creates job → prescan runs (`PRESCAN_RUNNING`)
3. Review modal opens (`AWAITING_REVIEW`)
4. User edits/confirms time, date, location, lines → summary step → confirm
5. Job → `READY` → GPU picks up when slot free → `PROCESSING` → `COMPLETED`
6. Aggregate runs automatically; artifacts linked on row

## Flow 4 — Batch intake (folder / multi-select)

1. User browses **folder** (top-level videos) or **multi-selects** files
2. Backend creates N jobs; orchestrator queues prescan for all
3. Queue **table** shows per-row prescan status
4. User reviews each `AWAITING_REVIEW` row (same modal); optional **apply lines + location** to siblings
5. Each confirmed job → `READY`; FIFO execution as GPUs free

## Flow 5 — Re-review from queue

1. User clicks **Review** on `AWAITING_REVIEW` or `READY` row
2. Same prescan review modal; edits saved → remains `READY`
3. Not allowed once `PROCESSING`

## Flow 6 — Prescan failure

1. Row shows **Prescan failed** + error summary
2. User clicks **Retry prescan** → `PRESCAN_PENDING` → retry cycle

## Flow 7 — Monitor processing

1. User clicks **Monitor** on `PROCESSING` row
2. Sidebar: partial processed video + progress/ETA + crossing feed + activity log
3. Detail telemetry for focused job only

## Flow 8 — Paused / failed job

1. `GET /jobs/{id}` → `PAUSED` or `FAILED`, `checkpoint_exists` when applicable
2. User chooses **Resume** or **Start fresh**

## Flow 9 — Completed job

1. `status: COMPLETED`
2. Row shows metadata + artifact download links
3. Info note if 15-min report unavailable (operator-friendly copy)
