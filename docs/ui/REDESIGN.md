# ViAna UI redesign spec (Step 1 — v0.1)

**Status:** ✅ Approved — discovery signed off 2026-08-19  
**Source:** [`DISCOVERY.md`](DISCOVERY.md)  
**Implements in:** Step 4 (`apps/web/`) after Steps 2–3

---

## 1. Goals

1. **Backend-owned job queue** with prescan as a first-class lifecycle (not UI localStorage drafts).
2. **ViAna_Moving:** prescan proposes time, date, location, horizon line, counting line; operator **confirms or edits** each via review summary before GPU execution.
3. **50+ video batches:** bulk intake, table queue, per-row review, FIFO execution.
4. **Operator-friendly monitoring:** structured telemetry (not raw JSON), live partial video, ETA.
5. **Extensibility** for `ViAnaNP` and `ViAnaJunction` (task picker visible; disabled in v0.1).

---

## 2. Layout overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Project bar: project_id · output_dir (browse) · task type · container     │
├──────────────────────────────────────────────────────────────────────────┤
│ Intake: [Browse file] [Browse folder] [Add selected]                      │
├──────────────────────────────────────────────────────────────────────────┤
│ Job queue table (FIFO)                                                    │
│  stem | status | metadata | progress/ETA | actions                       │
├──────────────────────────────────────────────────────────────────────────┤
│ (optional) Live monitor sidebar when Monitor clicked                      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Project bar

| Control | Behavior |
|---------|----------|
| `project_id` | Slug `[a-z0-9][a-z0-9_-]*`; persisted in localStorage |
| `output_dir` | Browsable override; default `{parent_dir}/{project_id}` from config; shared by all jobs in session |
| Task type | Dropdown: **ViAna_Moving** enabled; ViAnaNP / ViAnaJunction disabled (“Coming soon”) |
| Container | Health + start (existing `ContainerPanel`) |

---

## 4. Video intake

### Path browser (host)

- Next.js API route lists directories/files on **full host filesystem** (local + mounted external HDD).
- Modes: single file, multi-select files, folder (top-level only).
- **Extensions:** `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.m4v` (case-insensitive).
- On select → `POST /jobs/intake` (Step 2) creates one job per video with `PRESCAN_PENDING`.

### Container path warning

If host path is not visible inside the GPU container, show non-blocking warning:

> *This path may not be readable by the processing container. Mount the drive or add a bind-mount. (See ops docs.)*

Step 6 **6.7** addresses arbitrary host path access.

### Single-file flow

1. Browse → select file → job created → prescan runs.
2. Review modal opens immediately on `AWAITING_REVIEW`.
3. Operator confirms → job → `READY` (execution FIFO).

### Folder / multi-file flow

1. Browse → select folder or multiple files → N jobs created.
2. Orchestrator queues prescan for all (`PRESCAN_RUNNING` per row).
3. Table shows status; operator reviews each row when `AWAITING_REVIEW`.
4. Optional: **Apply lines + location** to other `AWAITING_REVIEW` rows (not time/date).

---

## 5. Job status lifecycle

| Status | Operator label |
|--------|----------------|
| `PRESCAN_PENDING` | Queued (PS) |
| `PRESCAN_RUNNING` | Pre-scanning |
| `PRESCAN_FAILED` | Prescan failed |
| `AWAITING_REVIEW` | Needs review |
| `READY` | Queued (GPU) |
| `PROCESSING` | Processing |
| `PAUSED` | Paused |
| `COMPLETED` | Completed |
| `FAILED` | Failed |
| `CANCELLED` | Cancelled |

**GPU gate:** worker picks up only `READY` jobs (FIFO).

**Re-review:** allowed while `AWAITING_REVIEW` or `READY` (before `PROCESSING`).

**Prescan failed:** `PRESCAN_FAILED` + error summary + **Retry prescan** → `PRESCAN_PENDING`.

Legacy `PENDING` removed.

---

## 6. Prescan review modal

**Layout:** wide side-by-side — **canvas left**, **fields right**.

### Steps (single component)

1. **Propose** — engine returns OCR + lines + preview (auto-skip dark frames).
2. **Edit** — scrubber with live frame preview; editable time (HH:MM:SS), date (DD-MM-YYYY), location; drag horizon + counting lines.
3. **Summary** — list all five values; operator confirms.
4. **Submit** — `PATCH /jobs/{id}/prescan` → `READY`.

### Validation

- All metadata fields mandatory; block confirm if empty or invalid format.
- Low OCR confidence: block until edited.
- Geometry: existing `validateCalibration()` rules.

### Optional actions

- Save as project profile (per camera/site location).
- Apply lines + location to other `AWAITING_REVIEW` jobs.

---

## 7. Job queue table

| Column | When shown |
|--------|------------|
| Video (stem) | Always |
| Status (operator label) | Always |
| Time · Date · Location | After confirmed; — before |
| Progress · ETA | `PROCESSING` |
| Crossing count | `PROCESSING` (from WS `MOVING_EVENT` count) |
| GPU | `PROCESSING` / `READY` |
| Actions | Always — three fixed slots; `COMPLETED` shows Open output only |
| Pagination | Footer: rows per page 10 / 25 / 50 / All (default 10); page nav; sticky header; body scrolls after 10 visible rows (`scrollbar-gutter: stable`) |

### Row actions

Non-completed rows always render **Review / Pause / Resume**, **Restart**, **Cancel** in that order (slot 1 label swaps by status). Disabled (muted) when N/A. **Retry prescan** uses slot 2 on `PRESCAN_FAILED` (no review dialog). **Monitor** is not a queue action (I001).

| Slot | Action | Enabled when |
|------|--------|----------------|
| 1 | **Review** / **Pause** / **Resume** | Review: `AWAITING_REVIEW`, `READY`; Pause: `PROCESSING`; Resume: `PAUSED` + checkpoint |
| 2 | **Restart** | Retry prescan: `PRESCAN_FAILED`; Restart (Overwrite): `PAUSED`, `FAILED` |
| 3 | **Cancel** | Not `COMPLETED` / `CANCELLED` |

**Completed row:** only **Open output** (no Review / Restart / Stop).

### Completed row

- Download links for artifacts.
- If `_15min.csv` empty: info banner — *"Start time was not set — 15-minute report is unavailable."*
- Aggregate runs automatically on completion (no manual rebuild button in v0.1).

---

## 8. Live monitor sidebar

Opened from **Monitor** on a `PROCESSING` row.

```
┌─────────────────────────┐
│ Partial _processed.mp4  │  ← HTTP range / poll growing file
├─────────────────────────┤
│ 42% · 22 fps · ~12m left│
├─────────────────────────┤
│ Crossing feed (table)   │  ← virtualized; if telemetry_detail
├─────────────────────────┤
│ Activity log            │
├─────────────────────────┤
│ ▸ Raw JSON (collapsed)  │
└─────────────────────────┘
```

**ETA:** `(total_frames - current_frame) / processing_fps`.

**Telemetry mapping:** see `DISCOVERY.md` §9.

Detail telemetry shown for **focused job only** (default).

---

## 9. Task-type extensibility

| Task | v0.1 | Prescan difference |
|------|------|-------------------|
| ViAna_Moving | ✅ | OCR + 2 lines |
| ViAnaNP | Disabled | OCR + parked zone hints; no lines |
| ViAnaJunction | Disabled | OCR + polygon + gates |

Prescan API should accept `task_type` when NP/Junction ship (Step 2 G2).

---

## 10. State & data ownership

| Data | Owner |
|------|-------|
| Job list + status | Backend `GET /jobs` |
| Prescan proposals + confirmed calibration | Backend job record (G15) |
| `project_id`, `output_dir`, telemetry_detail pref | localStorage (UI prefs only) |
| ~~`pendingPaths` / `drafts`~~ | **Removed** — replaced by backend job states |

---

## 11. Step ownership

| Work | Step |
|------|------|
| Contracts, intake/confirm APIs, `JobStatus`, job storage shape | **2** |
| Prescan engine, worker queue, auto-aggregate, partial MP4 | **3** |
| UI: table, browser, modal, sidebar, telemetry formatters | **4** |
| E2E 15-min verification | **5** |
| Container arbitrary paths | **6.7** |

See [`STEP_2_CONTRACTS_AND_API.md`](../steps/STEP_2_CONTRACTS_AND_API.md) and [`STEP_3_ENGINE_AND_ORCHESTRATOR.md`](../steps/STEP_3_ENGINE_AND_ORCHESTRATOR.md).

---

## 12. Copy reference

| Key | Text |
|-----|------|
| empty_15min | Start time was not set — 15-minute report is unavailable. |
| prescan_failed | Prescan failed |
| container_path_warn | This path may not be readable by the processing container. Mount the drive or add a bind-mount. |
