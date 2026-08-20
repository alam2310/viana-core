# UX discovery (Step 1 — living document)

**Status:** ✅ Complete — signed off 2026-08-19  
**Output:** [`REDESIGN.md`](REDESIGN.md)

> The UX agent maintains this file during Step 1. Record answers, decisions, and open questions here before finalizing the redesign spec.

---

## 1. Product context (agent fills from docs)

| Topic | Summary |
|-------|---------|
| **Platform** | Next.js UI on host orchestrates a GPU Docker container (FastAPI + `viana` CV engine). Local video paths only — no upload. |
| **v0.1 scope** | `ViAna_Moving` only — directional moving vehicle count via horizon + counting lines. |
| **Future tasks** | `ViAnaNP` (parked extraction, metadata-only calibration); `ViAnaJunction` (polygon + O/D gates). Design for extensibility; not shipped in v0.1. |
| **Prescan today** | Engine samples frame (configurable offset), runs OSD OCR (time/date/location), proposes horizon + counting lines (from profiles or heuristics), writes preview JPEG. API: `POST /utils/prescan`. |
| **Current UI** | Single modal: run prescan → canvas + editable OCR fields + frame scrubber → "Save calibration" (blocks on geometry validation). "Apply to pending" + optional profile save. |
| **User must confirm** | Step 1 goal: explicit **propose → confirm/edit** for time, date, location, horizon line, counting line before job submit. |
| **15-min report** | Wall-clock metadata from prescan → `time_map.json` → aggregate → `{stem}_15min.csv`. Empty CSV verification is **Step 5**. |
| **Hardware** | Dual RTX 3060; backend assigns `gpu_device`. Batch sizes up to 50+ videos per project (spec). |

---

## 2. Stakeholder Q&A

_Agent: ask the user questions in chat. Paste questions and answers below._

### Session log

| # | Date | Question | Answer / decision |
|---|------|----------|-------------------|
| 1 | 2026-08-19 | Before submit, which metadata fields are mandatory? | **All three:** time, date, and location |
| 2 | 2026-08-19 | Low OCR confidence / empty field UX? | **Block submit** until user fills or edits the field |
| 3 | 2026-08-19 | Date/time input format? | **24-hour time + DD-MM-YYYY date** (Indian traffic convention) |
| 4 | 2026-08-19 | Trust auto-proposed lines? | **Always review** — proposals are a starting point only |
| 5 | 2026-08-19 | "Apply to all pending" scope? | **Lines (horizon + counting) + location** — not time/date |
| 6 | 2026-08-19 | Prescan layout preference? | **Side-by-side:** canvas left, fields right (wide modal) |
| 7 | 2026-08-19 | Primary operators? | **Solo operator** (small team) |
| 8 | 2026-08-19 | Typical batch size? | **30–50+ videos** per session |
| 9 | 2026-08-19 | Confirm UX for proposals? | **Final review summary step** listing all five values before submit |
| 10 | 2026-08-19 | Running job card must show? | **Rich:** filename, status, progress, GPU, metadata, line source, live crossing count + ETA |
| 11 | 2026-08-19 | Completed job card must show? | **Metadata + artifact download links** (events, 15min, video) |
| 12 | 2026-08-19 | Empty `_15min.csv` UX? | **Informational note** with **operator-friendly** copy; downloads still available |
| 13 | 2026-08-19 | Frame scrubber default? | **Auto-skip dark frames** (engine picks best) + user scrub with **live frame preview** on slider |
| 14 | 2026-08-19 | Profile workflow for large batches? | Save **per camera/site (location)**; **not** auto-apply on resolution alone; optional future: auto-apply if **background/road matches** (only if feasible — do not overcomplicate v0.1) |
| 15 | 2026-08-19 | Task picker in v0.1? | **Visible** with NP/Junction **disabled** (“coming soon”) |
| 16 | 2026-08-19 | Queue ordering? | **FIFO** — submit order preserved |
| 17 | 2026-08-19 | ViAnaNP prescan (future)? | Time, date, location + **parked zone hints (draft)**; no lines |
| 18 | 2026-08-19 | ViAnaJunction prescan (future)? | OCR metadata + **auto-proposed junction polygon + gates** |
| 19 | 2026-08-19 | Detail telemetry default? | **On for active/focused job only** |
| 20 | 2026-08-19 | Aggregate trigger UX? | **Auto-run on job completion** — no manual “Rebuild 15-min” button in v0.1 |
| 21 | 2026-08-19 | 50+ video batch workflow? | **Bulk prescan all pending**, then review queue |
| 22 | 2026-08-19 | Discovery sign-off? | **Signed off** 2026-08-19 |
| 23 | 2026-08-19 | Queue UX after bulk prescan? | **Table** of jobs with prescan status; row action opens **same review modal** one-by-one. **Re-review allowed** until job starts executing (not `PROCESSING`). |
| 24 | 2026-08-19 | Video intake? | **Browse filesystem:** single file **or** folder (multi-select files OK). Path must be navigable — not free-text only. |
| 25 | 2026-08-19 | Single-file flow? | Add path → prescan runs → **immediate review** → on confirm job enters execution queue with **prescan-reviewed** status. |
| 26 | 2026-08-19 | Folder / multi-file flow? | All files queued for prescan; table shows prescan progress; status **not confirmed** until each reviewed. Same camera/location may share lines+location (not time/date). |
| 27 | 2026-08-19 | Execution gate? | GPU worker **must not pick up** job until prescan **reviewed/confirmed**. Review workflow reusable from queue at any time pre-`PROCESSING`. |
| 28 | 2026-08-19 | Job status model? | **Extend `JobStatus`** — prescan phases distinct from execution queue (see §7). Steps 2–3 contract + orchestrator. |
| 29 | 2026-08-19 | Running job monitor? | **Button on queue row** opens **sidebar** (or equivalent): live processed video view + telemetry panel below. |
| 30 | 2026-08-19 | Running job ETA? | Show **remaining estimated time** in addition to % progress (frame-based ETA acceptable). |
| 31 | 2026-08-19 | Empty `_15min.csv` copy? | **Operator-friendly** message (e.g. “Start time was not set — 15-minute report is unavailable”). |
| 32 | 2026-08-19 | Project / output directory? | Multiple videos per **project** share one **navigable output directory** (default preset); same camera/location → same project + output dir. |
| 33 | 2026-08-19 | Folder ingest? | **Top-level only** — no subfolder recursion |
| 34 | 2026-08-19 | Project vs output_dir? | **`project_id` + separate browsable `output_dir` override** per project (default from config) |
| 35 | 2026-08-19 | Live monitor video? | **Poll/serve partial `_processed.mp4`** as it grows during `PROCESSING` |
| 36 | 2026-08-19 | Status enum names? | **Approved draft** with `PRESCAN_RUNNING` (not `PRESCANNING`); operator label **"Pre-scanning video"** |
| 37 | 2026-08-19 | Video extensions for folder scan? | **Standard extensions:** `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.m4v` (case-insensitive) |
| 38 | 2026-08-19 | Path browser scope? | **Full filesystem** — local disks and mounted external drives; no artificial allowlist (operator machine) |
| 39 | 2026-08-19 | Background-match profile auto-apply? | **Deferred** post-v0.1 (Step 6 / later) |
| 40 | 2026-08-19 | Telemetry display today? | Raw JSON in `<pre>` — **not acceptable for operators**; structured UI required (§9) |
| 41 | 2026-08-19 | Prescan failure status? | Dedicated **`PRESCAN_FAILED`** with operator label **"Prescan failed"** (not generic `FAILED`) |
| 42 | 2026-08-19 | Container read any host path? | **Backlog:** backend/DevOps must allow container to read videos on local disk + mounted external drives (§5 G21) |

### Open questions

_All discovery blockers resolved. Remaining items deferred to REDESIGN defaults or Steps 2–6 (see §8)._

---

## 3. Task-type prescan matrix

| Task | Prescan proposes | User confirms / edits | Calibration UI | v0.1 ship? |
|------|------------------|----------------------|----------------|------------|
| **ViAna_Moving** | OCR: time, date, location; horizon + counting lines; preview frame | All five editable; **all metadata mandatory** before submit; block on low/empty OCR; always review lines | Side-by-side 2-line canvas + field panel | ✅ Yes |
| **ViAnaNP** | OCR: time, date, location; **parked zone hints (draft)** | Metadata + zone hints confirmed; no line geometry | Zone overlay canvas (TBD) — no horizon/counting lines | ❌ Future |
| **ViAnaJunction** | OCR metadata; **auto-proposed junction polygon + named gates** | Polygon + gates editable | Polygon + gate canvas | ❌ Future |

**Extensibility note:** Prescan request/response may need `task_type` (Step 2 schema; Step 3 engine).

---

## 4. Screen inventory (big picture)

| Screen | Purpose | Notes |
|--------|---------|-------|
| **Settings / first launch** | Container health, start backend, API health check | Flow 1 in `USER_FLOWS.md` |
| **Project bar** | `project_id`, **browsable output directory** (default from config), task type (Moving enabled; NP/Junction disabled) | Output dir shared by all videos in project |
| **Video intake** | Filesystem browser: **file**, **folder**, or **multi-select** | Host-side picker; **full filesystem**; standard video extensions (§7) |
| **Job queue table** | Unified table: all jobs with **prescan + execution status**, metadata summary, actions | Replaces loose “pending paths” list; FIFO |
| **Prescan review** | Side-by-side modal/sheet: canvas + OCR fields + review summary step | Same component from intake or queue row |
| **Live monitor sidebar** | Opens from queue row for `PROCESSING` job: partial video + **structured telemetry** below (§9) | Not raw JSON; crossing feed + progress + logs |
| **Paused / failed job** | Resume vs start-fresh | Checkpoint-aware row actions |
| **Completed job / artifacts** | Metadata + download links; operator-friendly note if 15-min empty | Auto-aggregate on completion |
| **Container / settings** | Output parent default, telemetry prefs | Host docker via Next API routes |

### Queue table columns (v0.1 draft)

| Column | Notes |
|--------|-------|
| Video path (stem) | Truncated + tooltip |
| Status | See §7 lifecycle |
| Prescan | Spinner / needs review / confirmed |
| Time · Date · Location | From confirmed metadata (— if not reviewed) |
| Progress · ETA | Only when `PROCESSING` |
| Actions | Review · Monitor · Cancel · Artifacts (contextual) |

---

## 7. Proposed job lifecycle (status model)

**Problem:** Today `PENDING` means “waiting for GPU” but prescan happens **outside** the job record (UI localStorage drafts). New UX requires **backend-owned** prescan state so the queue table is the single source of truth.

### Final `JobStatus` values (Steps 2–3)

| Status | Operator label | Meaning | Worker may start? |
|--------|----------------|---------|-----------------|
| `PRESCAN_PENDING` | Waiting for prescan | Job registered; prescan not started | No |
| `PRESCAN_RUNNING` | Pre-scanning video | Engine sampling frame + OCR + lines | No |
| `PRESCAN_FAILED` | Prescan failed | Prescan error (OCR, I/O, corrupt video, path inaccessible) | No |
| `AWAITING_REVIEW` | Needs review | Prescan complete; user must confirm/edit | No |
| `READY` | Ready | Prescan confirmed; in execution FIFO | No (queued) |
| `PROCESSING` | Processing | On GPU | Yes |
| `PAUSED` | Paused | Checkpoint saved | No |
| `COMPLETED` | Completed | Artifacts written | No |
| `FAILED` | Failed | Error | No |
| `CANCELLED` | Cancelled | User removed | No |

**Note:** Legacy `PENDING` is **removed** — replaced by `PRESCAN_*` / `AWAITING_REVIEW` / `READY` phases.

**Transitions (happy path):**

```
intake → PRESCAN_PENDING → PRESCAN_RUNNING → AWAITING_REVIEW → READY → PROCESSING → COMPLETED
                  |                |
                  └→ PRESCAN_FAILED ←┘ (Retry prescan → PRESCAN_PENDING)
                                    ↑_______________|  (re-review / edit until PROCESSING)
```

### Folder scan rules

- **Scope:** top-level of selected folder only (no subfolder recursion).
- **Extensions:** `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.m4v` (case-insensitive).
- Non-video files ignored silently.

### Path browser rules

- Browse **any path** on the host filesystem (internal disk or mounted external volume).
- **Ops constraint (document in REDESIGN):** paths must also be visible inside the GPU container (bind-mount / same mount point). UI warns if container cannot access selected path.

**Rules from stakeholder:**

- Re-open review from queue while status is `AWAITING_REVIEW` or `READY` (not yet `PROCESSING`).
- Prescan errors → `PRESCAN_FAILED` ("Prescan failed"); row shows error summary + **Retry prescan** (returns to `PRESCAN_PENDING`).
- Single file: prescan → review immediately (may skip visible `PRESCAN_PENDING` if sync).
- Folder/batch: all rows show prescan progress; each must reach `READY` independently.
- “Apply lines + location to pending” applies to other `AWAITING_REVIEW` rows (not time/date).

**Alternative considered:** separate `prescan_status` field on `JobStatusResponse` — rejected for v0.1 simplicity; one status column is clearer in the table.

**Migration:** Deprecate UI-only `pendingPaths` / `drafts` localStorage for calibration; persist prescan proposals server-side (see G14–G16).

---

## 5. Backend / prescan gaps (Steps 2–3)

_Work items split: **Step 2** = contracts + intake/confirm APIs; **Step 3** = engine + workers. See `docs/steps/STEP_2_CONTRACTS_AND_API.md` and `STEP_3_ENGINE_AND_ORCHESTRATOR.md`._

| ID | Gap | Step |
|----|-----|------|
| G1 | `proposed_*` vs confirmed on job record | **2** |
| G2 | `task_type` on prescan request | **2** (schema), **3** (engine) |
| G4 | Server metadata validation | **2** |
| G7 | Auto-skip dark frames | **3** |
| G8 | Live frame preview on scrub | **3** |
| G9 | ETA + crossing count | **3** |
| G12 | Auto-aggregate on COMPLETED | **3** |
| G13 | Bulk prescan worker queue | **3** |
| G14 | `JobStatus` enum extension | **2** |
| G15 | Persist proposal + confirmed on job | **2** |
| G16 | `POST /jobs/intake` | **2** |
| G17 | `PATCH /jobs/{id}/prescan` | **2** |
| G18 | Host filesystem browser | **4** (UI) |
| G19 | Partial `_processed.mp4` serving | **3** |
| G20 | `output_dir` override | **2** |
| G21 | Container arbitrary host paths | **6.7** |
| G22 | Telemetry presentation | **4** (UI) |

---

## 9. Telemetry presentation plan

**Today:** `TelemetryPanel` renders `JSON.stringify(messages)` — operator-unfriendly.

**Target (Step 4 UI — live monitor sidebar + optional dashboard strip):**

| `telemetry_type` | Operator-facing UI | Data mapped from `data` |
|------------------|-------------------|------------------------|
| `PROGRESS` | Progress bar, **%**, **processing fps**, **remaining time** (ETA) | `current_frame`, `total_frames`, `processing_fps` |
| `MOVING_EVENT` | **Crossing feed** — scrollable table (virtualized) | `class_name`, `direction`, `track_id`, `frame_index` |
| `LOG` | **Activity log** — timestamped plain-language lines | `message` → mapped labels (see below) |

### Crossing feed columns (v0.1)

| Column | Source |
|--------|--------|
| Time | Derived from frame + job metadata / time_map when available; else frame # |
| Vehicle | `class_name` (human-readable) |
| Direction | `direction` (capitalize) |
| Track | `track_id` (muted, for support) |

### LOG message mapping (UI-side, v0.1)

| Engine `data.message` | Operator text |
|-----------------------|---------------|
| `process_start` | Processing started |
| `process_complete` | Processing finished |
| `interrupted` | Processing paused |
| _(other)_ | Show message text as-is (errors verbatim) |

### Layout in live monitor sidebar

```
┌─────────────────────────────────────┐
│  Partial processed video (MP4)      │
├─────────────────────────────────────┤
│  Progress: 42% · 22 fps · ~12m left │
├─────────────────────────────────────┤
│  Latest crossings (table, auto-scroll) │
├─────────────────────────────────────┤
│  Activity log (last N lines)        │
├─────────────────────────────────────┤
│  ▸ Show raw JSON (collapsed, dev)   │
└─────────────────────────────────────┘
```

- **Detail telemetry:** crossing feed + log only when `telemetry_detail` was enabled for that job (or always show progress strip).
- **Focused job:** dashboard polls/WS for active job only (per Q#19).
- **Performance:** virtualized list for crossing feed; cap retained messages client-side (e.g. last 500 events, last 50 log lines).

**Step ownership:** UI formatting in **Step 4** (`features/telemetry/`).

---

## 8. Deferred defaults (REDESIGN — no discovery blocker)

| Topic | Default unless overridden in sign-off |
|-------|--------------------------------------|
| Prescan failure | Row → **`PRESCAN_FAILED`** ("Prescan failed"); **Retry prescan** → `PRESCAN_PENDING` |
| Cancel | Available in all statuses before `COMPLETED` |
| Concurrent prescan | Orchestrator limits parallel prescan (e.g. 2–4); table shows queue position |
| Container path access | UI validates path exists on host; warn if not in container mount set |
| G12 auto-aggregate | Step 3 implements; orchestrator hooks aggregate on `COMPLETED` |
| G1 proposed vs confirmed | Step 2 — persist both on job record |
| Empty 15-min copy | _"Start time was not set — 15-minute report is unavailable."_ |
| ETA formula | `(total_frames - current_frame) / processing_fps` when fps > 0 (`processing_fps` = wall-clock throughput, not source fps). Video length is `video_duration_sec` from prescan, not ETA. |

---

## 6. Discovery sign-off

- [x] User reviewed task-type matrix for `ViAna_Moving`
- [x] Open questions resolved or explicitly deferred (§8)
- [x] Screen inventory complete
- [x] User confirms ready to write `REDESIGN.md` (Phase 1.2)

**Signed off:** 2026-08-19 — stakeholder confirmed in chat
