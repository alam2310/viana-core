# Stabilization backlog (living)

**Rules:** [`STABILIZATION.md`](STABILIZATION.md)  
**Step 5 blocked while any blocker row is `open` or `in_progress`.**

> **Follow [`Execution path`](#execution-path) in Seq order.** One row = one unit of work. Do not skip ahead unless a dependency is `fixed` / `deferred`.

**Last updated:** 2026-08-21

---

## Summary

| Blockers open | Blockers fixed | Polish open | Path steps done |
|---------------|----------------|-------------|-----------------|
| 0 | 1 | 4 | 26 / 30 active |

**S09 (F006) closed in Step 6.7.** **Not counted** in path progress.

---

## Execution path

Work **top to bottom**. **Depends** = prior Seq that must be `fixed` or `deferred` before starting.

| Seq | ID | Lane | Blocker | Depends | Title | Status |
|-----|-----|------|---------|---------|-------|--------|
| **S01** | F004 | B | no | — | Verify preview JPEG survives orchestrator restart | fixed |
| **S02** | F003 | B | no | — | Add `GET /artifacts/{job_id}/source.mp4` (Range, prescan-phase jobs) | fixed |
| **S03** | F003 | A | no | S02 | Next.js `/api/proxy/source` — same-origin video stream | fixed |
| **S04** | F003 | A | no | S03 | Prescan scrub: video seek → canvas; **no** `prescan/preview` on slider | fixed |
| **S05** | F003 | A/B | no | S04 | `prescan/preview` only for **Re-scan OCR**; docs sync (`api_contracts`, `COMPONENT_MAP`) | fixed |
| **S06** | F005 | C | no | — | Triage EasyOCR in container (installed? hits on test frame?) | fixed |
| **S07** | F001 | C | **yes** | S06 | Corner ROI OSD OCR → populated `proposed_metadata` | fixed |
| **S08** | F002 | C | no | — | Reduce prescan wall-clock (OCR works; intake still 30s+) | fixed |
| **S10** | F007 | C | no | S07 | Improve horizon + counting line proposal (CV / geometry) | fixed |
| **S11** | F008 | B/D | no | — | `JobStatus.created_at` + sortable submitted time in API | fixed |
| **S12** | F009 | B/D | no | — | `JobStatus.video_duration_sec` + `processing_duration_sec` from API (not UI localStorage) | fixed |
| **S13** | F010 | B/C | no | — | Growing `_processed.mp4` streamable during job (moov/fragmented MP4) | fixed |
| **S14** | F011 | C | no | — | Emit `MOVING_EVENT` without `telemetry_detail` gate; include crossing timestamp | fixed |
| **S15** | F012 | B/D | no | — | 15-min CSV schema: add `date` column; change `window_start`/`window_end` to HH:MM | fixed |
| **S16** | F013 | A | no | — | Theme toggle regression: action buttons keep dark styling after returning to light mode | fixed |
| **S17** | F014 | A | no | — | Remove duplicate `Recent crossing` table from job details widget (superseded: I001 put Live Crossings in details) | fixed |
| **S18** | F015 | A/C | no | S14 | Normalize live monitor crossing title + `HH:MM:SS` actual-time formatter | fixed |
| **S19** | F016 | A/B/C | no | S12 | Fix queue video length / ETA inflation and validate MP4 codec metadata in container | fixed |
| **S20** | F010 | A/B | no | S13 | UI cannot render in-progress processed MP4 even when file is playable natively | **parked** (see S24) |
| **S21** | F017 | C | no | — | Prescan OCR misses time/location when OSD text appears in alternate screen regions | **fixed** |
| **S22** | F018 | B/C | no | — | Intake/prescan triggers `[Errno 24] Too many open files`, followed by API 502 in UI refresh | **fixed** |
| **S23** | F019 | C | no | — | Processing throughput regression: end-to-end run is much slower than earlier baseline | **fixed** |
| **S24** | F010 | A | no | S20 | Park in-progress `_processed.mp4` preview; crossings immediate (no delay). Player stays unmounted after I001. | parked |
| **S25** | F020 | A | no | — | Review job status UI labels vs lifecycle (Queued vs Ready inconsistency) | **fixed** |
| **S26** | F021 | A | no | — | Standardize Job Queue action icons (stable slots; enable/disable by status) | **fixed** |
| **S27** | F022 | B | no | — | After a job FAILED, next READY job did not start despite free GPU | **fixed** |
| **S28** | F023 | C | no | — | Missed counting-line events when class flicker drops the box for 1–2 frames | fixed |
| **S29** | F024 | B/C | no | — | Excess leftover files in output dir after successful COMPLETED (strategy + layout) | open |
| **S30** | F025 | A/B | no | — | UI API 502 (`fetch failed`) when restarting/resuming a job from the queue | open |
| **S31** | F026 | A | no | — | Prescan review: remove duplicate Close; rename Submit → Confirm | open |
| **S32** | F027 | C/D | no | — | Relook raw events + 15-min CSV schemas — keep only necessary columns | open |
| ~~**S09**~~ | F006 | B | no | — | API rejects container-unreadable intake paths | **fixed → Step 6.7** |

**After S07 is `fixed` or `deferred` (approved):** Step 5 may start. S08 and S10 are polish (may continue in parallel or after Step 5).

---

## Row detail

| Seq | Repro | Expected vs actual | Files / notes | Fix commit | Verified |
|-----|-------|-------------------|---------------|------------|----------|
| **S01** | 1. Prescan → `AWAITING_REVIEW` 2. Restart `viana_core` 3. Open review | **Expected:** `proposed_preview_url` JPEG still loads. **Actual:** in-memory registry empty → 404. | `src/orchestrator/preview_registry.py` — disk `rglob` fallback | `45a82a4` | orchestrator test (S01) |
| **S02** | 1. Job in `AWAITING_REVIEW` 2. `GET /artifacts/{id}/source.mp4` with Range | **Expected:** streams `job.source_video_path` for browser seek (mirror G19 partial MP4). **Actual:** no source endpoint; scrub must spawn prescan. | `src/orchestrator/routes/artifacts.py`, `api_contracts.md`, orchestrator test | `9b378c1` | orchestrator test (S02) |
| **S03** | 1. S02 deployed 2. Open review modal | **Expected:** browser loads source via same-origin proxy (like `/api/proxy/preview`). **Actual:** cross-origin or no URL. | `apps/web/src/app/api/proxy/source/route.ts`, `api-client.ts` | uncommitted | manual UI scrub |
| **S04** | 1. Open prescan review 2. Move frame-offset slider | **Expected:** frame updates in &lt;200ms from local video seek; lines unchanged. **Actual:** each scrub calls `GET /jobs/{id}/prescan/preview` → full `viana prescan` subprocess (OCR + lines + JPEG). | `prescan-review-modal.tsx`, `calibration-canvas.tsx` — hidden `<video>`, `seeked` → `drawImage`; remove offset→`prescanPreview` effect; `loadedmetadata` → `duration_sec` | uncommitted | manual UI scrub |
| **S05** | 1. Click **Re-scan OCR at Ns** | **Expected:** only explicit re-scan hits prescan API; slider does not. **Actual:** (after S04) re-scan still calls `prescanPreview`; metadata-only merge. Also: `applyToOthers` should use `job.proposed_*` / status, not `prescanPreview(0)` for resolution. | `prescan-review-modal.tsx`, `docs/api_contracts.md` § artifacts, `docs/ui/COMPONENT_MAP.md` | uncommitted | manual re-scan + apply-to-others |
| **S06** | 1. `docker exec` into API container 2. Run prescan on `hiv000001_inframe.mp4` 3. Inspect OCR stdout | **Expected:** EasyOCR installed; corner OSD yields hits. **Actual:** EasyOCR 1.7.2 installed; `optional_easyocr_reader()` returns `CornerOsdReader` (not no-op). Frame 0 has blank top band (no OSD); full-frame OCR 0 hits. Corner ROI at t≈3s yields date/time fragments; OSD fades in by t=2s. `paragraph=True` returns `[bbox,text]` without confidence — fixed in S07. | `ocr.py`, `prescan.py` — informs corner ROI + first-OSD frame pick | engine S06–S07 | engine S07 |
| **S07** | 1. Intake `data/raw/hiv000001_inframe.mp4` 2. `AWAITING_REVIEW` 3. Open review | **Expected:** `proposed_metadata` has time (HH:MM:SS), date (DD-MM-YYYY), location from 1–2 corner ROIs. **Actual (before):** fields empty. **After:** `02:21:25`, `18-10-2024`, `LITO-RARARANKI` on intake job `job_abec59713960`. | `src/viana/stages/ocr.py`, `prescan.py`, `time_map.py` | engine S06–S07 | engine S07 intake |
| **S08** | 1. Intake `hiv000001_inframe.mp4` 2. Time until `AWAITING_REVIEW` | **Expected:** prescan in a few seconds. **Actual (before):** 30s+ reported; engine CLI `viana prescan` **6.676s** (OCR 5.55s @ 4× wide ROI; opening scan + second VideoCapture). **After (2026-08-19):** CLI **4.60s** (sample_opening_frame 0.069s at t=2.0s; OCR parse 3.76s on 2× tight ROI). S07 fields unchanged: `02:21:25`, `18-10-2024`, `LITO-RARARANKI`. Scan window 30s→4s; probe t=2s first; CLI no longer imports process/YOLO on prescan. | `prescan.py`, `ocr.py`, `cli.py`, `configs/engine_defaults.yaml` | `153431f` | engine CLI before/after on `hiv000001_inframe.mp4` |
| **S10** | 1. Intake `hiv000001_inframe.mp4` (or parity clip) 2. `AWAITING_REVIEW` 3. Open review modal | **Expected:** `proposed_lines` match road geometry (horizon near vanishing point, counting line on lane boundary) — usable without large edits. **Actual (before):** global Hough median followed rooftops/gantries and inverted the `hiv000001` slope. **After (2026-08-21):** road-band slope clustering + parallel counting offset (`0.26 H`). Profile still overrides. Clip t=2s no-profile: **H `[0,579]→[1919,340]` C `[0,874]→[1919,635]`** (conf 0.725). Endpoint \|dy\| vs geometry **D** 213 / **C** 218 (was 887 / 984). Also checked `parity_golden`, `hiv00053_EDIT`, `hiv00013_shimoga`, `hiv00037_night`, `test_video`. | `src/viana/stages/lines.py`, `src/viana/stages/prescan.py`, `tests/viana/test_prescan.py`; targets in `tests/viana/fixtures/PARITY_NOTES.md` | uncommitted | `pytest tests/viana/test_prescan.py` (29 passed) + clip probe |
| **S11** | 1. Intake job 2. `GET /jobs` | **Expected:** each job has `created_at` (ISO) for queue sort. **Actual:** UI uses localStorage fallback until API field exists. | `job_status.schema.json`, `pool.py` `JobRecord`, `to_status()` | `81106c0` | orchestrator tests (S11) |
| **S12** | 1. Prescan completes 2. `GET /jobs/{id}` | **Expected:** `video_duration_sec` and `processing_duration_sec` on status for queue columns. **Actual:** UI estimates video length from progress fps when available; run time uses localStorage timestamps captured while dashboard is open during `PROCESSING` — shows `—` for jobs completed before first poll or after page refresh. | prescan worker persists meta on `JobRecord`; schema + TS types; `job-local-meta.ts` | `81106c0` | orchestrator tests (S12) |
| **S13** | 1. `PROCESSING` job 2. Open live monitor 3. Try partial MP4 in VLC | **Expected:** `_processed.mp4` playable while growing. **Actual (before):** file not playable until job completes (moov atom at end). **After:** FFmpeg writer always emits fragmented MP4 with frequent fragments/keyframes (`frag_duration=1s`) so in-progress output is streamable. | `src/viana/stages/render.py`, `tests/viana/test_render.py`, `tests/orchestrator/test_job_routes.py` | uncommitted | `pytest tests/viana/test_render.py tests/orchestrator/test_job_routes.py -q` |
| **S14** | 1. Confirm prescan with `telemetry_detail: true/false` 2. Monitor during `PROCESSING` | **Expected:** `MOVING_EVENT` on WS for every crossing with wall-clock or video timestamp. **Actual (before):** events gated on `telemetry_detail`; no timestamp in payload (frame_index only). **After:** emission is unconditional and payload carries `event_timestamp`, `event_timestamp_source`, `event_timestamp_confidence`, `video_pts_ms` (plus existing frame/fps fields). | `src/viana/stages/process.py`, `tests/viana/test_process.py`, `tests/orchestrator/test_job_routes.py` | uncommitted | `pytest tests/viana/test_process.py tests/orchestrator/test_job_routes.py -q` |
| **S15** | 1. Generate `_15min.csv` 2. Inspect header/rows | **Expected:** `date` plus `window_start`/`window_end` as `HH:MM` with strict schema header order. **Actual (before):** UI parser grouped by window/class only and could merge multi-day rows. **After:** contract/schema + aggregate writer emit `date` and `HH:MM`; parser requires `date` column and keys by `date+window+class` while remaining backward-compatible with ISO window values. | `packages/contracts/schemas/events_15min.schema.json`, `src/viana/io/csv_schema.py`, `src/viana/stages/aggregate.py`, `apps/web/src/lib/parse-15min-csv.ts`, `tests/viana/test_cli_aggregate.py` | uncommitted | `npm --prefix apps/web run typecheck`; python `pytest` unavailable in this environment |
| **S16** | 1. Switch to dark theme 2. Return to light theme 3. Visit intake, prescan dialogs, and live monitor | **Expected:** button colors and variants re-compute on every theme change. **Actual (before):** several buttons remained in dark-style colors after switching back to light. **After:** removed one-way dark-only button class overrides and standardized affected controls on shared `Button` variants so dark↔light toggles resolve symmetrically. | `apps/web/src/features/project/project-bar.tsx`, `apps/web/src/features/intake/path-browser.tsx`, `apps/web/src/features/prescan/prescan-review-modal.tsx`, `apps/web/src/features/monitor/monitor-sidebar.tsx` | uncommitted | UI smoke checks: intake dialog, prescan modal, monitor dialog |
| **S17** | 1. Open job details widget 2. Open live monitor for same job | **Expected (original):** one canonical crossings surface. **After S17:** `Recent crossings` removed from details; live monitor was canonical. **Superseded 2026-08-21 (I001 / 6.8):** Live Monitor widget removed; **Live Crossings** live in job details (S18 `HH:MM:SS` formatter kept). | `apps/web/src/features/telemetry/job-details-panel.tsx`, `live-crossings.tsx` | uncommitted | UI: details has Live Crossings; no monitor widget |
| **S18** | 1. Open live monitor while crossings stream 2. Inspect section header + crossing time column | **Expected:** title follows title case (`Live Crossings`) and displayed time is actual identified event time formatted strictly as `HH:MM:SS`. **Actual (before):** title/time formatting was inconsistent. **After:** normalized title usage to `Live Crossings`, constrained crossing column label to `Time (HH:MM:SS)`, and tightened timestamp formatter to return `HH:MM:SS` from canonical event timestamps (with bounded fallback behavior). | `apps/web/src/features/monitor/monitor-sidebar.tsx`, `apps/web/src/features/telemetry/telemetry-panel.tsx`, `apps/web/src/features/telemetry/crossings-table.tsx`, `apps/web/src/features/telemetry/telemetry-formatters.ts` | uncommitted | UI smoke checks: monitor dialog + job details widget |
| **S19** | 1. Queue a known long clip (~3h) 2. Observe Job Queue `Video length` and `Time remaining` | **Expected:** video length and ETA are in the right order of magnitude (3h clip should not show 20h+ remaining without evidence). **Actual (before):** Hikvision `.mp4` is MPEG-PS; OpenCV/ffprobe **header** duration on `hiv00013_shimoga.mp4` was **76240s (21.2h)** / **1,143,606 frames** @ 15 fps (implied **28 kbps**). True demux: **34,197 packets → 2279.8s (00:37:59.8)**. Queue showed ~21h length and 20h+ ETA because `eta_sec = remaining_frames / processing_fps` used the inflated count. **After:** `apply_container_timing` recounts packets when bitrate &lt; 80 kbps; video length **00:37:59.8**; ETA uses ~34k frames. Formula: see F016 note. | `src/viana/io/media.py`, `prescan.py`, `video.py`, `process.py`, queue formatters, `api_contracts.md` | uncommitted | container probe + `pytest tests/viana/test_media.py tests/viana/test_process.py` |
| **S20** | Growing `_processed.mp4` in live monitor browser player | **Expected:** stable live preview synced with crossings. **Actual:** H.264 encode fixed decode, but live-edge/seek UX remained unstable (blackouts, Range/FD storms). **Parked 2026-08-20 → S24** (UI unmounted; code retained). | `live-processed-video.tsx`, `crossing-media-sync.ts` (unused), artifact/proxy H.264 path | `2e56532` + later UI; **parked** | Do not remount player without S24 revisit |
| **S21** | 1. Run prescan on a video where OSD layout differs (location at top-center, timestamp at bottom-left) 2. Inspect `proposed_metadata` | **Expected:** prescan extracts time/date/location despite changed on-screen text positions. **Actual (before):** corner ROIs missed fields. **After (2026-08-20):** 2× corners still first (S08); if time/date miss, 4× wide corners (S07); if any required field still empty, full-width top/bottom bands at 2×. Layout-variant MJPG clip recovered `07:21:26`, `18-10-2024`, `LITO-TOPCENTER`. `hiv000001_inframe.mp4` unchanged: `02:21:25`, `18-10-2024`, `LITO-RARARANKI` in **4.00s** `run_prescan` (S08 CLI was 4.60s). **Follow-up:** `hiv00013_shimoga.mp4` opening frame (t=2s) OCR'd `06 :44:35` (space before colon); time was appended to location. Parser now accepts spaced separators and peels the clock out of location → `06:44:35` / `28-07-2026` / `Bangalorebypassjz`. **hiv00037 night:** mixed white/black location glyphs; clock OCR'd as `05:34"04` and wide ROI year `7074` overwrote `2024`. Quote-as-colon + year repair + stroke (gradient) pass on mixed-polarity location crops → `05:34:04` / `19-10-2024` / `LZTBARABANKI`. **hiv000001 follow-up (2026-08-21):** mixed-polarity location joined `LITO-RARARANKI L1TO-RARARANKI LIT-BRBNKI`; metadata invert turned `02` into `03` at t=10s. Parser now picks one hyphenated label; metadata ROI is not inverted; missing time with a known date goes to bands (not 4× wide). Intake location becomes `L1TO-RARARANKI` (still EasyOCR-limited vs `L1TO-BARABANKI`). **test_video.mp4 (2026-08-21):** opening OSD clock OCR'd as `08:38+31` (plus-as-colon) so time was empty; unhyphenated polarity variants were joined. Parser accepts `+` / `HH.MMSS` clocks and picks one long camera code (`08:38:31` / `L3TRARARANKT`). Prior clips rechecked: inframe t=2/t=10, shimoga, night. | `src/viana/stages/ocr.py`, `prescan.py` (`osd_band_score` multi-band), `time_map.py` (`07.21.26` + spaced-colon clock parse); `tests/viana/test_prescan.py` | uncommitted | `pytest tests/viana/test_prescan.py tests/viana/test_time_map.py` (37 passed in container) + inframe / shimoga / hiv00058 / night / test_video |
| **S22** | 1. Select/add two videos to intake job 2. Observe prescan/job processing in container 3. Repeat with one file after failure | **Expected:** intake and prescan work repeatedly without process/file descriptor exhaustion. **Actual (before):** `[Errno 24] Too many open files`; afterwards UI refresh hits API 502 (`fetch failed`) until container restart. **After (2026-08-21):** Root cause was leaked stdio pipes and orphaned process groups (prescan `run_viana` / GPU `Popen` kept by `Thread.args`, timeout/cancel not killing OpenCV/ffprobe/ffmpeg grandchildren, VideoCapture not released if the frame iterator was abandoned, ffmpeg stdin left open on hang). Engine+orchestrator now close pipes, kill the session, release captures, and join worker threads on success/fail/cancel. Next.js 502 remains a downstream symptom of EMFILE, not a UI bug. S24 live-monitor MP4 player stays unmounted. | `src/viana/io/proc.py`, `src/orchestrator/cli.py`, `src/orchestrator/workers/pool.py`, `src/viana/stages/{video,prescan,process,render}.py`, `src/viana/io/media.py` | uncommitted | `pytest tests/viana/test_proc.py tests/viana/test_process.py tests/orchestrator/test_s22_resources.py tests/orchestrator/test_job_routes.py` + FD loop (`/proc/self/fd`) |
| **S23** | 1. Run the same clip/config used previously for benchmark 2. Compare time-to-COMPLETED and average processing FPS vs earlier runs | **Expected:** throughput should be within an acceptable range of prior baseline. **Actual:** processing now takes significantly longer than before (user-observed regression). **After (2026-08-21, I003 / 6.9):** removed in-process OSD OCR (EasyOCR init + frame-0 parse + `recalibration_interval_sec` mid-run). Prescan OCR (S21) unchanged. Confirmed job metadata is the only clock; CSV/`time_map` interpolate `user_fallback`/`ocr_anchor` and never write `ocr_recalibrated` during `viana run`. **Benchmark** (same clip/config/GPU): `hiv000001_inframe.mp4` (2701 frames, 180s, 1920×1088 detect), geometry B, conf 0.75, `render_video=true`, `cuda:0` RTX 3060, container `viana_core`. Command: `python3 -m viana run -c <job.json>` (`start_fresh=true`). **Before (process-loop EasyOCR):** wall **203.2s**, avg FPS **13.45**, FPS@300 **6.43**. **After:** wall **179.3s**, avg FPS **15.26**, FPS@300 **14.43**. Stage split after: detect **161.7s**, track **2.7s**, render **9.1s**, telemetry **0.01s**, I/O **0.01s** (remainder ≈ decode + model/ffmpeg setup). GPU during after: ~84% util, ~1261 MiB. Dominant leftover cost is YOLO detect, not telemetry/I/O. On clips longer than 300s this also drops extra EasyOCR passes every 5 min. | uncommitted | `pytest tests/viana/test_process.py tests/viana/test_time_map.py tests/viana/test_prescan.py` + before/after `viana run` on `hiv000001_inframe.mp4` |
| **S24** | Live Monitor showed partial MP4 + delayed crossings | **Decision (2026-08-20):** Park in-progress video preview. **2026-08-21 (I001):** Live Monitor widget removed; job details shows progress + **Live Crossings** immediately (no frame delay). Do not remount the player. | parked: `live-processed-video.tsx`, `crossing-media-sync.ts`; Live Crossings: `live-crossings.tsx` | uncommitted | UI: no `<video>` in details; crossings update on emit |
| **S25** | 1. Submit job → wait for GPU while still in prescan queue 2. Confirm review → wait for GPU before `PROCESSING` 3. Scan Job Queue badges for all statuses | **Expected:** waiting-for-resource states read consistently; every lifecycle status has a clear operator-facing name. **Actual (before):** `PRESCAN_PENDING` showed **Queued** while post-review `READY` showed **Ready**. **After:** both wait states use **Queued** plus the resource — `Queued (PS)` / `Queued (GPU)`. Other labels: Pre-scanning, Prescan failed, Needs review (distinct from the Review action). API enums unchanged. | `job-status.ts` `STATUS_LABELS` + `STATUS_HINTS`; queue + job-details badges; `docs/ui/REDESIGN.md` / `DISCOVERY.md` | uncommitted | typecheck + label matrix (all `JobStatus`) |
| **S26** | 1. Queue a non-reviewable job (e.g. `PRESCAN_PENDING` / `PROCESSING`) — only Stop shows 2. Move same job to `AWAITING_REVIEW` / `READY` — Review appears left of Stop 3. Compare Actions column across rows | **Expected:** action icons stay in fixed positions. **Actual (before):** Actions mounted only when valid, so Stop jumped. **After:** non-completed rows always show **Review**, **Restart (Overwrite)**, **Stop** (muted when N/A; tooltip is the action name, no “Unavailable” prefix). `COMPLETED` shows **Open output** only. Retry prescan and Resume are not queue actions. | `job-queue-table.tsx`; `RoundIconButton` disabled + dark variants; `docs/ui/REDESIGN.md` / `COMPONENT_MAP.md` | uncommitted | typecheck + slot enable matrix vs `JobStatus` |
| **S27** | 1. Confirm ≥2 jobs so one is `PROCESSING` and the next is `READY` 2. Let the first job fail (engine/worker error → `FAILED`) 3. Observe GPU free and next job status | **Expected:** when a GPU slot frees on terminal `FAILED` (or equivalent), `_drain` starts the next FIFO `READY` job without operator intervention. **Actual (before):** previous job failed; GPU appeared free; next job stayed waiting. **After (2026-08-21):** `_drain` skips stale/non-READY queue heads; `_monitor`/`_finalize` always drain after FAILED (and spawn failure / missing process); `_release_gpu_slot` clears `gpu_device` + `process` so occupancy cannot stick. | `src/orchestrator/workers/pool.py`; `tests/orchestrator/test_s27_drain.py` | this commit | `pytest tests/orchestrator/test_s27_drain.py` (fail→next + stale-head skip) |
| **S28** | 1. Process `hiv00013_shimoga.mp4` (`job_0349289d5fe6`) 2. Watch white SUV/Jeep approach counting line ~06:44:50 3. Inspect `_events.csv` | **Expected:** one Jeep (or Car) crossing around 06:44:50. **Actual (before):** overlay showed `Car #3` then Jeep with class flicker; track absent from events (gap 06:44:43→06:45:01). Replay: track present frame 254 (side &lt; 0), missing 255–256, back at 257 already past line — `CrossingState` cleared `_previous` on any absence so no event. **After:** retain last bottom-center up to `max_gap_frames=15`; same window emits Jeep `in` at frame 257 (~06:44:52). | `src/viana/stages/crossing.py`, `tests/viana/test_track_crossing.py` | uncommitted | unit tests + shimoga window replay |
| **S29** | 1. Run a job to `COMPLETED` 2. List project `output_dir` for that stem | **Expected:** operator-facing deliverables are clear; ephemeral/intermediate artifacts are cleaned or isolated. **Actual:** many leftovers remain (JSON sidecars, prescan images, etc.) cluttering the output tree — inventory all files written before success and decide keep vs delete vs relocate (see F024). | `artifact_paths` / `prescan_dir` / profiles; engine write sites; ADR or docs for retention | — | Step 4 / hardening UI chat |
| **S30** | 1. From Job Queue, restart/resume a paused or failed job (UI Start Fresh / Resume) 2. Watch dashboard refresh | **Expected:** action succeeds or returns a clear API error; `GET /jobs` polling keeps working. **Actual:** Next.js throws `ApiClientError` **API 502: fetch failed** from `parseJson` → `Dashboard.refreshJobs` (`api-client.ts:166`). May be API down / proxy / EMFILE recurrence (S22) — triage required (see F025). | `dashboard.tsx` `onResume`/`onStartFresh` + `refreshJobs`; orchestrator resume/start-fresh; container logs | — | Step 4 / hardening UI chat |
| **S31** | 1. Open prescan review modal 2. Inspect header Close vs footer Cancel 3. Inspect primary footer button label | **Expected:** one dismiss control; primary action reads **Confirm**. **Actual:** header **Close** duplicates footer **Cancel**; primary button says **Submit**. | `apps/web/src/features/prescan/prescan-review-modal.tsx` | — | Step 4 / hardening UI chat |
| **S32** | 1. Inspect `{stem}_events.csv` and `{stem}_15min.csv` headers after a COMPLETED run 2. Cross-check `events_raw` / `events_15min` schemas + writers/parsers | **Expected:** CSV columns match operator/report needs only; no redundant or unused fields. **Actual:** schemas carry a wide column set (class taxonomy duplicates, debug anchors, etc.) — relook and trim to necessary (see F027). Contract-first if columns change. | `packages/contracts/schemas/events_raw.schema.json`, `events_15min.schema.json`; `csv_schema.py` / aggregate / UI parsers | — | Step 4 / hardening UI chat |

**Lane:** `A` UI · `B` API/orchestrator · `C` engine · `D` contract · `TBD`  
**Status:** `open` · `in_progress` · `fixed` · `deferred` · `parked` · `wontfix`  
**Blocker:** `yes` = gates Step 5 · `no` = polish

---

## F003 design note (scrub vs OCR)

Step 3 G8 implemented scrub as **full prescan re-run** (`pool.prescan_preview` → `viana prescan --frame-offset`). That was the fastest path to “live preview” but conflates two actions:

| Operator action | Should call | Should not call |
|-----------------|-------------|-----------------|
| Move frame slider | Local video seek (S02–S04) | `GET /prescan/preview` |
| **Re-scan OCR at Ns** | `GET /prescan/preview?frame_offset_sec=N` (S05) | — |

Optional fallback (only if browser codec fails): lightweight `GET /jobs/{id}/frame.jpg?offset_sec=N` — engine `sample_video_cv2` only; not on critical path unless S04 blocked.

---

## F007 design note (line proposal)

**Today:** Matching calibration profile still wins (`confidence` 0.85). Else `propose_lines(..., frame=)` uses the prescan sample: OSD-masked Hough, slope cluster scored in the road band (not rooftops), horizon intercept in the far/mid band, counting line parallel at `0.26 H`. No frame / weak cues → `geometric_lines()` (`confidence` 0.4). Endpoints stay clamped to frame bounds.

**Target:** Use the prescan sample frame (same frame as OCR / preview JPEG) to propose horizon and counting lines that align with visible road geometry — e.g. edge/vanishing-point heuristics, optional lane/horizon cues — while staying within frame bounds and returning `ProposedLines.confidence` honestly.

**Not Step 5 blocker:** operator can edit lines before confirm (discovery Q#4). Improves UX and reduces calibration time.

**Reference clips:** `hiv000001_inframe.mp4`; human-review geometry in `tests/viana/fixtures/PARITY_NOTES.md` § geometry B/D.

**Session:** Step 3 engine patch after S07 (or parallel with S08). Lane C only.

---

## F013 design note (theme consistency)

**Observed:** action buttons in intake/prescan surfaces do not consistently return to light-theme colors after a dark→light toggle.

**Hypothesis:** variant class composition or per-component theme branching is stateful/one-way (dark applied, light not fully restored), likely in local UI wrappers rather than global theme provider.

**Target:** centralize button variant mapping so dark and light tokens are both explicit and reversible for every relevant state (`default`, `secondary`, `ghost`, destructive/cancel), then verify on all repro controls listed in S16.

**Scope:** UI-only (Lane A), no API/engine changes.

---

## F014 design note (crossing UI dedupe)

**Observed:** crossing data appears in both job details (`Recent crossing`) and live monitor (`Live crossing`), creating redundant surfaces.

**Target (S17):** remove the `Recent crossing` table from job details and keep live crossing visualization in one place.

**Superseded (I001 / 6.8):** Live Monitor widget is gone; the single surface is **Live Crossings** in job details. Count uses API `progress.crossing_count` (I002 / 6.10), not session list length.

**Scope:** UI-only (Lane A), no API or telemetry contract changes.

---

## F015 design note (crossing title + time format)

**Observed:** live monitor crossing section has inconsistent title casing and non-standardized time display.

**Target:** use title case (`Live Crossings`) and render crossing event time as actual detected time in strict `HH:MM:SS` format.

**Dependency:** S14 timestamp emission path, so formatter consumes canonical event timestamp consistently.

**Scope:** UI formatting/copy in Lane A, with event-time source alignment from Lane C telemetry path.

---

## F016 design note (duration + ETA correctness)

**Observed:** queue shows unrealistic remaining time (e.g., 3h source appearing as 20h+), suggesting unit mismatch or bad source duration inputs.

**Target:** make `Video length` and `Time remaining` numerically trustworthy by:
- using canonical duration fields from API when available,
- ensuring ETA math uses consistent units (seconds vs frames vs fps),
- and validating MP4 metadata extraction in container runtime for deployed codecs.

**Source of truth (S19):**
- **Video length** = `video_duration_sec` from prescan `video_meta.duration_sec` (seconds of playback). Engine `src/viana/io/media.py` rejects container durations whose implied bitrate is &lt; 80 kbps (typical Hikvision MPEG-PS `.mp4` with a DVR clock span) and replaces them with `ffprobe -count_packets / fps`.
- **Time remaining** = `(total_frames − current_frame) / processing_fps` seconds. `processing_fps` is GPU/decode throughput, not source fps. Inflated `CAP_PROP_FRAME_COUNT` made this look like 20h+ on a ~38 min clip.

**Container check:** confirm codec/probe support for MP4 properties (duration/frame count/fps). If missing or inconsistent in current image, add/setup required tooling/libs and wire fallback probing.

**Scope:** cross-lane A/B/C (UI display, API fields, engine/container probe path). No contract-breaking changes without Lane D follow-up.

---

## F017 design note (adaptive OSD extraction)

**Observed:** fixed ROI assumptions fail on videos where timestamp/location overlays move (e.g., location top-center, timestamp bottom-left).

**Target:** prescan OCR should detect metadata robustly across varying overlay positions by combining:
- broader ROI candidates (top/bottom/center bands),
- confidence-ranked merge logic,
- and a safe fallback pass when corner ROIs miss required fields.

**Constraint:** maintain performance improvements from S08 while improving recall on layout-variant cameras.

**Scope:** engine prescan OCR path (Lane C), no UI/API contract shape changes.

---

## F018 design note (file descriptor exhaustion)

**Observed:** multi-file intake can trigger `[Errno 24] Too many open files`; after that, even single-file operations fail and UI sees API 502 until container restart.

**Target:** eliminate FD leaks and make worker lifecycle resilient by:
- closing subprocess pipes and video/ffmpeg handles deterministically,
- joining/tearing down worker threads/processes on terminal states,
- and adding guardrails/diagnostics for FD growth under repeated intake/prescan cycles.

**Symptom link:** Next.js `ApiClientError` 502 during `refreshJobs` is likely secondary once orchestrator becomes unhealthy.

**Scope:** orchestrator + engine runtime resource management (Lane B/C), plus minimal UI error-surface hardening if needed.

---

## F019 design note (processing performance regression)

**Observed:** end-to-end processing time regressed versus earlier runs on comparable inputs.

**Target:** restore throughput by profiling and fixing the dominant bottleneck(s) without reducing output correctness:
- compare baseline vs current (same clip, same config, same hardware),
- break down time across decode, inference, tracking, rendering, telemetry, and disk I/O,
- and lock a repeatable performance check to catch regressions.

**Fix (2026-08-21):** In-process OSD OCR (EasyOCR loaded on every `viana run`, first-frame parse, recalibration every `ocr.recalibration_interval_sec`) was the recoverable regression vs confirmed-prescan clock. Removed from `process.py` (I003 / Step 6.9). Remaining time on `hiv000001_inframe.mp4` is almost all detect (161.7s / 179.3s wall); track/render/telemetry/I/O are small. Repeat: same job JSON, `start_fresh=true`, `python3 -m viana run -c …` in `viana_core`.

**Scope:** engine/runtime performance path (Lane C), with container/runtime config verification as needed.

---

## F010 / S24 design note (live-monitor partial MP4 — PARKED)

**Decision (2026-08-20):** Do not show in-progress `_processed.mp4` in the UI for now. Still in force after I001 (job details; widget gone).

**Why:** After S13 (fragmented MP4) and S20 (H.264 for Chromium), browser live-edge preview remained unstable — seek/reload blackouts, Range-request FD exhaustion (`Errno 24`), and unreliable picture↔crossing sync.

**Current UI:** Job details = progress line + **Live Crossings**. Crossings render WS `MOVING_EVENT`s **immediately** (no frame-buffer delay). Header total is `progress.crossing_count` (GET /jobs + WS PROGRESS).

**Retained (unused) code:**
- `apps/web/src/features/monitor/live-processed-video.tsx`
- `apps/web/src/features/monitor/crossing-media-sync.ts`

**Do not** import/mount those modules until this note is explicitly reversed. API `GET /artifacts/{id}/partial.mp4` and proxy remain available for a future revisit. Live Monitor widget was removed (I001); do not bring it back with the player.

---

## F020 design note (job status UI labels)

**Issue:** Operators see different waiting labels for similar “blocked on resources” situations:
- `PRESCAN_PENDING` → UI **Queued** (waiting for a prescan worker / slot)
- `READY` → UI **Ready** (confirmed; waiting for GPU processing pool)

API enum values stay as-is (`JobStatusLiteral` / contracts). This item is **UI copy + badge clarity** only unless review decides a contract rename is warranted.

**Decision (2026-08-21):** Share the word **Queued** and name the resource. Do not rename API enums.

**Operator labels** (`apps/web/src/features/queue/job-status.ts`):

| API status | UI label | Lifecycle meaning |
|------------|----------|-------------------|
| `PRESCAN_PENDING` | Queued (PS) | Submitted; waiting for prescan capacity |
| `PRESCAN_RUNNING` | Pre-scanning | Prescan worker active |
| `PRESCAN_FAILED` | Prescan failed | Prescan error; may retry |
| `AWAITING_REVIEW` | Needs review | Prescan done; operator must confirm |
| `READY` | Queued (GPU) | Confirmed; waiting for GPU / process pool |
| `PROCESSING` | Processing | Engine running |
| `PAUSED` | Paused | Processing paused |
| `COMPLETED` | Completed | Success terminal |
| `FAILED` | Failed | Error terminal |
| `CANCELLED` | Cancelled | Cancelled terminal |

Badge `title` uses `STATUS_HINTS` (e.g. READY = “Confirmed — waiting for a GPU slot”).

**Scope:** Lane A (UI). No Step 5 blocker.

---

## F021 design note (Job Queue action icon layout)

**Issue:** The Actions column mounts buttons only when applicable. That makes the Stop (red cross) icon shift horizontally depending on which other actions are present — e.g. alone on the left vs to the right of Review when the job is under review.

**Decision (2026-08-21):** Non-completed rows always render three slots; `disabled` + muted zinc (light + dark) when invalid. Tooltip keeps the action name (no “Unavailable:” prefix). **Retry prescan** and **Resume** are not queue actions. **Monitor** is not a slot (I001).

| Slot | Action | Enabled |
|------|--------|---------|
| 1 | Review | `AWAITING_REVIEW`, `READY`, `PRESCAN_FAILED` |
| 2 | Restart (Overwrite) | `PAUSED`, `FAILED` |
| 3 | Stop | not `COMPLETED` / `CANCELLED` |

`COMPLETED` rows show **Open output** only (the three slots are not rendered).

`RoundIconButton` enabled colors have explicit `dark:` hover tokens so light↔dark does not leave a pale hover flash (F013).

**Scope:** Lane A (UI). No Step 5 blocker. No API change.

---

## F023 design note (missed crossings across detection gaps)

**Observed (`hiv00013_shimoga`, ~06:44:50):** a white SUV/Jeep crosses the counting line in the processed overlay with Car↔Jeep label flicker, but no row appears in `_events.csv`.

**Root cause:** `CrossingState` deleted the last bottom-center whenever a track was absent for a single frame. Class flicker / confidence dips near conf=0.75 often drop the box for 1–2 frames exactly while the vehicle straddles the line; when the box returns on the far side, `previous is None` so no crossing is emitted (once-per-track never fires).

**Fix:** keep the previous anchor for up to `DEFAULT_MAX_GAP_FRAMES` (15) empty frames; still once-per-track; forget after long disappearances to avoid false late counts.

**Scope:** Lane C (`crossing.py`). Re-process affected clips to refresh events CSV.

---

## F022 design note (drain after FAILED)

**Observed:** After a processing job entered `FAILED`, a subsequent `READY` job did not start even though a GPU was free.

**Expected behavior:** FIFO execution — when a GPU worker finishes (success or failure), `JobPool._drain()` should assign a free device to the head `READY` job and spawn it.

**Fix (2026-08-21):**
1. `_monitor` always calls `_drain()` in `finally` (FAILED JSON, non-zero exit, monitor exception, missing process). Spawn errors mark FAILED, release the slot, and continue draining.
2. `_drain` pops stale queue heads (`status != READY` or missing record) instead of returning.
3. `_release_gpu_slot` clears `process` and `gpu_device` on every leave-PROCESSING path so `occupied_gpus()` cannot keep a dead assignment.

**Scope:** Lane B (orchestrator). No Step 5 blocker (ops reliability; triage may promote if reproducible).

---

## F024 design note (output leftovers after COMPLETED)

**Observed:** After a successful job, the project output directory still contains many non-deliverable files (JSON sidecars, prescan images, etc.).

**Work:**
1. Inventory every path written from intake → prescan → run → aggregate (start from `artifact_paths`, `prescan_dir`, `profiles_dir`, job JSON, checkpoints, run_result, time_map, preview JPEGs).
2. Classify each as: **operator deliverable** (e.g. `_events.csv`, `_15min.csv`, `_processed.mp4`), **runtime required** (checkpoint / status for resume), **debug/ephemeral**, or **orchestrator-only**.
3. Strategy options (pick deliberately, document):
   - delete ephemerals on `COMPLETED`;
   - move intermediates under e.g. `{stem}/.work/` or `prescan/` / `_meta/`;
   - keep only what resume / job status / re-review needs, with a clear directory layout.
4. Do not delete anything needed for PAUSED resume or audit without an ADR/docs update.

**Scope:** Lane B/C (+ docs). Design before mass delete.

---

## F025 design note (502 on UI restart/resume)

**Observed:** Restarting/resuming a job from the UI surfaces `Runtime ApiClientError` — **API 502: fetch failed** in `parseJson` during `Dashboard.refreshJobs` (`apps/web/src/lib/api-client.ts`).

**Triage:**
1. Reproduce with Start Fresh vs Resume; capture whether the mutate call fails or only the subsequent `GET /jobs` poll.
2. Check API/container logs at the same moment (crash, timeout, proxy, EMFILE — related history S22).
3. Harden UI: avoid unhandled throw on refresh; show recoverable banner if API is briefly unreachable.
4. Fix root cause in orchestrator/engine if restart path takes down or saturates the API.

**Scope:** Lane A/B. Related to S22 if FD exhaustion returns.

---

## F026 design note (prescan Close vs Confirm)

**Observed:** Prescan review modal has header **Close** and footer **Cancel** doing the same dismiss; primary footer label is **Submit**.

**Fix:** Remove the redundant Close control; rename Submit → **Confirm** (and loading copy to **Confirming…** if present). Keep a single cancel/dismiss path.

**Scope:** Lane A. `prescan-review-modal.tsx` only unless shared dialog chrome is involved.

---

## F027 design note (CSV schema trim)

**Goal:** Relook `_events.csv` (raw) and `_15min.csv` so each column is justified for operators, aggregation, or a documented internal need — drop the rest.

**Current surfaces:**
- Raw: `packages/contracts/schemas/events_raw.schema.json` (e.g. ids, timing, wall_time*, class taxonomy, anchors, confidence…)
- 15-min: `packages/contracts/schemas/events_15min.schema.json` (window/date/location, class taxonomy, direction, count, partial)

**Review asks:**
1. List every column with producer + consumer (engine write, aggregate, UI, external report).
2. Mark keep / drop / optional-debug (debug may move to a separate artifact, not the operator CSV).
3. Especially scrutinize overlapping class fields (`class_name` vs `category` / `class_type` / `sub_class` / raw_*), geometry (`anchor_x`/`y`, `norm_area`), and duplicate location/date on both files.
4. If columns change: schema → engine writers → aggregate → UI parsers (CONTRACT_SYNC); note S15 date/HH:MM decisions stay unless explicitly revisited.

**Scope:** Lane C/D (+ A if UI parsers change). Design/approval before deleting columns.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-21 | **S09 (F006) fixed via Step 6.7** — intake/submit rewrite host paths onto `/data` and `/app/ViAna` or 400; extra mounts via `VIANA_INTAKE_ROOTS` + `VIANA_PATH_MAPS` |
| 2026-08-21 | Added **S32 (F027)** relook raw events + 15-min CSV schemas — keep only necessary columns |
| 2026-08-21 | **S25 (F020) + S26 (F021) fixed:** queue labels `Queued (PS)` / `Queued (GPU)`; actions Review → Restart (Overwrite) → Stop; Open output only on `COMPLETED` |
| 2026-08-21 | **S27 (F022) fixed:** after PROCESSING→FAILED, `_drain` starts the next FIFO READY job; skip stale queue heads; clear GPU occupancy |
| 2026-08-21 | Added **S29 (F024)** output leftovers after COMPLETED — inventory + retention/directory strategy |
| 2026-08-21 | Added **S30 (F025)** API 502 `fetch failed` when restarting/resuming job from UI (`refreshJobs`) |
| 2026-08-21 | Added **S31 (F026)** prescan review: remove duplicate Close; rename Submit → Confirm |
| 2026-08-21 | **S28 (F023) fixed:** retain counting-line previous anchors across brief detection gaps (class flicker); recovers missed Jeep on `hiv00013_shimoga` ~06:44:50 |
| 2026-08-21 | Added **S27 (F022)** next READY job did not start after prior job FAILED despite free GPU |
| 2026-08-21 | Added **S26 (F021)** standardize Job Queue action icons (Stop jumps left/right when Review mounts; prefer stable slots + enable/disable after UI standards check) |
| 2026-08-21 | **S10 fixed:** road-band slope clustering + parallel counting offset; `hiv000001_inframe` proposal near geometry C/D (endpoint \|dy\| 218/213 vs 984/887); profile override unchanged; `test_prescan.py` 29 passed |
| 2026-08-21 | **S23 fixed (I003 / 6.9):** no process-loop EasyOCR; confirmed prescan/user clock interpolated; `hiv000001_inframe` 203.2s/13.45 fps → 179.3s/15.26 fps (detect still ~90% of loop) |
| 2026-08-21 | **I001 / I002:** removed Live Monitor widget and action; Live Crossings in job details; count bound to `progress.crossing_count`; S24 player still unmounted |
| 2026-08-21 | Added **S25 (F020)** review job status UI labels (`Queued` vs `Ready` waiting inconsistency; full lifecycle naming pass) |
| 2026-08-21 | **S22 fixed:** close subprocess pipes / process groups, VideoCapture, and ffmpeg on success/fail/cancel; multi-file intake FD loop stable; UI 502 treated as EMFILE symptom (S24 player not remounted) |
| 2026-08-20 | **S19 fixed:** MPEG-PS/DVR header duration inflated video length/ETA; ffprobe packet recount when implied bitrate &lt; 80 kbps; units documented (sec vs frames vs processing_fps)
| 2026-08-20 | **S24 parked:** Live Monitor hides partial-MP4 preview (code retained, not mounted); Live Crossings show telemetry immediately with no UI delay; S20 follow-on live-edge work parked
| 2026-08-20 | S20 fixed: in-progress UI blank was HEVC-in-browser (headers/Range OK); prefer H.264 NVENC/libx264 for `_processed.mp4`, keep S13 fragmented flags; inline disposition on artifact/proxy; live monitor decode-error UX |
| 2026-08-19 | S10 partial update: added dominant-slope + parallel-band fallback refinement for no-profile proposals and test coverage; result improved but not yet at desired line placement on all sample views, so S10 remains in_progress |
| 2026-08-19 | S10 fixed (engine): deterministic frame-guided line proposal fallback for no-profile prescan, bounds-safe clamped outputs, confidence derived from cue support; tests added for confidence uplift/determinism/frame-shape fallback |
| 2026-08-19 | S15 fixed (contract/output alignment): `_15min.csv` includes `date`, `window_start`/`window_end` are `HH:MM`; parser now keys rows by `date+window+class`; aggregate header/value expectations tightened in tests |
| 2026-08-19 | S13 fixed (fragmented MP4 writer hardening + tests); S14 fixed (`MOVING_EVENT` timestamp payload enriched and verified on WS/tests) |
| 2026-08-19 | S11–S12 fixed — required `created_at`; `video_duration_sec` from prescan; frozen `processing_duration_sec` on GET /jobs |
| 2026-08-19 | S08 fixed — `viana prescan` 6.7s → 4.6s on `hiv000001_inframe.mp4`; 2× tight OSD ROI + t=2s probe; S07 metadata unchanged |
| 2026-08-19 | S11–S14 backend triage: created_at, video duration, streamable partial MP4, MOVING_EVENT |
| 2026-08-19 | Added S15 backend schema request for 15-min CSV window/date format alignment |
| 2026-08-19 | Added S16 (F013) UI theme toggle regression for intake + prescan action buttons (dark→light color restore) |
| 2026-08-19 | Added S17 (F014) remove duplicate `Recent crossing` table from job details (use live monitor crossing view only) |
| 2026-08-19 | Added S18 (F015) fix live monitor crossing title case + enforce actual-time `HH:MM:SS` formatter |
| 2026-08-19 | Added S19 (F016) queue video length/ETA inflation + container MP4 codec metadata validation |
| 2026-08-19 | Added S20 (F010 follow-on) UI still cannot render in-progress processed MP4 though native Ubuntu player can |
| 2026-08-19 | Added S21 (F017) prescan OCR misses metadata when OSD text shifts to non-corner regions |
| 2026-08-19 | Added S22 (F018) file-descriptor exhaustion (`Errno 24`) causing downstream API 502 until container restart |
| 2026-08-19 | Added S23 (F019) processing performance regression tracking and benchmark/profiling follow-up |
| 2026-08-19 | S06 fixed (EasyOCR triage); S07 fixed — corner ROI OCR, `proposed_metadata` on `hiv000001_inframe.mp4` (`job_abec59713960`) |
| 2026-08-19 | S09 (F006) deferred to Step 6.7; added S10 (F007) line proposal improvement |
| 2026-08-19 | Merged F001–F006 + F003 scrub plan into single execution path S01–S09 |
| 2026-08-19 | S03–S05 fixed: `/api/proxy/source`, video scrub canvas, prescan/preview on Re-scan only |
| 2026-08-19 | S01 fixed (`45a82a4` disk rglob fallback); S02 fixed (`GET /artifacts/{id}/source.mp4`) |
| 2026-08-19 | F001–F006 logged from Step 4 acceptance testing (UI chat thread) |
| 2026-08-19 | Backlog created; Step 5 gated on blockers |

---

## Deferred / wontfix (optional)

| Seq | Reason | Date |
|-----|--------|------|
| ~~S09~~ (F006) | Closed in **Step 6.7** (was deferred 2026-08-19; UI `container-paths.ts` remains) | 2026-08-21 |
