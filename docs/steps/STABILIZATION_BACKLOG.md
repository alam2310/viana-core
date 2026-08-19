# Stabilization backlog (living)

**Rules:** [`STABILIZATION.md`](STABILIZATION.md)  
**Step 5 blocked while any blocker row is `open` or `in_progress`.**

> **Follow [`Execution path`](#execution-path) in Seq order.** One row = one unit of work. Do not skip ahead unless a dependency is `fixed` / `deferred`.

**Last updated:** 2026-08-19

---

## Summary

| Blockers open | Blockers fixed | Polish open | Path steps done |
|---------------|----------------|-------------|-----------------|
| 0 | 1 | 9 | 13 / 22 active |

**Deferred to Step 6.7:** S09 (F006). **Not counted** in path progress.

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
| **S16** | F013 | A | no | — | Theme toggle regression: action buttons keep dark styling after returning to light mode | open |
| **S17** | F014 | A | no | — | Remove duplicate `Recent crossing` table from job details widget | open |
| **S18** | F015 | A/C | no | S14 | Normalize live monitor crossing title + `HH:MM:SS` actual-time formatter | open |
| **S19** | F016 | A/B/C | no | S12 | Fix queue video length / ETA inflation and validate MP4 codec metadata in container | open |
| **S20** | F010 | A/B | no | S13 | UI cannot render in-progress processed MP4 even when file is playable natively | open |
| **S21** | F017 | C | no | — | Prescan OCR misses time/location when OSD text appears in alternate screen regions | open |
| **S22** | F018 | B/C | no | — | Intake/prescan triggers `[Errno 24] Too many open files`, followed by API 502 in UI refresh | open |
| **S23** | F019 | C | no | — | Processing throughput regression: end-to-end run is much slower than earlier baseline | open |
| ~~**S09**~~ | F006 | B | no | — | API rejects container-unreadable intake paths | **deferred → Step 6.7** |

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
| **S10** | 1. Intake `hiv000001_inframe.mp4` (or parity clip) 2. `AWAITING_REVIEW` 3. Open review modal | **Expected:** `proposed_lines` match road geometry (horizon near vanishing point, counting line on lane boundary) — usable without large edits. **Actual (before):** `propose_lines()` used fixed normalized y-ratios or aspect-matched profile only (`lines.py` `geometric_lines`) with weak frame cues. **After:** when no profile matches, prescan uses deterministic frame-guided edge/segment fitting in horizon/counting bands with confidence tied to visual support; profile override remains authoritative; invalid/noisy frame data falls back to geometric defaults. | `src/viana/stages/lines.py`, `src/viana/stages/prescan.py`, `tests/viana/test_prescan.py`; reference geometry targets in `tests/viana/fixtures/PARITY_NOTES.md` | uncommitted | unit tests |
| **S11** | 1. Intake job 2. `GET /jobs` | **Expected:** each job has `created_at` (ISO) for queue sort. **Actual:** UI uses localStorage fallback until API field exists. | `job_status.schema.json`, `pool.py` `JobRecord`, `to_status()` | `81106c0` | orchestrator tests (S11) |
| **S12** | 1. Prescan completes 2. `GET /jobs/{id}` | **Expected:** `video_duration_sec` and `processing_duration_sec` on status for queue columns. **Actual:** UI estimates video length from progress fps when available; run time uses localStorage timestamps captured while dashboard is open during `PROCESSING` — shows `—` for jobs completed before first poll or after page refresh. | prescan worker persists meta on `JobRecord`; schema + TS types; `job-local-meta.ts` | `81106c0` | orchestrator tests (S12) |
| **S13** | 1. `PROCESSING` job 2. Open live monitor 3. Try partial MP4 in VLC | **Expected:** `_processed.mp4` playable while growing. **Actual (before):** file not playable until job completes (moov atom at end). **After:** FFmpeg writer always emits fragmented MP4 with frequent fragments/keyframes (`frag_duration=1s`) so in-progress output is streamable. | `src/viana/stages/render.py`, `tests/viana/test_render.py`, `tests/orchestrator/test_job_routes.py` | uncommitted | `pytest tests/viana/test_render.py tests/orchestrator/test_job_routes.py -q` |
| **S14** | 1. Confirm prescan with `telemetry_detail: true/false` 2. Monitor during `PROCESSING` | **Expected:** `MOVING_EVENT` on WS for every crossing with wall-clock or video timestamp. **Actual (before):** events gated on `telemetry_detail`; no timestamp in payload (frame_index only). **After:** emission is unconditional and payload carries `event_timestamp`, `event_timestamp_source`, `event_timestamp_confidence`, `video_pts_ms` (plus existing frame/fps fields). | `src/viana/stages/process.py`, `tests/viana/test_process.py`, `tests/orchestrator/test_job_routes.py` | uncommitted | `pytest tests/viana/test_process.py tests/orchestrator/test_job_routes.py -q` |
| **S15** | 1. Generate `_15min.csv` 2. Inspect header/rows | **Expected:** `date` plus `window_start`/`window_end` as `HH:MM` with strict schema header order. **Actual (before):** UI parser grouped by window/class only and could merge multi-day rows. **After:** contract/schema + aggregate writer emit `date` and `HH:MM`; parser requires `date` column and keys by `date+window+class` while remaining backward-compatible with ISO window values. | `packages/contracts/schemas/events_15min.schema.json`, `src/viana/io/csv_schema.py`, `src/viana/stages/aggregate.py`, `apps/web/src/lib/parse-15min-csv.ts`, `tests/viana/test_cli_aggregate.py` | uncommitted | `npm --prefix apps/web run typecheck`; python `pytest` unavailable in this environment |
| **S16** | 1. Switch to dark theme 2. Return to light theme 3. Visit intake, prescan dialogs, and live monitor | **Expected:** button colors and variants re-compute on every theme change. **Actual:** several buttons remain in dark-style colors after switching back to light. Repro controls: top widget `Output directory` browse button; Select Files dialog `Cancel` + `Add all videos` + `Add selected`; Prescan review dialog `Close` + `Cancel` + `Re-scan OCR`; Live monitor dialog `Close`. | Lane A (`apps/web/`): inspect variant token mapping and theme-driven class/state updates in project/intake + prescan-review + live-monitor components; ensure dark↔light toggles are symmetric and not cached/stale. | — | Step 4 UI chat |
| **S17** | 1. Open job details widget 2. Open live monitor for same job | **Expected:** crossings are shown in one canonical place (live monitor `Live crossing` view), avoiding duplicate UI sections. **Actual:** job details also shows `Recent crossing`, duplicating information and adding noise. | Lane A (`apps/web/`): remove `Recent crossing` table from job details widget; keep crossing visibility in live monitor only; verify no empty-state/layout regressions in details card. | — | Step 4 UI chat |
| **S18** | 1. Open live monitor while crossings stream 2. Inspect section header + crossing time column | **Expected:** title follows title case (`Live Crossings`) and displayed time is actual identified event time formatted strictly as `HH:MM:SS`. **Actual:** header casing is inconsistent (`Live crossing`) and time formatting is not constrained to `HH:MM:SS` actual-time display. | Lane A/C: update live monitor copy + formatter in `apps/web/` to render `HH:MM:SS`; ensure value is sourced from actual event timestamp (from S14 payload path) rather than derived/local clock text. | — | Step 4 UI chat |
| **S19** | 1. Queue a known long clip (~3h) 2. Observe Job Queue `Video length` and `Time remaining` | **Expected:** video length and ETA are in the right order of magnitude (3h clip should not show 20h+ remaining without evidence). **Actual:** ETA/video-length logic appears inflated/incorrect for long MP4 inputs. Also need to confirm container can read MP4 duration/properties reliably for this codec profile. | Lane A/B/C: audit ETA formula and units in `apps/web/` + API status fields (`video_duration_sec`, `processing_duration_sec`, fps/progress source). Validate media probe path in container (OpenCV/ffprobe/codec support) and fix image/runtime setup if metadata extraction is wrong for MP4 variants. | — | Step 4 UI chat |
| **S20** | 1. Start a job and wait for `_processed.mp4` to grow 2. Verify file plays via Ubuntu native player 3. Open same artifact in live monitor/UI | **Expected:** if file is already stream-playable on disk, the UI player should render it while processing. **Actual:** native apps can play in-progress file, but browser/UI still fails to render (same user-facing symptom persists). | Follow-on under F010: Lane A/B to inspect artifact endpoint/proxy headers (Range, Content-Type, Accept-Ranges, CORS, cache), player source URL lifecycle, and browser codec/container compatibility vs native playback. | — | Step 4 UI chat |
| **S21** | 1. Run prescan on a video where OSD layout differs (location at top-center, timestamp at bottom-left) 2. Inspect `proposed_metadata` | **Expected:** prescan extracts time/date/location despite changed on-screen text positions. **Actual:** current corner-ROI logic misses fields when OSD is not in assumed regions. | Lane C (`src/viana/stages/ocr.py`, `prescan.py`): extend OCR region strategy beyond fixed corners (adaptive multi-region scan / fallback full-frame text pass), preserve confidence handling, and avoid regressions on existing clips. | — | Step 4 UI chat |
| **S22** | 1. Select/add two videos to intake job 2. Observe prescan/job processing in container 3. Repeat with one file after failure | **Expected:** intake and prescan work repeatedly without process/file descriptor exhaustion. **Actual:** `[Errno 24] Too many open files`; afterwards UI refresh hits API 502 (`fetch failed`) until container restart. | Lane B/C: investigate FD leaks in orchestrator workers and engine prescan/process paths (open files, pipes, VideoCapture/ffmpeg handles, subprocess stdio). Add runtime diagnostics (`lsof`/fd counts) and ensure cleanup on success/failure/cancel. Confirm Next.js proxy 502 is downstream symptom, not root cause. | — | Step 4 UI chat |
| **S23** | 1. Run the same clip/config used previously for benchmark 2. Compare time-to-COMPLETED and average processing FPS vs earlier runs | **Expected:** throughput should be within an acceptable range of prior baseline. **Actual:** processing now takes significantly longer than before (user-observed regression). | Lane C: capture before/after metrics (wall-clock, avg FPS, GPU utilization), identify regressions in detect/track/render/telemetry path, and isolate whether slowdown is model/runtime/container/config-related. Add reproducible benchmark command and clip in notes. | — | Step 4 UI chat |

**Lane:** `A` UI · `B` API/orchestrator · `C` engine prescan · `D` contract · `TBD`  
**Status:** `open` · `in_progress` · `fixed` · `deferred` · `wontfix`  
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

**Today:** `propose_lines(width, height, profiles)` never inspects the sampled frame pixels. Fallback is `geometric_lines()` with fixed norms (`_HORIZON_Y`, `_COUNTING_Y`) or a profile matched by aspect ratio only (`confidence` 0.4 vs 0.85).

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

**Target:** remove the `Recent crossing` table from job details and keep live crossing visualization in monitor as the single source in UI.

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

**Scope:** engine/runtime performance path (Lane C), with container/runtime config verification as needed.

---

## Changelog

| Date | Change |
|------|--------|
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
| **S09** (F006) | API intake path validation — UI mitigated via `container-paths.ts`; full fix → **Step 6.7** | 2026-08-19 |
