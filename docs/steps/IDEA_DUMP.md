# Idea dump (manual review only)

**Status:** parking lot — **not** a work queue  
**Last updated:** 2026-08-22 (I007 promoted → 6.12)  
**Owner:** human (this chat / later review)

> **Agents: do not implement, prioritize, or start a Step/Seq from this file.**  
> Capture ideas here only when the user dumps them in the idea-dump chat.  
> Do **not** pull items into `TRACKER.md`, `STABILIZATION_BACKLOG.md`, or `STEP_6_HARDENING.md` unless the user **explicitly names an idea ID** (e.g. `I003`) and asks to promote it.

This file exists so active coding sessions stay on the current Step. Come back here after that work is done, scan the list, and pick what is worth promoting.

---

## How this list is used

| Role | What to do |
|------|------------|
| **You (later)** | Read the table, keep / drop / promote. Promotion = copy into Step 6, stabilization, or a new Step — then set status `promoted`. |
| **Idea-dump chat** | Append rows only. Suggest **impact** + a short **comment**. Do not open patches. |
| **All other agents** | Ignore this file unless the user names an idea ID and asks to work it. |

**Impact (suggested, not committed):**

| Level | Meaning |
|-------|---------|
| **critical** | Blocks correctness, safety, or a signed-off product promise |
| **high** | Clear user/ops pain or large quality/speed win |
| **medium** | Useful; not blocking current Step |
| **low** | Nice-to-have, speculative, or high cost vs benefit |

**Status:** `open` · `review` · `promoted` · `dropped`

### Review ranking (2026-08-21)

Human review of I001–I007. Promoted Step items **6.8–6.11 are ✅ complete**. **6.12** is the active promoted item from I007. Remaining dump rows (**I004**, **I005**) are not work until re-promoted.

| Rank | ID | Decision |
|------|----|----------|
| 1 | **I001** | **Promoted** → Step **6.8** (UI) — ✅ done |
| 2 | **I003** | **Promoted** → Step **6.9** (engine) — ✅ done |
| 3 | **I002** | **Promoted** → Step **6.10** (UI) — ✅ done |
| 4 | **I006** | **Promoted** → Step **6.11** (UI) — ✅ done (`render_video` toggle) |
| 5 | **I007** | **Promoted** → Step **6.12** (UI + API + engine) — `track_pedestrians` toggle; skip pedestrian YOLO + exclude from `_15min.csv` when false |
| 6 | **I005** | Stay in dump — mux soft event subs into `_processed.mp4` (direction set) |
| 7 | **I004** | **Demoted** — later / aesthetics. Do not start from this dump. |

---

## Open ideas

| ID | Date | Category | Idea | Impact | Comment | Status |
|----|------|----------|------|--------|---------|--------|
| I004 | 2026-08-21 | `ui` | Play processed video from job details; investigate in-progress (play bytes already written, not live-edge) vs complete-only | **low** | **Demoted** (aesthetics / later). Not in Step 6. Job details exist (6.8); still do not start until re-promoted. | open |
| I005 | 2026-08-21 | `engine` / `product` | Crossing events as soft subtitles muxed into `_processed.mp4` (post-process from `_events.csv`) | **medium** | **Direction set:** CSV → soft SRT/VTT → FFmpeg mux into a **single** deliverable file (`-c:v copy`). No burn-in; no GPU-loop change. See detail below. | open |

### Categories (use these labels)

`engine` · `api` · `ui` · `contracts` · `ops` · `product` · `dx` · `qa` · `docs`

### I001 — Job details instead of Live Monitor widget (promoted → 6.8)

**Dump:** The Live Monitoring widget complicates the flow. Remove it entirely and put the Live Crossing view in job details. Remove the Live Monitor action button. A click on the job row opens job details, which makes it easier to move between jobs.

**Suggested impact: high.** Queue UX is currently split (row vs Live Monitor action vs a separate widget). Folding crossings into details is a product simplification, not a ship blocker. Likely work: queue row click → details surface; relocate Live Crossings; unmount/remove monitor widget and its action. Watch for in-progress jobs that still need some live signal in details.

### I002 — Authoritative live crossing count (promoted → 6.10)

**Dump:** The live crossing count looks UI-only. It restarts when Live Monitoring is closed and opened again, so it does not reflect true status. Change it to the actual total from the backend if the current API already exposes it, or drop the count from UI for now.

**API check (2026-08-21):** Field exists — `JobStatus.crossing_count` (`job_status.schema.json`, orchestrator `models.py`). Worker updates it from process telemetry / WS PROGRESS. **Do not add a schema field.** Bind the UI badge to this total (GET job + progress events), not `crossings.length` from the in-session WS table. Fits 6.8 details; can also fix the current Live Crossings header.

**Suggested impact: high.** A resetting count is worse than no count.

### I003 — No OSD OCR during processing (promoted → 6.9)

**Dump:** Prescan OCR accuracy is improved and is manually reviewed before the job starts. Do not OCR-rescan while the video is processing. Completely remove the in-process text scan for time, date, and location. That should improve overall performance and stop drifting values in the CSV.

**Suggested impact: high.** Two wins: (1) CSV wall times stay on the confirmed prescan/user clock instead of later OSD misreads; (2) EasyOCR off the GPU loop. Today `process.py` still runs OSD parse on an initial frame and again every `ocr.recalibration_interval_sec` (default 300s), writing `ocr_recalibrated` into the time map / events. Keep OCR on **prescan only**; interpolate from the confirmed anchor for the rest of the clip. Watch: jobs with no confirmed datetime, and tests that assert mid-run recalibration.

### I004 — Play processed video in job details

**Dump:** Add a play-video option in job details for the processed video. Investigate whether it is safe to keep this while the job is still processing (play what is already written, **not** as live), or restrict playback until the job completes.

**Suggested impact: low (demoted).** Completed-job playback is a details-page nicety and fits 6.8 later. In-progress playback is an open product/tech call: S13 already writes fragmented `_processed.mp4`; S20/S24 parked **live-edge** preview because browser Range/seek was unstable. Not Step 6 until re-promoted.

### I005 — Crossing events as video subtitles

**Dump:** Investigate how subtitles work in a video and whether crossing events can be added as subtitles on the processed video. Assess the performance impact.

**Suggested impact: medium.** This is a research item, not a Step 6 patch. Crossing rows already have video timestamps in `_events.csv`, so a **post-process sidecar** (SRT / WebVTT) or an optional muxed text track is the low-cost path: no extra GPU work in the detect/track loop. **Burned-in** captions (FFmpeg `subtitles`/`drawtext` on every frame) would compete with the current `_processed.mp4` encode and likely regress FPS (see S23). Investigation should cover: (1) soft vs hard subs, (2) cue text (class, direction, wall time), (3) player support in VLC vs browser, (4) mux-at-end vs grow-during-job (fragmented MP4 + timed text is harder). Do not implement until promoted.

**Conclusion (2026-08-21):** Prefer soft subtitles as an **event log**, not on-box burn-in or ASS spatial cues. After the job completes, **post-process** `_events.csv` → SRT/WebVTT cues (`video_pts_ms`, class, direction, wall time), then **mux** that track into `_processed.mp4` so operators get a **single file** (FFmpeg stream copy for video; no re-encode). Performance and size impact stay near zero. Skip mid-run mux on fragmented growing MP4. Burn-in / client overlay remain out of scope unless re-promoted later (I004 for in-app playback).

### I006 — Render-video flag in prescan review (promoted → 6.11 — ✅ complete)

**Dump:** Implement a video rendering flag in prescan review and pass it through the API on job submission. Expect performance to improve when output video file writing is disabled.

**Suggested impact: high.** `_processed.mp4` encode/write is a major cost beside detect/track. Contract and engine already have `task_parameters.render_video` (default `true`; false → no-op renderer in `process`/`render`). Gap was **UI**: prescan confirm sent `render_video: true` fixed.

**Done (2026-08-22, Step 6.11 / S31):** Toggle in `prescan-review-modal.tsx` (default true); value on `PATCH /jobs/{id}/prescan` `task_parameters.render_video`. No schema change. Verified `test_video` with false → COMPLETED, no `_processed.mp4`.

### I007 — Track pedestrians toggle (promoted → 6.12)

**Dump:** UI checkbox in prescan review: track pedestrians true/false. Pass through API for processing to skip pedestrian detection when false. Ensure Pedestrian is not populated in `_15min.csv` (or `_events.csv`) when disabled. If API does not support this, add contract + orchestrator + engine + E2E.

**Suggested impact: high.** Engine always runs a second YOLO (`yolo11l.pt`) for pedestrians today (`ultralytics_detect.py`, `process.py`). Skipping it should improve FPS/wall time. **No existing job field** — unlike `render_video` (6.11). Needs schema-first: `task_parameters.track_pedestrians` (default `true`?) on prescan confirm / job submit; engine skips pedestrian predict + merge; aggregate must omit Pedestrian rows when false (S33 currently always aggregates Pedestrian). UI toggle in prescan review; verify `_15min.csv` + E2E on a clip with/without pedestrian crossings.

---

## Promoted / dropped

| ID | Date closed | Outcome | Notes |
|----|-------------|---------|-------|
| I001 | 2026-08-21 | **promoted** → **6.8** | Remove Live Monitor widget; Live Crossings in job details; row click opens details. |
| I002 | 2026-08-21 | **promoted** → **6.10** | Bind UI count to existing `crossing_count` (JobStatus / WS PROGRESS). No new API field. |
| I003 | 2026-08-21 | **promoted** → **6.9** | No in-process OSD OCR; lock clock/location to confirmed prescan. |
| I006 | 2026-08-21 | **promoted** → **6.11** | Prescan-review `render_video` toggle; pass existing confirm/submit field (no new schema). **Implemented 2026-08-22.** |
| I007 | 2026-08-22 | **promoted** → **6.12** | `track_pedestrians` checkbox in prescan review; new API field if missing; skip pedestrian YOLO + exclude from `_15min.csv` when false. |

---

## Log

| Date | Note |
|------|------|
| 2026-08-22 | I007 added + promoted → **6.12** — `track_pedestrians` toggle; skip pedestrian detect + exclude from `_15min.csv`; schema/API/engine/E2E if needed. |
| 2026-08-22 | **I006 / 6.11 complete** — prescan `render_video` toggle (default true); Confirm/Confirming…; `test_video` false skips `_processed.mp4`. |
| 2026-08-21 | **6.9 implemented** — process loop no longer OCR-rescans; clock/location locked to confirmed prescan/user metadata; S23 before/after documented. |
| 2026-08-22 | I004 note: 6.8 job details exist; still demoted until re-promoted |
| 2026-08-21 | File created. Manual-review parking lot; agents must not self-assign. |
| 2026-08-21 | I001–I003 restored on `main` (were only in the idea-dump chat). I004 added — play `_processed.mp4` from job details; investigate in-progress VOD vs complete-only. |
| 2026-08-21 | Review: **I001 → 6.8**, **I003 → 6.9**. I002 stays dump P3 (API availability check first). I004 demoted (aesthetics / later). |
| 2026-08-21 | I002 promoted → **6.10** (`crossing_count` confirmed on JobStatus and WS PROGRESS). I004 still demoted. |
| 2026-08-21 | I005 added — investigate crossing events as processed-video subtitles + performance. |
| 2026-08-21 | I005 conclusion — post-process `_events.csv` → soft SRT/VTT → mux into a single `_processed.mp4` (no re-encode / no burn-in). |
| 2026-08-21 | I006 added — prescan-review `render_video` toggle; pass through existing API (engine already supports false). |
| 2026-08-21 | I006 promoted → **6.11** (high impact; UI toggle on existing `render_video`). |
