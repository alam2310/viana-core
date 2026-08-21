# Idea dump (manual review only)

**Status:** parking lot — **not** a work queue  
**Last updated:** 2026-08-21 (I001, I002, I003 promoted)  
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

Human review of I001–I004. **Agents: implement only promoted Step 6 items** (`6.8`, `6.9`, `6.10`), not remaining dump rows.

| Rank | ID | Decision |
|------|----|----------|
| 1 | **I001** | **Promoted** → Step **6.8** (UI) |
| 2 | **I003** | **Promoted** → Step **6.9** (engine) |
| 3 | **I002** | **Promoted** → Step **6.10** (UI) — `crossing_count` already on JobStatus and WS PROGRESS |
| 4 | **I004** | **Demoted** — later / aesthetics. Do not start from this dump. |

---

## Open ideas

| ID | Date | Category | Idea | Impact | Comment | Status |
|----|------|----------|------|--------|---------|--------|
| I004 | 2026-08-21 | `ui` | Play processed video from job details; investigate in-progress (play bytes already written, not live-edge) vs complete-only | **low** | **Demoted** (aesthetics / later). Not in Step 6. Revisit after 6.8 job details exist. | open |

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

---

## Promoted / dropped

| ID | Date closed | Outcome | Notes |
|----|-------------|---------|-------|
| I001 | 2026-08-21 | **promoted** → **6.8** | Remove Live Monitor widget; Live Crossings in job details; row click opens details. |
| I002 | 2026-08-21 | **promoted** → **6.10** | Bind UI count to existing `crossing_count` (JobStatus / WS PROGRESS). No new API field. |
| I003 | 2026-08-21 | **promoted** → **6.9** | No in-process OSD OCR; lock clock/location to confirmed prescan. |

---

## Log

| Date | Note |
|------|------|
| 2026-08-21 | **6.9 implemented** — process loop no longer OCR-rescans; clock/location locked to confirmed prescan/user metadata; S23 before/after documented. |
| 2026-08-21 | File created. Manual-review parking lot; agents must not self-assign. |
| 2026-08-21 | I001–I003 restored on `main` (were only in the idea-dump chat). I004 added — play `_processed.mp4` from job details; investigate in-progress VOD vs complete-only. |
| 2026-08-21 | Review: **I001 → 6.8**, **I003 → 6.9**. I002 stays dump P3 (API availability check first). I004 demoted (aesthetics / later). |
| 2026-08-21 | I002 promoted → **6.10** (`crossing_count` confirmed on JobStatus and WS PROGRESS). I004 still demoted. |
