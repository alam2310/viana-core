# Stabilization backlog (living)

**Rules:** [`STABILIZATION.md`](STABILIZATION.md)  
**Step 5 blocked while any blocker row is `open` or `in_progress`.**

> **Follow [`Execution path`](#execution-path) in Seq order.** One row = one unit of work. Do not skip ahead unless a dependency is `fixed` / `deferred`.

**Last updated:** 2026-08-19

---

## Summary

| Blockers open | Blockers fixed | Polish open | Path steps done |
|---------------|----------------|-------------|-----------------|
| 0 | 1 | 2 | 7 / 8 active |

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
| **S08** | F002 | C | no | — | Reduce prescan wall-clock (OCR works; intake still 30s+) | open |
| **S10** | F007 | C | no | S07 | Improve horizon + counting line proposal (CV / geometry) | open |
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
| **S08** | 1. Intake `hiv000001_inframe.mp4` 2. Time until `AWAITING_REVIEW` | **Expected:** prescan in a few seconds. **Actual:** still very long (30s+). **2026-08-19 retest:** corner OCR now populates time/date/location (S07 ✅) but wall-clock unchanged — triage dark-frame scan window, first-OSD frame search, EasyOCR cold start, subprocess overhead. | `prescan.py`, `configs/engine_defaults.yaml`, `ocr.py` (frame pick loop) | — | Step 4 UI chat |
| **S10** | 1. Intake `hiv000001_inframe.mp4` (or parity clip) 2. `AWAITING_REVIEW` 3. Open review modal | **Expected:** `proposed_lines` match road geometry (horizon near vanishing point, counting line on lane boundary) — usable without large edits. **Actual:** `propose_lines()` uses fixed normalized y-ratios or aspect-matched profile only (`lines.py` `geometric_lines`); no frame-based CV. Lines often misaligned vs operator “best” calibration (see `PARITY_NOTES.md` geometry B vs D). | `src/viana/stages/lines.py`, `prescan.py` (pass sampled frame), `tests/viana/test_prescan.py`; reference `legacy/` parity geometry only for bounds | — | — |

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

## Changelog

| Date | Change |
|------|--------|
| 2026-08-19 | S08 retest logged: OCR metadata OK but prescan latency still high (Step 4 UI chat) |
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
