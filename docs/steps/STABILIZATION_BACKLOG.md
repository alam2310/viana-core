# Stabilization backlog (living)

**Rules:** [`STABILIZATION.md`](STABILIZATION.md)  
**Step 5 blocked while any blocker row is `open` or `in_progress`.**

> **Follow [`Execution path`](#execution-path) in Seq order.** One row = one unit of work. Do not skip ahead unless a dependency is `fixed` / `deferred`.

**Last updated:** 2026-08-19

---

## Summary

| Blockers open | Blockers fixed | Polish open | Path steps done |
|---------------|----------------|-------------|-----------------|
| 1 | 0 | 5 | 1 / 9 |

---

## Execution path

Work **top to bottom**. **Depends** = prior Seq that must be `fixed` or `deferred` before starting.

| Seq | ID | Lane | Blocker | Depends | Title | Status |
|-----|-----|------|---------|---------|-------|--------|
| **S01** | F004 | B | no | — | Verify preview JPEG survives orchestrator restart | fixed |
| **S02** | F003 | B | no | — | Add `GET /artifacts/{job_id}/source.mp4` (Range, prescan-phase jobs) | open |
| **S03** | F003 | A | no | S02 | Next.js `/api/proxy/source` — same-origin video stream | open |
| **S04** | F003 | A | no | S03 | Prescan scrub: video seek → canvas; **no** `prescan/preview` on slider | open |
| **S05** | F003 | A/B | no | S04 | `prescan/preview` only for **Re-scan OCR**; docs sync (`api_contracts`, `COMPONENT_MAP`) | open |
| **S06** | F005 | C | no | — | Triage EasyOCR in container (installed? hits on test frame?) | open |
| **S07** | F001 | C | **yes** | S06 | Corner ROI OSD OCR → populated `proposed_metadata` | open |
| **S08** | F002 | C | no | — | Reduce first prescan latency (dark-frame scan / cold start) | open |
| **S09** | F006 | B | no | — | API rejects or normalizes container-unreadable intake paths | open |

**After S07 is `fixed` or `deferred` (approved):** Step 5 may start. S08–S09 are polish (may continue in parallel).

---

## Row detail

| Seq | Repro | Expected vs actual | Files / notes | Fix commit | Verified |
|-----|-------|-------------------|---------------|------------|----------|
| **S01** | 1. Prescan → `AWAITING_REVIEW` 2. Restart `viana_core` 3. Open review | **Expected:** `proposed_preview_url` JPEG still loads. **Actual:** in-memory registry empty → 404. | `src/orchestrator/preview_registry.py` — disk `rglob` fallback | disk rglob fallback (uncommitted) | Step 4 UI chat |
| **S02** | 1. Job in `AWAITING_REVIEW` 2. `GET /artifacts/{id}/source.mp4` with Range | **Expected:** streams `job.source_video_path` for browser seek (mirror G19 partial MP4). **Actual:** no source endpoint; scrub must spawn prescan. | `src/orchestrator/routes/artifacts.py`, `api_contracts.md`, orchestrator test | — | — |
| **S03** | 1. S02 deployed 2. Open review modal | **Expected:** browser loads source via same-origin proxy (like `/api/proxy/preview`). **Actual:** cross-origin or no URL. | `apps/web/src/app/api/proxy/source/route.ts`, `api-client.ts` | — | — |
| **S04** | 1. Open prescan review 2. Move frame-offset slider | **Expected:** frame updates in &lt;200ms from local video seek; lines unchanged. **Actual:** each scrub calls `GET /jobs/{id}/prescan/preview` → full `viana prescan` subprocess (OCR + lines + JPEG). | `prescan-review-modal.tsx`, `calibration-canvas.tsx` — hidden `<video>`, `seeked` → `drawImage`; remove offset→`prescanPreview` effect; `loadedmetadata` → `duration_sec` | — | — |
| **S05** | 1. Click **Re-scan OCR at Ns** | **Expected:** only explicit re-scan hits prescan API; slider does not. **Actual:** (after S04) re-scan still calls `prescanPreview`; metadata-only merge. Also: `applyToOthers` should use `job.proposed_*` / status, not `prescanPreview(0)` for resolution. | `prescan-review-modal.tsx`, `docs/api_contracts.md` § artifacts, `docs/ui/COMPONENT_MAP.md` | — | — |
| **S06** | 1. `docker exec` into API container 2. Run prescan on `hiv000001_inframe.mp4` 3. Inspect OCR stdout | **Expected:** EasyOCR installed; corner OSD yields hits. **Actual:** `optional_easyocr_reader()` may return `[]`; full-frame read misses small corner text. | `ocr.py`, `cli.py`, Docker image — outcome informs S07 scope | — | — |
| **S07** | 1. Intake `data/raw/hiv000001_inframe.mp4` 2. `AWAITING_REVIEW` 3. Open review | **Expected:** `proposed_metadata` has time (HH:MM:SS), date (DD-MM-YYYY), location from 1–2 corner ROIs. **Actual:** fields empty; operator types all metadata. **Gates Step 5.** | `src/viana/stages/ocr.py`, `prescan.py`, `time_map.py` | — | — |
| **S08** | 1. Intake one short clip 2. Time until `AWAITING_REVIEW` | **Expected:** prescan in a few seconds. **Actual:** 30s+ (dark-frame scan up to 30s + EasyOCR cold start). | `prescan.py`, `configs/engine_defaults.yaml` | — | — |
| **S09** | 1. POST intake with host path outside docker mounts (no UI translation) | **Expected:** API rejects or normalizes before prescan. **Actual:** `Video not found` at prescan. **UI mitigated:** `container-paths.ts`. Defer full fix to Step 6.7 if approved. | `pool.py`, `docker-compose.yml` | — | — |

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

## Changelog

| Date | Change |
|------|--------|
| 2026-08-19 | Merged F001–F006 + F003 scrub plan into single execution path S01–S09 |
| 2026-08-19 | F001–F006 logged from Step 4 acceptance testing (UI chat thread) |
| 2026-08-19 | Backlog created; Step 5 gated on blockers |

---

## Deferred / wontfix (optional)

| Seq | Reason | Date |
|-----|--------|------|
| — | | |
