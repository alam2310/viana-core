# Step 6 — Hardening backlog

| Field | Value |
|-------|-------|
| **Status** | 🔄 In progress — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | Per item |
| **Blocked by** | Steps 1–5 (recommended) |

**On each item:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 6.

---

## Ordered backlog

| Item | Work | Chat |
|------|------|------|
| **6.1** | Bake `trackers` + `numpy<2` in Docker image | ✅ API / DevOps |
| **6.2** | Pause / resume / PAUSED UX | ✅ UI (+ API) |
| **6.3** | Faster DELETE → CANCELLED | ✅ API |
| **6.4** | Browser / Playwright click-through | UI / QA |
| **6.5** | Extra camera clip validation | Engine / QA |
| **6.6** | GPU tests in CI | DevOps |
| **6.7** | Container host path access + API intake path validation (**S09 / F006**) | ✅ DevOps / API |
| **6.8** | Job details instead of Live Monitor widget; Live Crossings in details; row click (**I001**) | ✅ UI |
| **6.9** | No OSD OCR during processing; lock time/date/location to confirmed prescan (**I003**) | ✅ Engine |
| **6.10** | Bind crossing total to existing `crossing_count` (JobStatus / WS PROGRESS), not session list length (**I002**) | ✅ UI |
| **6.11** | Prescan-review `render_video` toggle; wire existing confirm/submit field (**I006**) | ✅ UI |
| **6.12** | Prescan `track_pedestrians` toggle; skip pedestrian YOLO; exclude from `_15min.csv` when false; schema + API + E2E (**I007**) | UI + API + Engine |
| **6.13** | Project/intake UX re-discovery + redesign: camera-folder sources, analytics-type row, output widget; simplify `project_id` (**I008**) | UI / product |

Promoted from [`IDEA_DUMP.md`](IDEA_DUMP.md). Remaining dump items **I004** (demoted) and **I005** (mux soft subs) are **not** Step 6 work until re-promoted.

---

## Exit criteria

Mark each item ✅ in `TRACKER.md`. When all done or deferred, mark Step 6 ✅.

---

## Log

| Date | Note |
|------|------|
| 2026-08-22 | Promoted **I008 → 6.13** — project/intake UX re-discovery (Step 1-style discovery before build); camera-folder source flow; simplify `project_id`. |
| 2026-08-22 | **6.2 complete:** `POST /jobs/{id}/pause` (SIGINT → checkpoint → `PAUSED`); UI queue Pause/Resume, `PRESCAN_FAILED` Retry prescan (slot 2), Cancel (was Stop); resume `PAUSED` only; S30 refreshJobs; `test_s62_pause.py`. |
| 2026-08-22 | **6.11 complete (I006 / S31):** `render_video` checkbox in prescan review (default true); Confirm/Confirming…; no schema change. Verified `test_video` COMPLETED with false → no `_processed.mp4`. |
| 2026-08-22 | Doc sync: mark **6.8–6.10** ✅ in ordered backlog (already complete in TRACKER). |
| 2026-08-21 | Promoted **I006 → 6.11** (`render_video` toggle in prescan review; field already in contracts/engine). |
| 2026-08-21 | **6.7 complete (S09 / F006):** `POST /jobs/intake` and `POST /jobs` rewrite host prefixes (`VIANA_HOST_DATA_ROOT` → `/data`, repo → `/app/ViAna`, suffix `/data/` fallback) and **400** unmapped paths. Extra bind-mount: uncomment compose volume, set `VIANA_EXTRA_INTAKE_ROOT` + `VIANA_PATH_MAPS`. UI still maps via `container-paths.ts`. Tests: `tests/orchestrator/test_s67_intake_paths.py`. |
| 2026-08-21 | **6.3 complete:** `WorkerPool.cancel` sets `CANCELLED` on DELETE, releases GPU occupancy, SIGTERM/SIGKILL off the request thread; `_finalize` keeps user cancel (does not map checkpoint SIGTERM to PAUSED). Drain-after-fail (S27) unchanged. Tests: `tests/orchestrator/test_s63_cancel.py`. |
| 2026-08-21 | **6.9 complete (I003 / S23):** removed in-process OSD OCR; CSV uses confirmed clock; before/after on `hiv000001_inframe.mp4` 203.2s/13.45 fps → 179.3s/15.26 fps (detect remains the bulk). |
| 2026-08-21 | **6.8 + 6.10** complete — Live Monitor removed; Live Crossings in details while processing; count from `progress.crossing_count`. |
| 2026-08-21 | Promoted **I001 → 6.8**, **I003 → 6.9**. I002/I004 stay in idea dump. |
| 2026-08-20 | Stabilization **S21 (F017)** — adaptive OSD OCR (bands, clock salvage, mixed-polarity crops); UI retest OK. |
| 2026-08-20 | **6.1** follow-up — bake EasyOCR English CRAFT + `english_g2` at image build; `_run_prescan` marks `PRESCAN_FAILED` on `TimeoutExpired` instead of leaving the job running. |
| 2026-08-20 | **6.1** complete — Dockerfile re-pins `numpy>=1.26,<2` and installs `trackers==2.6.0 --no-deps` after the editable package; `docker-compose.yml` starts uvicorn only. Rebuild: `docker compose build`. |
| 2026-08-19 | Renumbered from Step 5; six-step plan |
| 2026-08-19 | **6.7** absorbs stabilization **S09** (F006 API intake path validation) |
| 2026-08-19 | **6.7** from Step 1 discovery (G21) |
