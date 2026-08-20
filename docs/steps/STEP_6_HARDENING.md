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
| **6.2** | Pause / resume / PAUSED UX | UI (+ API) |
| **6.3** | Faster DELETE → CANCELLED | API |
| **6.4** | Browser / Playwright click-through | UI / QA |
| **6.5** | Extra camera clip validation | Engine / QA |
| **6.6** | GPU tests in CI | DevOps |
| **6.7** | Container host path access + API intake path validation (**S09 / F006**) | DevOps / API |

---

## Exit criteria

Mark each item ✅ in `TRACKER.md`. When all done or deferred, mark Step 6 ✅.

---

## Log

| Date | Note |
|------|------|
| 2026-08-20 | Stabilization **S21 (F017)** — adaptive OSD OCR (bands, clock salvage, mixed-polarity crops); UI retest OK. |
| 2026-08-20 | **6.1** follow-up — bake EasyOCR English CRAFT + `english_g2` at image build; `_run_prescan` marks `PRESCAN_FAILED` on `TimeoutExpired` instead of leaving the job running. |
| 2026-08-20 | **6.1** complete — Dockerfile re-pins `numpy>=1.26,<2` and installs `trackers==2.6.0 --no-deps` after the editable package; `docker-compose.yml` starts uvicorn only. Rebuild: `docker compose build`. |
| 2026-08-19 | Renumbered from Step 5; six-step plan |
| 2026-08-19 | **6.7** absorbs stabilization **S09** (F006 API intake path validation) |
| 2026-08-19 | **6.7** from Step 1 discovery (G21) |
