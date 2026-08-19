# Step 5 — Hardening backlog

| Field | Value |
|-------|-------|
| **Status** | ⬜ Not started — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | Per item (API / UI / Engine / DevOps) |
| **Blocked by** | Steps 1–4 (recommended) |

**On each item:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 5.

---

## Ordered backlog

| Item | Work | Chat |
|------|------|------|
| **5.1** | Bake `trackers` + `numpy<2` in Docker image | API / DevOps |
| **5.2** | Pause / resume / PAUSED UX | UI (+ API) |
| **5.3** | Faster DELETE → CANCELLED | API |
| **5.4** | Browser / Playwright click-through | UI / QA |
| **5.5** | Extra camera clip validation | Engine / QA |
| **5.6** | GPU tests in CI | DevOps |
| **5.7** | Container read arbitrary host paths (local + mounted external HDD) | DevOps / API |

---

## Exit criteria (per item)

Mark ✅ in `TRACKER.md` and log here. When all items done or deferred, mark Step 5 ✅.

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Migrated from parked list; numbered Steps 1–5 |
| 2026-08-19 | Added **5.7** — container access to arbitrary host paths (external HDD) from Step 1 discovery |
